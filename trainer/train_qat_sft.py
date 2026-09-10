"""Dense MiniMind SFT trainer with SPEC-Q4 fake-quantized FFN projections."""

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from contextlib import nullcontext
from torch import optim

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import torch
from transformers import AutoTokenizer

from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quantization.qat import QATLinear, QAT_TENSOR_COUNT, QAT_WEIGHT_COUNT, apply_spec_q4_qat, spec_q4_fake_dequant
from trainer.sft_plan import build_deterministic_sft_loader, build_sft_index_plan
from trainer.trainer_utils import Logger, get_lr, setup_seed
from trainer.q4_t3_metrics import (
    MetricsCollector,
    check_finite_gradient_norm,
    check_finite_loss,
    finish_parent_delta_evidence,
    summarize_weight_differences,
)

warnings.filterwarnings('ignore')

metrics = None

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _save_fp32_state_dict(model, path):
    state_dict = {}
    for key, value in model.state_dict().items():
        if value.is_floating_point():
            state_dict[key] = value.detach().to(dtype=torch.float32, device='cpu').clone()
        else:
            state_dict[key] = value.detach().cpu().clone()
    temporary_path = path + '.tmp'
    torch.save(state_dict, temporary_path)
    os.replace(temporary_path, path)


def _clip_gradients(model, max_norm):
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    check_finite_gradient_norm(grad_norm, stage='qat', collector=metrics)
    if metrics:
        metrics.record_gradient_norm(grad_norm)


def _train_epoch(epoch, loader, iters, model, optimizer, scaler, autocast_ctx, args):
    if metrics:
        metrics.start(args.epochs * iters)
    start_time = time.time()
    last_step = 0
    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        batch_started = time.perf_counter()
        with autocast_ctx:
            result = model(input_ids, labels=labels)
            loss = result.loss + result.aux_loss
            check_finite_loss(loss, stage='qat', collector=metrics)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
        current_loss = None
        if metrics or step % args.log_interval == 0 or step == iters:
            current_loss = loss.item() * args.accumulation_steps
        if metrics:
            metrics.record_batch(
                current_loss,
                labels=labels,
                duration_seconds=time.perf_counter() - batch_started,
            )
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            _clip_gradients(model, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if metrics:
                metrics.record_optimizer_step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            elapsed = time.time() - start_time
            aux_loss = result.aux_loss.item() if result.aux_loss is not None else 0.0
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_loss - aux_loss:.4f}, '
                f'aux_loss: {aux_loss:.4f}, lr: {optimizer.param_groups[-1]["lr"]:.8f}, '
                f'epoch_time: {elapsed / max(step, 1) * (iters - step) / 60:.1f}min'
            )
        del input_ids, labels, result, loss

    if last_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        _clip_gradients(model, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if metrics:
            metrics.record_optimizer_step()
        optimizer.zero_grad(set_to_none=True)


def _qat_weight_keys(model):
    return tuple(
        f'{name}.weight'
        for name, module in model.named_modules()
        if isinstance(module, QATLinear)
    )


def _qat_quantization_error(model, keys):
    modules = dict(model.named_modules())
    return summarize_weight_differences(
        (
            modules[key[:-len('.weight')]].weight,
            spec_q4_fake_dequant(modules[key[:-len('.weight')]].weight),
        )
        for key in keys
    )


def main():
    global metrics
    parser = argparse.ArgumentParser(description='MiniMind SPEC-Q4 QAT SFT')
    parser.add_argument('--save_dir', type=str, default='../out')
    parser.add_argument('--save_weight', type=str, default='qat_sft')
    parser.add_argument('--parent_path', type=str, default='../out/pretrain_768.pth')
    parser.add_argument('--tokenizer_path', type=str, default='../model')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=768)
    parser.add_argument('--use_moe', type=int, default=0, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--holdout_seed', type=int, default=4242)
    parser.add_argument('--holdout_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default='../dataset/sft_t2t_mini.jsonl')
    parser.add_argument('--metrics_path', type=str, default=None)
    parser.add_argument('--expected_plan_sha256', type=str, default=None)
    parser.add_argument('--expected_holdout_sha256', type=str, default=None)
    args = parser.parse_args()

    if args.metrics_path:
        metrics = MetricsCollector(
            'qat',
            config={key: getattr(args, key) for key in (
                'epochs', 'batch_size', 'learning_rate', 'dtype', 'num_workers',
                'accumulation_steps', 'max_seq_len', 'use_moe', 'seed',
            )},
            metrics_path=args.metrics_path,
        )

    if args.use_moe:
        raise ValueError('SPEC-Q4 QAT requires dense use_moe=0')
    setup_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = MiniMindForCausalLM(lm_config)
    parent_path = os.path.abspath(args.parent_path)
    parent_state_dict = torch.load(parent_path, map_location='cpu')
    model.load_state_dict(parent_state_dict, strict=True)
    parent_sha256 = _sha256_file(parent_path)
    model = model.to(args.device)

    apply_spec_q4_qat(model)
    qat_tensor_count = sum(isinstance(module, QATLinear) for module in model.modules())
    qat_weight_count = sum(module.weight.numel() for module in model.modules() if isinstance(module, QATLinear))
    if (qat_tensor_count, qat_weight_count) != (QAT_TENSOR_COUNT, QAT_WEIGHT_COUNT):
        raise RuntimeError(
            f'unexpected QAT targets: tensors={qat_tensor_count}, weights={qat_weight_count}'
        )
    Logger(f'SPEC-Q4 QAT targets: {qat_tensor_count} tensors, {qat_weight_count} weights')

    qat_weight_keys = _qat_weight_keys(model)
    if len(qat_weight_keys) != QAT_TENSOR_COUNT:
        raise RuntimeError(f'unexpected QAT weight keys: {len(qat_weight_keys)}')
    parent_ffn_weights = {
        key: parent_state_dict[key].detach().to(dtype=torch.float32, device='cpu').clone()
        for key in qat_weight_keys
    }
    quantization_error_start = None
    if metrics:
        quantization_error_start = _qat_quantization_error(model, qat_weight_keys)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len, seed=args.seed)
    plan = build_sft_index_plan(
        args.data_path,
        len(train_ds),
        train_seed=args.seed,
        holdout_seed=args.holdout_seed,
        holdout_size=args.holdout_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
    )
    if args.expected_plan_sha256 and plan.plan_sha256 != args.expected_plan_sha256:
        raise RuntimeError(f'index plan SHA mismatch: {plan.plan_sha256} != {args.expected_plan_sha256}')
    if args.expected_holdout_sha256 and plan.holdout_sha256 != args.expected_holdout_sha256:
        raise RuntimeError(f'holdout SHA mismatch: {plan.holdout_sha256} != {args.expected_holdout_sha256}')
    Logger(f'index plan SHA-256: {plan.plan_sha256}')
    Logger(f'holdout SHA-256: {plan.holdout_sha256} ({len(plan.holdout_indices)} indices)')

    compact_plan = plan.to_compact_dict()

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_source_count = 0
    train_padded_count = 0
    train_processed_count = 0
    for epoch in range(args.epochs):
        setup_seed(args.seed + epoch)
        loader = build_deterministic_sft_loader(
            train_ds,
            plan,
            epoch=epoch,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        train_source_count += loader.sft_num_source_samples
        train_padded_count += loader.sft_num_padded_samples
        train_processed_count += loader.sft_num_processed_samples
        try:
            _train_epoch(epoch, loader, len(loader), model, optimizer, scaler, autocast_ctx, args)
        except Exception:
            if metrics:
                metrics.finish(extra={'training_failed': True})
            raise

    if metrics:
        training_duration = metrics.stop()
        quantization_error_end = _qat_quantization_error(model, qat_weight_keys)
        master_weights = {
            key: model.state_dict()[key]
            for key in qat_weight_keys
        }
        finish_parent_delta_evidence(
            metrics,
            master_weights,
            parent_ffn_weights,
            qat_weight_keys,
            extra={
                'quantization_error_start': quantization_error_start,
                'quantization_error_end': quantization_error_end,
                'slowdown_vs_control': None,
                'memory_overhead_vs_control': None,
                'comparison_fields_unavailable_reason': 'CONTROL baseline not supplied to trainer',
            },
            duration_seconds=training_duration,
        )
    else:
        finish_parent_delta_evidence(
            None,
            {key: model.state_dict()[key] for key in qat_weight_keys},
            parent_ffn_weights,
            qat_weight_keys,
        )

    metadata = {
        'trainer': 'qat',
        'parent_path': parent_path,
        'parent_sha256': parent_sha256,
        'qat_tensor_count': qat_tensor_count,
        'qat_weight_count': qat_weight_count,
        'checkpoint_state_dtype': 'float32',
        **compact_plan,
        'train_source_count': train_source_count,
        'train_padded_count': train_padded_count,
        'train_processed_count': train_processed_count,
        'holdout_source_count': len(plan.holdout_indices),
        'holdout_padded_count': 0,
        'holdout_processed_count': 0,
    }
    metadata_path = os.path.join(args.save_dir, f'{args.save_weight}_{lm_config.hidden_size}_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=True, indent=2)

    checkpoint_path = os.path.join(args.save_dir, f'{args.save_weight}_{lm_config.hidden_size}.pth')
    _save_fp32_state_dict(model, checkpoint_path)


if __name__ == '__main__':
    main()
