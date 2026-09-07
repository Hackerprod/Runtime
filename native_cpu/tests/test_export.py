import json, struct, unittest
from pathlib import Path
import uuid
import numpy as np
from native_cpu.tools.export_weights import ExportError, _atomic_write, quantize_q4, validate_config, _tensor_record

class ExportTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('checkpoints/minimind-3-hf/config.json').read_text())
    def test_official_config_contract(self):
        validate_config(self.config)
        bad = dict(self.config); bad['rope_scaling'] = {'type':'linear'}
        with self.assertRaises(ExportError): validate_config(bad)
    def test_tensor_header_contract_is_dims_before_group(self):
        record, _ = _tensor_record("model.norm.weight", np.ones((2, 3), dtype=np.float32), "fp32")
        n = struct.unpack_from("<I", record, 0)[0]; off = 4 + n
        dtype, rank = struct.unpack_from("<II", record, off); off += 8
        dims = struct.unpack_from("<2I", record, off); off += 8
        group = struct.unpack_from("<I", record, off)[0]
        self.assertEqual((dtype, rank, dims, group), (0, 2, (2, 3), 0))

    def test_q4_nibbles_scales_and_odd_padding(self):
        a = np.array([[0., 1., -1., 3., -3.]], dtype=np.float32)
        packed, scales = quantize_q4(a)
        self.assertEqual(len(packed), 3)
        # maxabs=3 => scale=3/7; values round to [0,2,-2,7,-7] + 8.
        self.assertEqual(list(packed), [0xA8, 0xF6, 0x81])
        self.assertAlmostEqual(float(scales[0]), 3/7, places=6)
    def test_q4_zero_group_is_finite_unit_scale(self):
        packed, scales = quantize_q4(np.zeros((1, 32), dtype=np.float32))
        self.assertEqual(scales.tolist(), [1.0]); self.assertEqual(packed, bytes([0x88])*16)
    def test_atomic_write_refuses_overwrite(self):
        d = Path('outputs') / ('test-export-' + uuid.uuid4().hex); d.mkdir(parents=True, exist_ok=False)
        try:
            p = d/'x.bin'; _atomic_write(p, b'a')
            with self.assertRaises(FileExistsError): _atomic_write(p, b'b')
        finally:
            for child in d.iterdir(): child.unlink()
            d.rmdir()

if __name__ == '__main__': unittest.main()
