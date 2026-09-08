cmake_minimum_required(VERSION 3.24)
project(minimind_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(mm_compile_options target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /EHsc /fp:precise)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

add_library(minimind_cpu SHARED src/runtime.cpp src/kernels.cpp src/parallel.cpp)
target_include_directories(minimind_cpu PUBLIC include)
target_compile_definitions(minimind_cpu PRIVATE MM_RUNTIME_BUILD)
mm_compile_options(minimind_cpu)

include(CTest)
if(BUILD_TESTING)
  add_executable(mm_test_kernels tests/test_kernels.cpp src/kernels.cpp)
  target_include_directories(mm_test_kernels PRIVATE include)
  mm_compile_options(mm_test_kernels)
  add_test(NAME mm_kernels COMMAND mm_test_kernels)

  # Standalone CPU-E1 benchmark; it is not connected to the runtime or CTest.
  add_executable(mm_bench_cpu_e1 benchmarks/cpu-e1/kernel_bench.cpp src/kernels.cpp)
  target_include_directories(mm_bench_cpu_e1 PRIVATE include)
  mm_compile_options(mm_bench_cpu_e1)

  add_executable(mm_test_runtime tests/test_runtime.cpp)
  target_link_libraries(mm_test_runtime PRIVATE minimind_cpu)
  mm_compile_options(mm_test_runtime)
  add_test(NAME mm_runtime COMMAND mm_test_runtime)

  # Tests use assertions as executable checks, including in Release builds.
  foreach(test_target mm_test_kernels mm_test_runtime)
    if(MSVC)
      target_compile_options(${test_target} PRIVATE /UNDEBUG)
    else()
      target_compile_options(${test_target} PRIVATE -UNDEBUG)
    endif()
  endforeach()
endif()
