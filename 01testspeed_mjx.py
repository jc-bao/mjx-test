from typing import Sequence

from absl import app
from absl import flags
from etils import epath
import mujoco
from mujoco import mjx
import pandas as pd

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=true'

_MJCF = flags.DEFINE_string('mjcf', None, 'path to model `.xml` or `.mjb`', required=True)
_BASE_PATH = flags.DEFINE_string(
    'base_path', None, 'base path, defaults to mujoco.mjx resource path'
)
_NSTEP = flags.DEFINE_integer('nstep', 1000, 'number of steps per rollout')
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 1024, 'number of parallel rollouts'
)
_UNROLL = flags.DEFINE_integer('unroll', 1, 'loop unroll length')
_SOLVER = flags.DEFINE_enum(
    'solver', 'cg', ['cg', 'newton'], 'constraint solver'
)
_ITERATIONS = flags.DEFINE_integer(
    'iterations', 1, 'number of solver iterations'
)
_LS_ITERATIONS = flags.DEFINE_integer(
    'ls_iterations', 4, 'number of linesearch iterations'
)
_OUTPUT = flags.DEFINE_enum(
    'output', 'text', ['text', 'tsv', 'csv'], 'format to print results'
)


def _main(argv: Sequence[str]):
  """Runs testpeed function."""
  path = epath.resource_path('mujoco.mjx') / 'test_data'
  path = _BASE_PATH.value or path
  f = epath.Path(path) / _MJCF.value
  if f.suffix == '.mjb':
    m = mujoco.MjModel.from_binary_path(f.as_posix())
  else:
    m = mujoco.MjModel.from_xml_path(f.as_posix())

  print(f'Rolling out {_NSTEP.value} steps at dt = {m.opt.timestep:.3f}...')
  jit_time, run_time, steps = mjx.benchmark(
      m,
      _NSTEP.value,
      _BATCH_SIZE.value,
      _UNROLL.value,
      _SOLVER.value,
      _ITERATIONS.value,
      _LS_ITERATIONS.value,
  )

  name = argv[0]
  if _OUTPUT.value == 'text':
    print(f"""
Summary for {_BATCH_SIZE.value} parallel rollouts

 Total JIT time: {jit_time:.2f} s
 Total simulation time: {run_time:.2f} s
 Total steps per second: { steps / run_time:.0f}
 Total realtime factor: { steps * m.opt.timestep / run_time:.2f} x
 Total time per step: { 1e6 * run_time / steps:.2f} µs
 Single realtime factor: {steps * m.opt.timestep / run_time / _BATCH_SIZE.value:.2f} x
 """)
  elif _OUTPUT.value == 'tsv':
    name = _MJCF.value.split('/')[-1]
    print(f'model: {name}\tjit: {jit_time:.2f}s\tfps: {steps / run_time:.0f}\tsingle_fps: {steps / run_time / _BATCH_SIZE.value:.0f}\tsingle_realtime_factor: {steps * m.opt.timestep / run_time / _BATCH_SIZE.value:.2f}x')
  elif _OUTPUT.value == 'csv':
    name = _MJCF.value.split('/')[-1].split('.')[0]
    df_new = pd.DataFrame({
      'model': [name],
      'jit': [jit_time],
      'fps': [steps / run_time],
      'single_fps': [steps / run_time / _BATCH_SIZE.value],
      'single_realtime_factor': [steps * m.opt.timestep / run_time / _BATCH_SIZE.value],
      'solver': [_SOLVER.value],
      'iterations': [_ITERATIONS.value],
      'ls_iterations': [_LS_ITERATIONS.value],
      'batch_size': [_BATCH_SIZE.value],
      'unroll': [_UNROLL.value],
    })
    # check if file exists
    filename = "mjx_speed_test.csv"
    if epath.Path(filename).exists():
      df = pd.read_csv(filename)
      df = pd.concat([df, df_new], ignore_index=True)
    else:
      df = df_new
    df.to_csv(filename, index=False)


def main():
  app.run(_main)


if __name__ == '__main__':
  main()
