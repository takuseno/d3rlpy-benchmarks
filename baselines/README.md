# Baseline Results
This directory contains the training results of the official implementations.

## BEAR
- repository: https://github.com/rail-berkeley/d4rl_evaluations
- commit: c93b9d2bf443cd5b8c43542fca9937dc50e630c3

### command
Basically, the command follows the instruction in their README.

- Hopper: `kernel_type=laplacian`, `mmd_sigma=20`, `num_samples=100`
- Walker2d: `kernel_type=laplacian`, `mmd_sigma=20`, `num_samples=100`
- HalfCheetah: `kernel_type=gaussian`, `mmd_sigma=20`, `num_samples=100`

```
$ python examples/bear_hdf5_d4rl.py --env=walker2d-medium-expert-v0 --policy_lr=1e-4 --num_samples=100 --kernel_type=laplacian --mmd_sigma=20 --seed 1 --gpu 0
```

## CQL
- repository: https://github.com/aviralkumar2907/CQL
- commit: d67dbe9cf5d2b96e3b462b6146f249b3d6569796

### command
Basically, the command follows the instruction in their README. For medium datasets,
```
$ python examples/cql_mujoco_new.py --env halfcheetah-medium-v0 --policy_lr=1e-4 --seed=1 --lagrange_thresh=-1.0  --min_q_weight=10.0 --min_q_version=3 --gpu 0
```

For other datasets,
```
$ python examples/cql_mujoco_new.py --env halfcheetah-random-v0 --policy_lr=1e-4 --seed=1 --lagrange_thresh=-1.0  --min_q_weight=5.0 --min_q_version=3 --gpu 0
```

## IQL
- repository: https://github.com/ikostrikov/implicit_q_learning
- commit: 09d700248117881a75cb21f0adb95c6c8a694cb2

### command
Basically, the command follows the instruction in their README.
```
$ python train_offline.py --env_name=halfcheetah-medium-expert-v0 --config=configs/mujoco_config.py --max_steps=500000
```
