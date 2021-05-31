# Baseline Results
This directory contains the training results of the official implementations.

## CQL
repository: https://github.com/aviralkumar2907/CQL
commit: d67dbe9cf5d2b96e3b462b6146f249b3d6569796

### command
Basically, the command follows the instruction in their README. For medium datasets,
```
$ python examples/cql_mujoco_new.py --env halfcheetah-medium-v0 --policy_lr=1e-4 --seed=1 --lagrange_thresh=-1.0  --min_q_weight=10.0 --min_q_version=3 --gpu 0
```

For other datasets,
```
$ python examples/cql_mujoco_new.py --env halfcheetah-random-v0 --policy_lr=1e-4 --seed=1 --lagrange_thresh=-1.0  --min_q_weight=5.0 --min_q_version=3 --gpu 0
```
