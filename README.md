# d3rlpy-benchmarks

This repository contains the benchmark results of d3rlpy.

Library repository: https://github.com/takuseno/d3rlpy

## baselines
The `baselines` directory contains the training results of the official implementations.
You can see the summary of the results at `baseline_table.csv`.

## reproductions
### D4RL
The `d4rl` directory contains the training results of the d3rlpy's benchmark scripts with D4RL.
You can see the summary of the results at `d4rl_table.csv`.

### Atari 2600
The `atari` directory contains the training results of the d3rlpy's benchmark scripts with Atari 2600 dataset.
You can see the summary of the results at `atari_table.csv`.


## analysis tools
### installation
```
$ pip install -e .
```

### library
This repository provides lightweight analysis tools for researchers to conduct the further analysis.
Here is the example snippet:

```py
import matplotlib.pyplot as plt
from d3rlpy_benchmarks.data_loader import load_d4rl_score

score = load_d4rl_score("CQL", "hopper", "medium-v0")
plt.plot(score.steps[0], np.mean(score.scores, axis=0))
plt.show()
```

There are ready-to-go analysis scripts in `scripts/analysis` directory.
```
$ python scripts/analysis/d4rl/plot_curve.py --env hopper --dataset medium-v0 --window 100
```


### coding style
```
$ pip install black isort mypy
$ python scripts/utils/static_check.py
```
