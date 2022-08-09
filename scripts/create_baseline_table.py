import glob
import os
import csv

import numpy as np


EXPERT_SCORES = {
    "halfcheetah": 12135.0,
    "hopper": 3234.3,
    "walker2d": 4592.3,
}

# calculated from d4rl paper
BASE_RANDOM = {
    "halfcheetah": -280.178953,
    "hopper": -20.272305,
    "walker2d": 1.629008,
}

TASKS = ["halfcheetah", "hopper", "walker2d"]

DATASET_TYPES = ["random", "medium", "medium_replay", "medium_expert"]

ALGOS = ["cql", "bear"]


def compute_normalized_score(raw_score, env):
    expert_score = EXPERT_SCORES[env]
    random_score = BASE_RANDOM[env]
    return (raw_score - random_score) / (expert_score - random_score)


def main():
    table = {}
    for algo in ALGOS:
        table[algo] = {}
        for task in TASKS:
            table[algo][task] = {}
            for dataset_type in DATASET_TYPES:
                base_path = os.path.join("baselines", algo, task, dataset_type)
                table[algo][task][dataset_type] = []
                for log_path in glob.glob(f"{base_path}/**/**/**/progress.csv"):
                    with open(log_path, "r") as f:
                        reader = csv.reader(f)
                        results = [row for row in reader]
                    header = results[0]
                    index = header.index("evaluation/Average Returns")
                    table[algo][task][dataset_type].append(float(results[-1][index]))

    with open("baseline_table.csv", "w") as f:
        writer = csv.writer(f)

        header = ["algo", "env", "dataset", "return", "std", "normalized return", "normalized std"]
        writer.writerow(header)

        for algo in table.keys():
            for env in table[algo].keys():
                for dataset in table[algo][env]:
                    returns = table[algo][env][dataset]
                    avg = sum(returns) / len(returns)
                    std = np.std(returns)
                    normalized_score = 100.0 * compute_normalized_score(avg, env)
                    normalized_std = 100.0 * compute_normalized_score(std, env)
                    row = [algo, env, dataset, avg, std, normalized_score, normalized_std]
                    print(row)
                    writer.writerow(row)


if __name__ == '__main__':
    main()
