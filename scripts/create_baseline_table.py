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


def compute_denormalized_score(score, env):
    expert_score = EXPERT_SCORES[env]
    random_score = BASE_RANDOM[env]
    return score * (expert_score - random_score) + random_score


def get_iql_scores():
    table = {}
    for env in TASKS:
        table[env] = {}
        for dataset_type in DATASET_TYPES:
            table[env][dataset_type] = []
            for seed in range(3):
                file_name = f"{env}-{dataset_type.replace('_', '-')}-v0_{seed+1}.txt"
                path = os.path.join('baselines', 'iql', file_name)
                with open(path, "r") as f:
                    data = np.loadtxt(f, delimiter=" ")
                table[env][dataset_type].append(data[-1, 1])
    return table


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

    # add iql
    table["iql"] = get_iql_scores()

    with open("baseline_table.csv", "w") as f:
        writer = csv.writer(f)

        header = ["algo", "env", "dataset", "return", "std", "normalized return", "normalized std"]
        writer.writerow(header)

        for algo in table.keys():
            for env in table[algo].keys():
                for dataset in table[algo][env]:
                    if algo == "iql":
                        normalized_score = np.mean(table[algo][env][dataset])
                        normalized_std = np.std(table[algo][env][dataset])
                        avg = compute_denormalized_score(normalized_score / 100.0, env)
                        std = compute_denormalized_score(normalized_std / 100.0, env)
                    else:
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
