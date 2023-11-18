import csv

import numpy as np

from d3rlpy_benchmarks.data_loader import get_d4rl_algo_list, load_d4rl_score

ENVS = sorted(["halfcheetah", "walker2d", "hopper", "ant"])
DATASETS = sorted(
    [
        "random-v0",
        "medium-v0",
        "medium-replay-v0",
        "medium-expert-v0",
        "random-v2",
        "medium-v2",
        "medium-replay-v2",
        "medium-expert-v2",
    ]
)


def format_float(score):
    return "{:.1f}".format(score)


def main():
    with open("d4rl_table.csv", "w") as f:
        writer = csv.writer(f)

        header = [
            "algo",
            "env",
            "dataset",
            "final return",
            "final std",
            "final normalized return",
            "final normalized std",
            "best return",
            "best std",
            "best normalized return",
            "best normalized std",
        ]
        writer.writerow(header)

        for algo in get_d4rl_algo_list():
            for env in ENVS:
                for dataset in DATASETS:
                    score = load_d4rl_score(algo, env, dataset)
                    if score is None:
                        continue
                    final_returns = score.raw_scores[:, -1]
                    final_normalized_returns = score.scores[:, -1]
                    final_avg = np.mean(final_returns)
                    final_std = np.std(final_returns)
                    final_normalized_avg = np.mean(final_normalized_returns)
                    final_normalized_std = np.std(final_normalized_returns)
                    best_returns = np.max(score.raw_scores, axis=1)
                    best_normalized_returns = np.max(score.scores, axis=1)
                    best_avg = np.mean(best_returns)
                    best_std = np.std(best_returns)
                    best_normalized_avg = np.mean(best_normalized_returns)
                    best_normalized_std = np.std(best_normalized_returns)
                    row = [
                        score.algo,
                        env,
                        dataset,
                        format_float(final_avg),
                        format_float(final_std),
                        format_float(final_normalized_avg),
                        format_float(final_normalized_std),
                        format_float(best_avg),
                        format_float(best_std),
                        format_float(best_normalized_avg),
                        format_float(best_normalized_std),
                    ]
                    print(row)
                    writer.writerow(row)


if __name__ == "__main__":
    main()
