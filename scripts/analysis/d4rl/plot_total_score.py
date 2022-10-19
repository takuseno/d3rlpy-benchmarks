import argparse

import matplotlib.pyplot as plt
import numpy as np

from d3rlpy_benchmarks.data_loader import load_all_algos_d4rl_scores

ENVS = ["hopper", "walker2d", "halfcheetah"]
DATASETS = ["random-v0", "medium-v0", "medium-replay-v0", "medium-expert-v0"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    labels = []
    total_scores = {}
    for env in ENVS:
        for dataset in DATASETS:
            scores = load_all_algos_d4rl_scores(env, dataset, exclude=["CRR"])
            for score in scores:
                if score.algo not in labels:
                    labels.append(score.algo)
                    total_scores[score.algo] = score.scores[:, -1]
                else:
                    total_scores[score.algo] += score.scores[:, -1]

    means = []
    stds = []
    for algo in labels:
        means.append(np.mean(total_scores[algo]))
        stds.append(np.std(total_scores[algo]))

    x_values = np.arange(len(labels))

    plt.bar(x_values, means, yerr=stds, color="b", ecolor="black", width=0.3, align="center", capsize=5)
    plt.xticks(x_values, labels)
    plt.ylabel("Total Normalized Score")

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
