import argparse

import matplotlib.pyplot as plt
import numpy as np

from d3rlpy_benchmarks.data_loader import load_all_algos_d4rl_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--last_num", type=int, default=1)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    scores = load_all_algos_d4rl_scores(args.env, args.dataset, exclude=["CRR"])

    # compute topN and bottomN scores
    max_means = []
    min_means = []
    max_stds = []
    min_stds = []
    labels = []
    for score in scores:
        max_means.append(np.mean(np.max(score.scores[:, -args.last_num :], axis=1)))
        min_means.append(np.mean(np.min(score.scores[:, -args.last_num :], axis=1)))
        max_stds.append(np.std(np.max(score.scores[:, -args.last_num :], axis=1)))
        min_stds.append(np.std(np.min(score.scores[:, -args.last_num :], axis=1)))
        labels.append(score.algo)
    x_values = np.arange(len(labels))

    plt.bar(
        x_values,
        min_means,
        yerr=min_stds,
        color="b",
        ecolor="black",
        width=0.3,
        align="center",
        capsize=5,
        label=f"Mean Min in Last {args.last_num}",
    )
    plt.bar(
        x_values + 0.3,
        max_means,
        yerr=max_stds,
        color="r",
        ecolor="black",
        width=0.3,
        align="center",
        capsize=5,
        label=f"Mean Max in Last {args.last_num}",
    )
    plt.xticks(x_values + 0.3 / 2, labels)
    plt.ylabel("Normalized Score")
    plt.legend()

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
