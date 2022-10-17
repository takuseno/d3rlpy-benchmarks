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
    smooth_means = []
    final_means = []
    smooth_stds = []
    final_stds = []
    labels = []
    for score in scores:
        smooth_means.append(np.mean(score.scores[:, -args.last_num :]))
        final_means.append(np.mean(score.scores[:, -1]))
        smooth_stds.append(np.std(score.scores[:, -args.last_num :]))
        final_stds.append(np.std(score.scores[:, -1]))
        labels.append(score.algo)
    x_values = np.arange(len(labels))

    plt.bar(
        x_values,
        smooth_means,
        yerr=smooth_stds,
        color="b",
        ecolor="black",
        width=0.3,
        align="center",
        capsize=5,
        label=f"Last {args.last_num}",
    )
    plt.bar(
        x_values + 0.3,
        final_means,
        yerr=final_stds,
        color="r",
        ecolor="black",
        width=0.3,
        align="center",
        capsize=5,
        label=f"Final",
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
