import argparse

import matplotlib.pyplot as plt

from d3rlpy_benchmarks.data_loader import load_all_algos_d4rl_scores
from d3rlpy_benchmarks.plot_utils import plot_score_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--window", default=100, type=int)
    parser.add_argument("--save", type=str)
    args = parser.parse_args()

    scores = load_all_algos_d4rl_scores(args.env, args.dataset, exclude=["CRR"])
    for score in scores:
        plot_score_curve(score, window_size=args.window)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
