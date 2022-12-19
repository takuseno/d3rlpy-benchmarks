import argparse
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import uniform_filter1d

sns.set()


def plot(score_list, label):
    x = []
    y = []
    for path in score_list:
        data = np.loadtxt(path, delimiter=",")
        x.append(data[:, 1])
        y.append(uniform_filter1d(data[:, 2], size=10))
    x = np.array(x)
    y = np.array(y)
    sns.lineplot(x=x.reshape(-1), y=y.reshape(-1), label=label)


def collect_paths(directory, env):
    pattern = f"{directory}/SAC_online_n_1_{env}_.*/evaluation\.csv"
    return [path for path in glob.glob(f"{directory}/**", recursive=True) if re.search(pattern, path) is not None]


def collect_dist_paths(directory, env, dist_type):
    pattern = f"{directory}/SAC_online_{dist_type}_{env}_.*/evaluation\.csv"
    return [path for path in glob.glob(f"{directory}/**", recursive=True) if re.search(pattern, path) is not None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--save', type=str)
    args = parser.parse_args()

    plot(collect_paths("extra/multistep", args.env), "SAC")
    plot(collect_dist_paths("extra/dist", args.env, "qr"), "QR-SAC")
    plot(collect_dist_paths("extra/dist", args.env, "iqn"), "IQN-SAC")

    plt.title(args.env)
    plt.xlabel("Million Step")
    plt.xticks([0, 200000, 400000, 600000, 800000, 1000000],
               ["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    plt.xlim(0, 1000000)
    plt.ylabel("Average Return")
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
