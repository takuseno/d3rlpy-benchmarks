import glob
import os
import csv
import numpy as np


def format_float(score):
    return "{:.1f}".format(score)


def main():
    table = {}
    for log_dir in sorted(glob.glob("atari/*")):
        base = log_dir.split('/')[-1]
        splits = base.split('_')

        algo = splits[0]
        env = splits[1]

        if algo not in table:
            table[algo] = {}

        if env not in table[algo]:
            table[algo][env] = {"final": [], "best": []}

        with open(os.path.join(log_dir, 'environment.csv'), 'r') as f:
            reader = csv.reader(f)
            results = [list(map(float, row)) for row in reader]

        table[algo][env]["final"].append(float(results[-1][-1]))
        table[algo][env]["best"].append(float(np.max(np.array(results)[:, -1])))

    with open("atari_table.csv", "w") as f:
        writer = csv.writer(f)

        header = [
            "algo",
            "env",
            "final return",
            "final std",
            "best return",
            "best std",
        ]
        writer.writerow(header)

        for algo in table.keys():
            for env in table[algo].keys():
                final_returns = table[algo][env]["final"]
                final_avg = np.mean(final_returns)
                final_std = np.std(final_returns)
                best_returns = table[algo][env]["best"]
                best_avg = np.mean(best_returns)
                best_std = np.std(best_returns)
                row = [
                    algo,
                    env,
                    format_float(final_avg),
                    format_float(final_std),
                    format_float(best_avg),
                    format_float(best_std),
                ]
                print(row)
                writer.writerow(row)


if __name__ == '__main__':
    main()
