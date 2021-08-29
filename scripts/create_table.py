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


def compute_normalized_score(raw_score, env):
    expert_score = EXPERT_SCORES[env]
    random_score = BASE_RANDOM[env]
    return (raw_score - random_score) / (expert_score - random_score)


def main():
    table = {}
    for log_dir in sorted(glob.glob("reproductions/*")):
        base = log_dir.split('/')[-1]
        splits = base.split('_')

        algo = splits[0]
        env = splits[1].split('-')[0]
        dataset = '-'.join(splits[1].split('-')[1:])

        if algo not in table:
            table[algo] = {}

        if env not in table[algo]:
            table[algo][env] = {}

        if dataset not in table[algo][env]:
            table[algo][env][dataset] = {"final": [], "best": []}

        with open(os.path.join(log_dir, 'environment.csv'), 'r') as f:
            reader = csv.reader(f)
            results = [list(map(float, row)) for row in reader]

        table[algo][env][dataset]["final"].append(float(results[-1][-1]))
        table[algo][env][dataset]["best"].append(float(np.max(np.array(results)[:, -1])))

    with open("table.csv", "w") as f:
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

        for algo in table.keys():
            for env in table[algo].keys():
                for dataset in table[algo][env]:
                    final_returns = table[algo][env][dataset]["final"]
                    final_normalized_returns = list(map(lambda v: compute_normalized_score(v, env), final_returns))
                    final_avg = np.mean(final_returns)
                    final_std = np.std(final_returns)
                    final_normalized_avg = np.mean(final_normalized_returns)
                    final_normalized_std = np.std(final_normalized_returns)
                    best_returns = table[algo][env][dataset]["best"]
                    best_normalized_returns = list(map(lambda v: compute_normalized_score(v, env), best_returns))
                    best_avg = np.mean(best_returns)
                    best_std = np.std(best_returns)
                    best_normalized_avg = np.mean(best_normalized_returns)
                    best_normalized_std = np.std(best_normalized_returns)
                    row = [
                        algo,
                        env,
                        dataset,
                        final_avg,
                        final_std,
                        100.0 * final_normalized_avg,
                        100.0 * final_normalized_std,
                        best_avg,
                        best_std,
                        100.0 * best_normalized_avg,
                        100.0 * best_normalized_std,
                    ]
                    print(row)
                    writer.writerow(row)


if __name__ == '__main__':
    main()
