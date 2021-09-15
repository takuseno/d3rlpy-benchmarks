import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

SAC = {
    "Hopper-v2": [
        "extra/multistep/SAC_online_n_1_Hopper-v2_1_20210907092100/evaluation.csv",
        "extra/multistep/SAC_online_n_1_Hopper-v2_2_20210907121407/evaluation.csv",
        "extra/multistep/SAC_online_n_1_Hopper-v2_3_20210907150406/evaluation.csv",
    ],
    "HalfCheetah-v2": [
        "extra/multistep/SAC_online_n_1_HalfCheetah-v2_1_20210907175459/evaluation.csv",
        "extra/multistep/SAC_online_n_1_HalfCheetah-v2_2_20210907204725/evaluation.csv",
        "extra/multistep/SAC_online_n_1_HalfCheetah-v2_3_20210907233759/evaluation.csv",
    ],
    "Walker2d-v2": [
        "extra/multistep/SAC_online_n_1_Walker2d-v2_1_20210908022849/evaluation.csv",
        "extra/multistep/SAC_online_n_1_Walker2d-v2_2_20210908052149/evaluation.csv",
        "extra/multistep/SAC_online_n_1_Walker2d-v2_3_20210908081437/evaluation.csv",
    ],
    "HopperBulletEnv-v0": [
        "extra/multistep/SAC_online_n_1_HopperBulletEnv-v0_1_20210913021439/evaluation.csv",
        "extra/multistep/SAC_online_n_1_HopperBulletEnv-v0_2_20210913065751/evaluation.csv",
        "extra/multistep/SAC_online_n_1_HopperBulletEnv-v0_3_20210913114051/evaluation.csv",
    ],
    "HalfCheetahBulletEnv-v0": [
        "extra/multistep/SAC_online_n_1_HalfCheetahBulletEnv-v0_1_20210912112616/evaluation.csv",
        "extra/multistep/SAC_online_n_1_HalfCheetahBulletEnv-v0_2_20210912162156/evaluation.csv",
        "extra/multistep/SAC_online_n_1_HalfCheetahBulletEnv-v0_3_20210912211840/evaluation.csv",
    ],
    "Walker2DBulletEnv-v0": [
        "extra/multistep/SAC_online_n_1_Walker2DBulletEnv-v0_1_20210913162159/evaluation.csv",
        "extra/multistep/SAC_online_n_1_Walker2DBulletEnv-v0_2_20210913211400/evaluation.csv",
        "extra/multistep/SAC_online_n_1_Walker2DBulletEnv-v0_3_20210914015659/evaluation.csv",
    ]
}

N_3 = {
    "Hopper-v2": [
        "extra/multistep/SAC_online_n_3_Hopper-v2_1_20210909220846/evaluation.csv",
        "extra/multistep/SAC_online_n_3_Hopper-v2_2_20210910001422/evaluation.csv",
        "extra/multistep/SAC_online_n_1_Hopper-v2_3_20210907150406/evaluation.csv",
    ],
    "HalfCheetah-v2": [
        "extra/multistep/SAC_online_n_3_HalfCheetah-v2_1_20210909155142/evaluation.csv",
        "extra/multistep/SAC_online_n_3_HalfCheetah-v2_2_20210909175657/evaluation.csv",
        "extra/multistep/SAC_online_n_3_HalfCheetah-v2_3_20210909200312/evaluation.csv",
    ],
    "Walker2d-v2": [
        "extra/multistep/SAC_online_n_3_Walker2d-v2_1_20210909093210/evaluation.csv",
        "extra/multistep/SAC_online_n_3_Walker2d-v2_2_20210909113845/evaluation.csv",
        "extra/multistep/SAC_online_n_3_Walker2d-v2_3_20210909134419/evaluation.csv",
    ],
    "HopperBulletEnv-v0": [
        "extra/multistep/SAC_online_n_3_HopperBulletEnv-v0_1_20210913020338/evaluation.csv",
        "extra/multistep/SAC_online_n_3_HopperBulletEnv-v0_2_20210913064135/evaluation.csv",
        "extra/multistep/SAC_online_n_3_HopperBulletEnv-v0_3_20210913111945/evaluation.csv",
    ],
    "HalfCheetahBulletEnv-v0": [
        "extra/multistep/SAC_online_n_3_HalfCheetahBulletEnv-v0_1_20210912112648/evaluation.csv",
        "extra/multistep/SAC_online_n_3_HalfCheetahBulletEnv-v0_2_20210912161448/evaluation.csv",
        "extra/multistep/SAC_online_n_3_HalfCheetahBulletEnv-v0_3_20210912211132/evaluation.csv",
    ],
    "Walker2DBulletEnv-v0": [
        "extra/multistep/SAC_online_n_3_Walker2DBulletEnv-v0_1_20210913155739/evaluation.csv",
        "extra/multistep/SAC_online_n_3_Walker2DBulletEnv-v0_2_20210913205526/evaluation.csv",
        "extra/multistep/SAC_online_n_3_Walker2DBulletEnv-v0_3_20210914014601/evaluation.csv",
    ]
}

N_5 = {
    "Hopper-v2": [
        "extra/multistep/SAC_online_n_5_Hopper-v2_1_20210911002337/evaluation.csv",
        "extra/multistep/SAC_online_n_5_Hopper-v2_2_20210911022819/evaluation.csv",
        "extra/multistep/SAC_online_n_5_Hopper-v2_3_20210911043422/evaluation.csv",
    ],
    "HalfCheetah-v2": [
        "extra/multistep/SAC_online_n_5_HalfCheetah-v2_1_20210910104926/evaluation.csv",
        "extra/multistep/SAC_online_n_5_HalfCheetah-v2_2_20210910201218/evaluation.csv",
        "extra/multistep/SAC_online_n_5_HalfCheetah-v2_3_20210910221836/evaluation.csv",
    ],
    "Walker2d-v2": [
        "extra/multistep/SAC_online_n_5_Walker2d-v2_1_20210910042308/evaluation.csv",
        "extra/multistep/SAC_online_n_5_Walker2d-v2_2_20210910063225/evaluation.csv",
        "extra/multistep/SAC_online_n_5_Walker2d-v2_3_20210910084058/evaluation.csv",
    ],
    "HopperBulletEnv-v0": [
        "extra/multistep/SAC_online_n_5_HopperBulletEnv-v0_1_20210914195133/evaluation.csv",
        "extra/multistep/SAC_online_n_5_HopperBulletEnv-v0_2_20210914225633/evaluation.csv",
        "extra/multistep/SAC_online_n_5_HopperBulletEnv-v0_3_20210915015848/evaluation.csv",
    ],
    "HalfCheetahBulletEnv-v0": [
        "extra/multistep/SAC_online_n_5_HalfCheetahBulletEnv-v0_1_20210914100751/evaluation.csv",
        "extra/multistep/SAC_online_n_5_HalfCheetahBulletEnv-v0_2_20210914132127/evaluation.csv",
        "extra/multistep/SAC_online_n_5_HalfCheetahBulletEnv-v0_3_20210914163859/evaluation.csv",
    ],
    "Walker2DBulletEnv-v0": [
        "extra/multistep/SAC_online_n_5_Walker2DBulletEnv-v0_1_20210915050107/evaluation.csv",
        "extra/multistep/SAC_online_n_5_Walker2DBulletEnv-v0_2_20210915081538/evaluation.csv",
        "extra/multistep/SAC_online_n_5_Walker2DBulletEnv-v0_3_20210915113118/evaluation.csv",
    ]
}


def plot(score_list, label):
    data = []
    for path in score_list:
        data.append(np.loadtxt(path, delimiter=","))
    x = np.transpose(np.array(data), [2, 1, 0])[1, :, :]
    y = np.transpose(np.array(data), [2, 1, 0])[2, :, :]
    sns.lineplot(x=x.reshape(-1), y=y.reshape(-1), label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--save', type=str)
    args = parser.parse_args()

    plot(SAC[args.env], "SAC")
    plot(N_3[args.env], "SAC (N=3)")
    plot(N_5[args.env], "SAC (N=5)")

    plt.title(args.env)
    plt.xlabel("million step")
    plt.xticks([0, 200000, 400000, 600000, 800000, 1000000],
               ["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    plt.xlim(0, 1000000)
    plt.ylabel("average return")
    plt.legend()

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
