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

QR_SAC = {
    "Hopper-v2": [
        "extra/dist/SAC_online_qr_Hopper-v2_1_20210909011159/evaluation.csv",
        "extra/dist/SAC_online_qr_Hopper-v2_2_20210909033231/evaluation.csv",
        "extra/dist/SAC_online_qr_Hopper-v2_3_20210909055317/evaluation.csv",
    ],
    "HalfCheetah-v2": [
        "extra/dist/SAC_online_qr_HalfCheetah-v2_1_20210908180856/evaluation.csv",
        "extra/dist/SAC_online_qr_HalfCheetah-v2_2_20210908202936/evaluation.csv",
        "extra/dist/SAC_online_qr_HalfCheetah-v2_3_20210908225154/evaluation.csv",
    ],
    "Walker2d-v2": [
        "extra/dist/SAC_online_qr_Walker2d-v2_1_20210908110605/evaluation.csv",
        "extra/dist/SAC_online_qr_Walker2d-v2_2_20210908132528/evaluation.csv",
        "extra/dist/SAC_online_qr_Walker2d-v2_3_20210908154657/evaluation.csv",
    ],
    "HopperBulletEnv-v0": [
        "extra/dist/SAC_online_qr_HopperBulletEnv-v0_1_20210913032234/evaluation.csv",
        "extra/dist/SAC_online_qr_HopperBulletEnv-v0_2_20210913055127/evaluation.csv",
        "extra/dist/SAC_online_qr_HopperBulletEnv-v0_3_20210913081937/evaluation.csv",
    ],
    "HalfCheetahBulletEnv-v0": [
        "extra/dist/SAC_online_qr_HalfCheetahBulletEnv-v0_1_20210912193246/evaluation.csv",
        "extra/dist/SAC_online_qr_HalfCheetahBulletEnv-v0_2_20210912220918/evaluation.csv",
        "extra/dist/SAC_online_qr_HalfCheetahBulletEnv-v0_3_20210913004612/evaluation.csv",
    ],
    "Walker2DBulletEnv-v0": [
        "extra/dist/SAC_online_qr_Walker2DBulletEnv-v0_1_20210912115500/evaluation.csv",
        "extra/dist/SAC_online_qr_Walker2DBulletEnv-v0_2_20210912142816/evaluation.csv",
        "extra/dist/SAC_online_qr_Walker2DBulletEnv-v0_3_20210912165944/evaluation.csv",
    ]
}

IQN_SAC = {
    "Hopper-v2": [
        "extra/dist/SAC_online_iqn_Hopper-v2_1_20210908233348/evaluation.csv",
        "extra/dist/SAC_online_iqn_Hopper-v2_2_20210909052157/evaluation.csv",
        "extra/dist/SAC_online_iqn_Hopper-v2_3_20210909112027/evaluation.csv",
    ],
    "HalfCheetah-v2": [
        "extra/dist/SAC_online_iqn_HalfCheetah-v2_1_20210909171057/evaluation.csv",
        "extra/dist/SAC_online_iqn_HalfCheetah-v2_2_20210909231110/evaluation.csv",
        "extra/dist/SAC_online_iqn_HalfCheetah-v2_3_20210910050232/evaluation.csv",
    ],
    "Walker2d-v2": [
        "extra/dist/SAC_online_iqn_Walker2d-v2_1_20210908233406/evaluation.csv",
        "extra/dist/SAC_online_iqn_Walker2d-v2_2_20210909051902/evaluation.csv",
        "extra/dist/SAC_online_iqn_Walker2d-v2_3_20210909111702/evaluation.csv",
    ],
    "HopperBulletEnv-v0": [
        "extra/dist/SAC_online_iqn_HopperBulletEnv-v0_1_20210913030912/evaluation.csv",
        "extra/dist/SAC_online_iqn_HopperBulletEnv-v0_2_20210913080425/evaluation.csv",
        "extra/dist/SAC_online_iqn_HopperBulletEnv-v0_3_20210913130150/evaluation.csv",
    ],
    "HalfCheetahBulletEnv-v0": [
        "extra/dist/SAC_online_iqn_HalfCheetahBulletEnv-v0_1_20210912115226/evaluation.csv",
        "extra/dist/SAC_online_iqn_HalfCheetahBulletEnv-v0_2_20210912170009/evaluation.csv",
        "extra/dist/SAC_online_iqn_HalfCheetahBulletEnv-v0_3_20210912220503/evaluation.csv",
    ],
    "Walker2DBulletEnv-v0": [
        "extra/dist/SAC_online_iqn_Walker2DBulletEnv-v0_1_20210913175909/evaluation.csv",
        "extra/dist/SAC_online_iqn_Walker2DBulletEnv-v0_2_20210913230802/evaluation.csv",
        "extra/dist/SAC_online_iqn_Walker2DBulletEnv-v0_3_20210914041137/evaluation.csv",
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
    plot(QR_SAC[args.env], "QR-SAC")
    plot(IQN_SAC[args.env], "IQN-SAC")

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
