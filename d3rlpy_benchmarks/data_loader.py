import dataclasses
import glob
import os
from typing import Optional, Sequence

import numpy as np

from .path import ATARI_DIR, D4RL_DIR
from .utils import get_canonical_algo_name, normalize_d4rl_score


@dataclasses.dataclass(frozen=True)
class ScoreData:
    algo: str
    env: str
    dataset: str
    steps: np.ndarray
    raw_scores: np.ndarray
    scores: np.ndarray


def get_d4rl_algo_list() -> Sequence[str]:
    algos = []
    for log_dir in glob.glob(os.path.join(D4RL_DIR, "*")):
        algo = log_dir.split("/")[-1]
        if algo not in algos:
            algos.append(algo)
    return sorted(algos)


def load_d4rl_score(algo: str, env: str, dataset: str) -> Optional[ScoreData]:
    score_list = []
    step_list = []
    for log_dir in glob.glob(os.path.join(D4RL_DIR, algo, f"*_{env}-{dataset}_*")):
        with open(os.path.join(log_dir, "environment.csv"), "r") as f:
            data = np.loadtxt(f, delimiter=",", skiprows=1)
            score_list.append(data[:, 2])
            step_list.append(data[:, 1])

    if len(score_list) == 0:
        return None

    raw_scores = np.array(score_list)
    steps = np.array(step_list)

    # drop warming-up steps
    if algo in ["plas", "plas_with_perturbation"]:
        raw_scores = raw_scores[:, -500:]
        steps = steps[:, :500]

    return ScoreData(
        algo=get_canonical_algo_name(algo),
        env=env,
        dataset=dataset,
        steps=steps,
        raw_scores=raw_scores,
        scores=normalize_d4rl_score(env, raw_scores),
    )


def load_all_algos_d4rl_scores(env: str, dataset: str, exclude: Optional[Sequence[str]] = None) -> Sequence[ScoreData]:
    algos = get_d4rl_algo_list()
    if exclude:
        algos = [algo for algo in algos if algo not in exclude]
    rets = []
    for algo in algos:
        score = load_d4rl_score(algo, env, dataset)
        if score:
            rets.append(score)
    return rets


def get_atari_algo_list() -> Sequence[str]:
    algos = []
    for log_dir in glob.glob(os.path.join(ATARI_DIR, "*")):
        base = log_dir.split("/")[-1]
        splits = base.split("_")
        algo = splits[0]
        if algo not in algos:
            algos.append(algo)
    return sorted(algos)


def get_atari_env_list() -> Sequence[str]:
    envs = []
    for log_dir in glob.glob(os.path.join(ATARI_DIR, "*")):
        base = log_dir.split("/")[-1]
        splits = base.split("_")
        env = splits[1]
        if env not in envs:
            envs.append(env)
    return sorted(envs)


def load_atari_score(algo: str, env: str) -> ScoreData:
    score_list = []
    step_list = []
    for log_dir in glob.glob(os.path.join(ATARI_DIR, f"{algo}_{env}_*")):
        with open(os.path.join(log_dir, "environment.csv"), "r") as f:
            data = np.loadtxt(f, delimiter=",", skiprows=1)
            score_list.append(data[:, 2])
            step_list.append(data[:, 1])
    raw_scores = np.array(score_list)
    steps = np.array(step_list)
    return ScoreData(
        algo=algo,
        env=env,
        dataset="",
        steps=steps,
        raw_scores=raw_scores,
        scores=raw_scores,
    )


def load_all_algos_atari_scores(env: str, exclude: Optional[Sequence[str]] = None) -> Sequence[ScoreData]:
    algos = get_atari_algo_list()
    if exclude:
        algos = [algo for algo in algos if algo not in exclude]
    return [load_atari_score(algo, env) for algo in algos]
