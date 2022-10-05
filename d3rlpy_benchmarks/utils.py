import numpy as np

EXPERT_SCORES = {
    "halfcheetah": 12135.0,
    "hopper": 3234.3,
    "walker2d": 4592.3,
}

BASE_RANDOM = {
    "halfcheetah": -280.178953,
    "hopper": -20.272305,
    "walker2d": 1.629008,
}


def normalize_d4rl_score(env: str, score: np.ndarray) -> np.ndarray:
    expert_score = EXPERT_SCORES[env]
    random_score = BASE_RANDOM[env]
    return 100.0 * (score - random_score) / (expert_score - random_score)


def get_canonical_algo_name(name: str) -> str:
    if name == "PLASWithPerturbation":
        return "PLAS+P"
    elif name == "TD3PlusBC":
        return "TD3+BC"
    else:
        return name
