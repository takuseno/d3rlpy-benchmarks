from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rliable import library as rly
from rliable import metrics, plot_utils
from scipy.ndimage.filters import uniform_filter1d

from .data_loader import ScoreData


def plot_score_curve(score: ScoreData, window_size: int = 100, label: Optional[str] = None) -> None:
    y_values = [uniform_filter1d(v, size=window_size) for v in score.scores]
    sns.lineplot(
        x=np.reshape(score.steps, [-1]),
        y=np.reshape(y_values, [-1]),
        label=score.algo if label is None else label,
    )


def plot_performance_profile(scores: Sequence[ScoreData], last_num: int = 1) -> Tuple[plt.Figure, plt.Axes]:
    score_dict = {score.algo: np.transpose(score.scores[:, -last_num:], [1, 0]) for score in scores}
    thresholds = np.linspace(0.0, 120.0, 121)
    score_distributions, score_distributions_cis = rly.create_performance_profile(score_dict, thresholds)
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        score_distributions,
        thresholds,
        performance_profile_cis=score_distributions_cis,
        colors=dict(zip(list(score_dict.keys()), sns.color_palette("colorblind"))),
        xlabel=r"Normalized Score $(\tau)$",
        ax=ax,
    )
    return fig, ax


def plot_aggregate_metrics(
    scores: Sequence[ScoreData], last_num: int = 1, sort: bool = True
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    score_dict = {score.algo: np.transpose(score.scores[:, -last_num:], [1, 0]) for score in scores}
    labels = list(score_dict.keys())
    if sort:
        labels = sorted(labels, key=lambda k: np.mean(score_dict[k]))  # type: ignore
    aggregate_func = lambda x: np.array(
        [metrics.aggregate_median(x), metrics.aggregate_iqm(x), metrics.aggregate_mean(x)]
    )
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(score_dict, aggregate_func, reps=5000)
    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_score_cis,
        metric_names=["Median", "IQM", "Mean"],
        algorithms=labels,
        xlabel="Normalized Score",
    )
    return fig, axes
