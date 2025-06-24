# encoding: utf-8
from collections import defaultdict
from collections.abc import Collection

import plotly.graph_objects as go
from mc_dagprop import DagContext, SimResult
from plotly.subplots import make_subplots


def retrieve_absolute_and_relative_delays(
    context: DagContext, result: SimResult
) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    """
    Bucket absolute and relative delays by activity_type.
    """
    absolute_delays_by_type = defaultdict(list)
    relative_delays_by_type = defaultdict(list)

    for activity in context.activities.values():
        a_type = activity.activity_type
        base = activity.minimal_duration
        link_index = activity.idx
        delta = result.durations[link_index] - base
        absolute_delays_by_type[a_type].append(delta)
        if base != 0.0:
            relative_delays_by_type[a_type].append(delta / base)

    return dict(absolute_delays_by_type), dict(relative_delays_by_type)


def plot_activity_delays(context: DagContext, results: Collection[SimResult]) -> go.Figure:
    """
    Build a Plotly figure with two rows of histograms per activity type:
      • row 1: absolute delays
      • row 2: relative delays
    """
    # aggregate across all results
    abs_all = defaultdict(list)
    rel_all = defaultdict(list)
    for res in results:
        abs_by_type, rel_by_type = retrieve_absolute_and_relative_delays(context, res)
        for t, vals in abs_by_type.items():
            abs_all[t].extend(vals)
        for t, vals in rel_by_type.items():
            rel_all[t].extend(vals)

    # sort activity types for consistent ordering
    activity_types = sorted(abs_all.keys())
    n_types = len(activity_types)
    if n_types == 0:
        raise ValueError("No activity types found in context.activities")

    # create 2 × n_types subplot grid
    fig = make_subplots(
        rows=2,
        cols=n_types,
        shared_yaxes=False,
        subplot_titles=[f"Abs delay (type {t})" for t in activity_types]
        + [f"Rel delay (type {t})" for t in activity_types],
        vertical_spacing=0.1,
    )

    # add one histogram per cell
    for idx, t in enumerate(activity_types, start=1):
        fig.add_trace(go.Histogram(x=abs_all[t], name=f"abs t={t}", showlegend=False), row=1, col=idx)
        fig.add_trace(go.Histogram(x=rel_all[t], name=f"rel t={t}", showlegend=False), row=2, col=idx)

        fig.update_xaxes(title_text="sec", row=1, col=idx)
        fig.update_xaxes(title_text="× base", row=2, col=idx)

    fig.update_layout(
        height=600,
        width=300 * n_types,
        title_text="Activity-type delay distributions\n(Top: absolute, Bottom: relative)",
        bargap=0.1,
    )

    return fig
