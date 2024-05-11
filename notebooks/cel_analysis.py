import json
import os

import pandas as pd

idx = pd.IndexSlice


def load_log_history(
    file, return_df=True, return_cols=["model", "step", "loss", "grad_norm"]
):
    """
    Load log history from json file
    """
    log_history = json.load(open(file))["log_history"]

    losses, metrics = log_history[:-1], log_history[-1]

    if return_df:
        loss_df = pd.DataFrame(losses)
        model_type = "fused_cel" if "fused" in file else "no-fused"
        loss_df["model"] = model_type
        loss_df = loss_df[return_cols]
        metrics_df = pd.Series(metrics).to_frame(name=model_type).loc[idx["step":], :]
    return (loss_df, metrics_df) if return_df else (losses, metrics)


def get_diff(pivoted_df, col, diff1, diff2):
    return abs(pivoted_df[col][diff1] - pivoted_df[col][diff2])


def get_pivoted_df(df):
    pivot = df.pivot(index="step", columns="model", values=["loss", "grad_norm"])
    pivot.columns.names = ["metric", "model"]
    loss_diff = get_diff(pivot, "loss", "no-fused", "fused_cel")
    grad_diff = get_diff(pivot, "grad_norm", "no-fused", "fused_cel")
    pivot.loc[:, ("loss", "absdiff")] = loss_diff
    pivot.loc[:, ("grad_norm", "absdiff")] = grad_diff
    return pd.concat(
        [pivot.loc[:, idx["loss", :]], pivot.loc[:, idx["grad_norm", :]]], axis=1
    )


def load_log_diffs(
    trace_dir, return_df=True, return_cols=["model", "step", "loss", "grad_norm"]
):
    traces = [os.path.join(trace_dir, trace) for trace in sorted(os.listdir(trace_dir))]
    losses = []
    metrics = []
    for trace in traces:
        loss, metric = load_log_history(
            trace, return_df=return_df, return_cols=return_cols
        )
        losses.append(loss)
        metrics.append(metric)

    if return_df:
        losses = pd.concat(losses)
        losses = get_pivoted_df(losses)
        metrics = pd.concat(metrics, axis=1)
    return (losses, metrics)
