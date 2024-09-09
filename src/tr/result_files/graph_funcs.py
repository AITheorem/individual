import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def collect_nested_data(nested_dict):
    data_collection_dict = {}
    for key, dicts_list in nested_dict.items():
        value_lists = {}
        for d in dicts_list:
            for inner_key, value in d.items():
                if inner_key in value_lists:
                    value_lists[inner_key].append(value)
                else:
                    value_lists[inner_key] = [value]
        data_collection_dict[key] = value_lists
    return data_collection_dict


def average_nested_dicts(nested_dict):
    data_collection_dict = collect_nested_data(nested_dict)
    average_dict = {}
    for key, value_lists in data_collection_dict.items():
        average_dict[key] = {
            inner_key: sum(values) / len(values)
            for inner_key, values in value_lists.items()
        }
    return average_dict


def max_nested_dicts(nested_dict):
    data_collection_dict = collect_nested_data(nested_dict)
    max_dict = {}
    for key, value_lists in data_collection_dict.items():
        max_dict[key] = {
            inner_key: max(values) for inner_key, values in value_lists.items()
        }
    return max_dict


def plot_line_with_err(sorted_df_avg, sorted_df_err):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.5, 7.5))
    plt.rcParams.update({"font.size": 16})
    axes = axes.flatten()

    for i, metric in enumerate(sorted_df_avg.index):
        ax = axes[i]
        sorted_df_avg.loc[metric].plot(kind="line", marker="o", title=metric, ax=ax)
        ax.errorbar(
            sorted_df_avg.columns,
            sorted_df_avg.loc[metric],
            yerr=sorted_df_err.loc[metric],
            fmt="o",
            capsize=5,
        )
        ax.set_xlabel("MLP Width")
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_ylabel("Mean Logit Diff")
        ax.grid(True)

    for ax in axes[i + 1 :]:
        ax.set_visible(False)
    for ax in axes:
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)
        ax.tick_params(axis="both", labelsize=16)

    plt.tight_layout()
    # plt.savefig("charts/l2h1_mlpv.png")
    plt.show()


def standard_error(values):
    return np.std(values, ddof=1) / np.sqrt(len(values))


def create_df_for_violin(data_points):
    df_list = []
    for mlp_width, metrics in data_points.items():
        for metric, values in metrics.items():
            df_list.extend(
                [
                    {"MLP Width": mlp_width, "Metric": metric, "Value": value}
                    for value in values
                ]
            )
    return pd.DataFrame(df_list)


def plot_violin(results, title, bw_adjust, x_label="MLP Width"):
    # logic still works for template count, just need to change the x_label
    data_points = collect_nested_data(results)
    df_violin = create_df_for_violin(data_points)
    metrics = sorted(df_violin["Metric"].unique())

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(13.5, 7.5))
    plt.rcParams.update({"font.size": 16})
    fig.suptitle(title, fontsize=16, y=1.02)
    axes = np.atleast_1d(axes).flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = df_violin[df_violin["Metric"] == metric]

        if metric_data["MLP Width"].dtype.kind in "biufc":
            metric_data["MLP Width"] = pd.to_numeric(metric_data["MLP Width"])
            metric_data = metric_data.sort_values("MLP Width")

        parts = sns.violinplot(
            data=metric_data,
            x="MLP Width",
            y="Value",
            ax=ax,
            cut=0,
            inner=None,
            scale="width",
            linewidth=0,
            bw_adjust=bw_adjust,
        )

        for collection in parts.collections:
            collection.set_edgecolor("face")

        sns.pointplot(
            data=metric_data,
            x="MLP Width",
            y="Value",
            ax=ax,
            color="darkorange",
            scale=1,
            errwidth=0,
            capsize=0,
            linestyles="",
        )

        ax.set_title(metric, fontsize=16)
        ax.set_xlabel(x_label, fontsize=16)
        if i == 0:
            ax.set_ylabel("Mean Logit Diff", fontsize=16)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle="--", alpha=0.7)

    for ax in axes[i + 1 :]:
        ax.set_visible(False)
    for ax in axes:
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)
        ax.tick_params(axis="both", labelsize=16)

    plt.tight_layout()
    # plt.savefig("charts/l2h1_mlpv_violin.png")
    plt.show()
