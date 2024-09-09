# ruff: noqa: E402


import os
from functools import partial
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t

from tr.analysis.analysis_funcs import import_checkpoint
from tr.analysis.patching_funcs import a_minus_b_logitdiff
from tr.analysis.path_patch_pos import calculate_path_patch, freeze_z, freeze_z_and_mlp
from tr.ioi_types.data_prep import load_data_from_config
from tr.model_setup.model_settings import load_model
from tr.result_files.graph_funcs import (
    average_nested_dicts,
    collect_nested_data,
    plot_line_with_err,
    standard_error,
)
from tr.result_files.patch_and_stats import aba_patching_maxh_search

while "\\src" in os.getcwd():
    os.chdir("..")


single_string = "model/dv50257_ds16000_l2_h1_64064_nmlp_pln_0.0k_ac100_lr5e-05_nes_bs32_ttmixed_nbos_ntw_nfb_predio_ifull_sp0_wd0_crpt0p0_sd120_atnst0w0_tchn_nbt0_inhn.pt model/dv50257_ds16000_l2_h1_64064_nmlp_pln_0.0k_ac100_lr5e-05_nes_bs32_ttmixed_nbos_ntw_nfb_predio_ifull_sp0_wd0_crpt0p0_sd120_atnst0w0_tchn_nbt0_inhn.pt"
paths = single_string.split(" ")


# %%

patch_configs = [
    dict(
        label="S2_v_l0 -> end_q_l1",
        sender_name_filter=["blocks.0.attn.hook_v"],
        sender_pos="S2",
        receiver_name_filter=["blocks.1.attn.hook_q"],
        receiver_pos="end",
    ),  # key result
    dict(
        label="end_z_l0 -> Unembed",
        sender_name_filter=["blocks.0.attn.hook_z"],
        sender_pos="end",
        receiver_name_filter=["blocks.1.hook_resid_post"],
        receiver_pos="end",
        freeze_name_filter=freeze_z,
    ),  # key result
]

aba_results = {}
max_sender_results = {}
max_receiver_results = {}
max_sender_receiver_results = {}
min_acc = 500  # dummy
prior_nb_templates = -1  # dummy
for checkpoint_filename in paths:
    checkpoint_data, train_config, model_state = import_checkpoint(checkpoint_filename)
    min_acc = min(min_acc, checkpoint_data["accuracy"])
    model_config = SimpleNamespace(**checkpoint_data["model_config"])
    model = load_model(**vars(model_config))
    model.load_state_dict(model_state)

    if train_config.nb_templates != prior_nb_templates:
        data_config = SimpleNamespace(
            ioi_type=train_config.ioi_type,
            seed=3498,
            batch_size_test=100,
            template_type="mixed",
            prepend_bos=train_config.prepend_bos,
            nb_templates=train_config.nb_templates,
            data_files=train_config.data_files,
            combined_type=train_config.combined_type,
            syn_proportion=0.0,
        )
        if (
            "mixed" in train_config.template_type
            and train_config.nb_templates is not None
        ):
            data_config.nb_templates = (
                train_config.nb_templates // 2
                if not isinstance(train_config.nb_templates, list)
                else (
                    [(i // 2) for i in train_config.nb_templates if i % 2 == 0]
                    if data_config.template_type == "BABA"
                    else [(i // 2) for i in train_config.nb_templates if i % 2 == 1]
                )
            )
        _, _, test_set, test_loader = load_data_from_config(
            data_config, model_config.d_vocab, load_train=False, load_test=True
        )
    prior_nb_templates = train_config.nb_templates

    current_results = {}
    aba_patching_maxh_search(
        current_results,
        model,
        test_set,
        patch_configs=patch_configs,
    )
    aba_results.setdefault(model.cfg.n_heads, []).append(current_results)

    current_results = {}
    aba_patching_maxh_search(
        current_results,
        model,
        test_set,
        sender_heads="max",
        max_head_by_prompt=True,
        patch_configs=patch_configs,
    )
    max_sender_results.setdefault(model.cfg.n_heads, []).append(current_results)

    current_results = {}
    aba_patching_maxh_search(
        current_results,
        model,
        test_set,
        receiver_heads="max",
        max_head_by_prompt=True,
        patch_configs=patch_configs,
    )
    max_receiver_results.setdefault(model.cfg.n_heads, []).append(current_results)

print("Min accuracy:", min_acc)


datasets = {
    "Patch All Heads": aba_results,
    "Patch Single Sender Head": max_sender_results,
    "Patch Single Receiver Head": max_receiver_results,
    "Single Sender, Single Receiver": max_sender_receiver_results,
}
styles = {
    "Patch All Heads": {"color": "blue", "linestyle": "-", "marker": "o"},
    "Patch Single Sender Head": {"color": "green", "linestyle": "--", "marker": "s"},
    "Patch Single Receiver Head": {
        "color": "red",
        "linestyle": (0, (3, 5, 1, 5)),
        "marker": "^",
    },
    "Single Sender, Single Receiver": {
        "color": "purple",
        "linestyle": (0, (5, 10)),
        "marker": "d",
    },
}

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.5, 7.5))
plt.rcParams.update({"font.size": 16})
axes = axes.flatten()

metric_data = {}
for name, results in datasets.items():
    if not results:
        continue

    average_results = average_nested_dicts(results)
    df_avg = pd.DataFrame.from_dict(average_results)
    data_points = collect_nested_data(results)
    df_err = pd.DataFrame(
        {
            key: {metric: standard_error(values) for metric, values in metrics.items()}
            for key, metrics in data_points.items()
        }
    )
    sorted_df_avg = df_avg.sort_index(axis=1)
    sorted_df_err = df_err.sort_index(axis=1)

    for i, metric in enumerate(sorted_df_avg.index):
        ax = axes[i]
        style = styles[name]
        sorted_df_avg.loc[metric].plot(
            kind="line",
            ax=ax,
            label=name,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
        )
        ax.set_title(metric)
        ax.errorbar(
            sorted_df_avg.columns,
            sorted_df_avg.loc[metric],
            yerr=sorted_df_err.loc[metric],
            fmt=style["marker"],
            capsize=5,
            color=style["color"],
            alpha=0.6,
        )
        ax.set_xlabel("Num Heads")
        if i == 0:
            ax.set_ylabel("Mean Logit Diff")
        ax.grid(True)
        ax.set_ylim([0, 1])

for ax in axes[len(sorted_df_avg.index) :]:
    ax.set_visible(False)

for ax in axes:
    ax.title.set_fontsize(16)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.tick_params(axis="both", labelsize=16)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 0.93))

plt.tight_layout()
# plt.savefig("charts/varyh_mlp0_maxh_pp_varydhead.png")
plt.show()
