# ruff: noqa: E402

import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t

from tr.analysis.analysis_funcs import import_checkpoint
from tr.ioi_types.data_prep import load_data_from_config
from tr.model_setup.model_settings import load_model, setup_model
from tr.result_files.graph_funcs import max_nested_dicts
from tr.result_files.patch_and_stats import double_path_patch_search

while "\\src" in os.getcwd():
    os.chdir("..")


single_string = "model/dv50257_ds16000_l2_h1_64064_nmlp_pln_0.0k_ac100_lr5e-05_nes_bs32_ttmixed_nbos_ntw_nfb_predio_ifull_sp0_wd0_crpt0p0_sd120_atnst0w0_tchn_nbt0_inhn.pt model/dv50257_ds16000_l2_h1_64064_nmlp_pln_0.0k_ac100_lr5e-05_nes_bs32_ttmixed_nbos_ntw_nfb_predio_ifull_sp0_wd0_crpt0p0_sd120_atnst0w0_tchn_nbt0_inhn.pt"
paths = single_string.split(" ")


# %%


all_paths = [paths]
line_labels = ["MLP 80"]
acc_results = {label: {} for label in line_labels}
aba_results = {}
max_sender_results = {}
max_receiver_results = {}
max_sender_receiver_results = {}
min_acc = 500  # dummy
prior_nb_templates = -1  # dummy
for line_idx, line_label in enumerate(line_labels):
    for checkpoint_filename in all_paths[line_idx]:
        checkpoint_data, train_config, model_state = import_checkpoint(
            checkpoint_filename
        )
        min_acc = min(min_acc, checkpoint_data["accuracy"])
        model_config = SimpleNamespace(**checkpoint_data["model_config"])
        model = load_model(**vars(model_config))
        model.load_state_dict(model_state)
        # model = setup_model("gpt2-small")  # uncomment to check gpt2 results

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
        double_path_patch_search(current_results, model, test_set)
        acc_results[line_label].setdefault(model.cfg.n_layers, []).append(
            current_results
        )

print("Min accuracy:", min_acc)

datasets = acc_results
styles = {
    line_labels[-1]: {"color": "blue", "linestyle": "-", "marker": "o"},
}

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7.5))
plt.rcParams.update({"font.size": 16})
if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]

metric_data = {}
for name, results in datasets.items():
    if not results:
        continue
    max_results = max_nested_dicts(results)
    df_avg = pd.DataFrame.from_dict(max_results)

    sorted_df_avg = df_avg.sort_index(axis=1)

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
        ax.set_xlabel("Num Layers")
        ax.set_ylabel("Max Logit Diff")
        ax.grid(True)
        ax.set_ylim(0, 1.0)

for ax in axes[len(sorted_df_avg.index) :]:
    ax.set_visible(False)

for ax in axes:
    ax.title.set_fontsize(16)  # type: ignore
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.tick_params(axis="both", labelsize=16)

plt.tight_layout()
# plt.savefig("charts/varyL_double_patch.png")
plt.show()

print(df_avg)
