# ruff: noqa: E402

import os
from types import SimpleNamespace

import pandas as pd

from tr.analysis.analysis_funcs import import_checkpoint
from tr.ioi_types.data_prep import load_data_from_config
from tr.model_setup.model_settings import load_model
from tr.result_files.graph_funcs import (
    average_nested_dicts,
    collect_nested_data,
    plot_line_with_err,
    plot_violin,
    standard_error,
)
from tr.result_files.patch_and_stats import aba_patching, cbb_patching

while "\\src" in os.getcwd():
    os.chdir("..")


# %%

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
        label="S2_v_l1 -> S2_v_l1",
        sender_name_filter=["blocks.1.attn.hook_v"],
        sender_pos="S2",
        receiver_name_filter=["blocks.1.attn.hook_v"],
        receiver_pos="S2",
    ),  # key result
]


aba_results = {}
cbb_results = {}
min_acc = 500
for checkpoint_filename in paths:
    checkpoint_data, train_config, model_state = import_checkpoint(checkpoint_filename)
    min_acc = min(min_acc, checkpoint_data["accuracy"])
    model_config = SimpleNamespace(**checkpoint_data["model_config"])
    model = load_model(**vars(model_config))
    model.load_state_dict(model_state)

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
    if "mixed" in train_config.template_type and train_config.nb_templates is not None:
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

    label = (
        data_config.nb_templates
        if isinstance(data_config.nb_templates, int) and data_config.nb_templates > 0
        else (
            30
            if data_config.nb_templates is None
            or (
                isinstance(data_config.nb_templates, int)
                and data_config.nb_templates == 0
            )
            else len(data_config.nb_templates) * 2  # type: ignore
        )  # "_".join(data_config.nb_templates)
    )
    # current_results = {}
    # cbb_patching(current_results, model, test_set)
    # cbb_results.setdefault(label, []).append(current_results)

    current_results = {}
    aba_patching(current_results, model, test_set, patch_configs)
    aba_results.setdefault(label, []).append(current_results)

    # %%
print("Min accuracy:", min_acc)


for results in [aba_results]:
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
    plot_line_with_err(sorted_df_avg, sorted_df_err)

# Example usage
for results, title in [(aba_results, "ABA Results")]:
    plot_violin(results, title, bw_adjust=0.35, x_label="Template Count")
