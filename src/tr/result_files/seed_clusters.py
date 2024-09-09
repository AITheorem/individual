# ruff: noqa: E402


# %%

import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

from tr.analysis.analysis_funcs import import_checkpoint
from tr.ioi_types.data_prep import load_data_from_config
from tr.model_setup.model_settings import load_model
from tr.result_files.patch_and_stats import aba_patching

single_string = "model/l3h2_mlp80_consth/dv50257_ds16000_l3_h2_648064_ymlp_pln_0.0k_ac100_lr5e-05_nes_bs32_ttmixed_nbos_ntw_nfb_predio_ifull_sp0_wd0_crpt0p0_sd6_atnst0w0_tchn_nbt0_inhn.pt model/dv50257_ds16000_l3_h2_648064_ymlp_pln_0.0k_ac100_lr5e-05_nes_bs32_ttmixed_nbos_ntw_nfb_predio_ifull_sp0_wd0_crpt0p0_sd6_atnst0w0_tchn_nbt0_inhn_noise0.pt"


paths = single_string.split(" ")


while "\\src" in os.getcwd():
    os.chdir("..")

# %%

patch_configs = [
    dict(
        label="S2_v_l2 -> S2_v_l2",
        sender_name_filter=["blocks.2.attn.hook_v"],
        sender_pos="S2",
        receiver_name_filter=["blocks.2.attn.hook_v"],
        receiver_pos="S2",
    ),  # key result
    dict(
        label="S2_v_l0 -> end_q_l1",
        sender_name_filter=["blocks.0.attn.hook_v"],
        sender_pos="S2",
        receiver_name_filter=["blocks.1.attn.hook_q"],
        receiver_pos="end",
    ),  # key result
    # dict(
    #     label="S2_v_l1 -> end_q_l2",
    #     sender_name_filter=["blocks.1.attn.hook_v"],
    #     sender_pos="S2",
    #     receiver_name_filter=["blocks.2.attn.hook_q"],
    #     receiver_pos="end",
    #     # freeze_name_filter=freeze_z,
    # ),  # important check - no S2 inhibition in l1
]


aba_results = {}
cbb_results = {}
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
    aba_patching(current_results, model, test_set, patch_configs)
    aba_results.setdefault(train_config.seed, []).append(current_results)

seeds = []
x_coords = []
y_coords = []

for seed, results in aba_results.items():
    for result in results:
        seeds.append(seed)
        x_coords.append(result["S2_v_l2 -> S2_v_l2"])
        y_coords.append(result["S2_v_l0 -> end_q_l1"])
df_from_dict = pd.DataFrame({"Seed": seeds, "x": x_coords, "y": y_coords})
df_from_dict.set_index("Seed", inplace=True)

colors = [
    "red" if x < 0.5 and y > 0.5 else "green" if x < 0.5 and y < 0.5 else "blue"
    for x, y in zip(x_coords, y_coords)
]

print("green seeds")
green_seeds = df_from_dict[(df_from_dict["x"] < 0.5) & (df_from_dict["y"] < 0.5)]
print(green_seeds)

print("red seeds")
red_seeds = df_from_dict[(df_from_dict["x"] < 0.5) & (df_from_dict["y"] > 0.5)]
print(red_seeds)

plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 14})
plt.scatter(
    df_from_dict["x"],
    df_from_dict["y"],
    marker="x",  # type: ignore
    c=colors,
    # alpha=0.5,
    zorder=3,
)
plt.title("3 Layers: Logit Diff of Two Path Patches")
plt.xlabel("S2_v_l2 -> S2_v_l2")
plt.ylabel("S2_v_l0\n->\nend_q_l1", rotation=0, labelpad=30)
plt.grid(True, zorder=1)
plt.show()
# plt.savefig("charts/l3h1_seed_clusters.png")


print(1)
