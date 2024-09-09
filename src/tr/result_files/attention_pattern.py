# %%
import os
from types import SimpleNamespace

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from transformers import AutoTokenizer

from tr.analysis.analysis_funcs import create_and_cache_model, import_checkpoint
from tr.training.train_funcs import predict

device = t.device("cuda" if t.cuda.is_available() else "cpu")
t.set_printoptions(precision=3)
t.manual_seed(1)
np.random.seed(1)

single_string = "model/dv50257_ds16000_l2_h1_64064_nmlp_pln_0.0k_ac100_lr5e-05_nes_bs32_ttmixed_nbos_ntw_nfb_predio_ifull_sp0_wd0_crpt0p0_sd120_atnst0w0_tchn_nbt0_inhn.pt"

paths = single_string.split(" ")

prev_nbt = -1  # dummy value not used
for i, checkpoint_filename in enumerate(paths):
    checkpoint_data, train_config, model_state = import_checkpoint(checkpoint_filename)
    model_config = SimpleNamespace(**checkpoint_data["model_config"])
    nbt = train_config.nb_templates
    if i == 0 or nbt != prev_nbt:
        model, cache, test_set, _ = create_and_cache_model(
            model_config, model_state, train_config, cache_model=True, load_train=False
        )
    else:
        model, cache, _, _ = create_and_cache_model(
            model_config,
            model_state,
            train_config,
            cache_model=True,
            load_train=False,
            load_test=False,
        )
    prev_nbt = nbt

    n_prompts = 0
    sample_gap = 40
    assert n_prompts * sample_gap < len(test_set)  # type: ignore
    for prompt_idx in range(0, sample_gap * n_prompts + 1, sample_gap):
        prompt_idx = prompt_idx + 8  # + 8
        prompt_toks, context = test_set[prompt_idx]  # type: ignore
        end_idx = context["end_idx"]  # type: ignore
        prompt_toks = prompt_toks.unsqueeze(0).to(device)  # type: ignore
        context = {
            k: t.tensor([v]).to(device)
            for k, v in context.items()  # type: ignore
            if not isinstance(v, str)
        }
        pred, target = predict(model, prompt_toks, "io", context)

        if len(prompt_toks[0]) > 6:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            prompt_str = [tokenizer.decode([tok]) for tok in prompt_toks[0].tolist()][
                : end_idx + 1  # type: ignore
            ]
            print("prompt:", prompt_str)
            print("pred:", tokenizer.decode(pred[0].argmax()))
        else:
            # min ioi case uses random toks, so replace these with names
            if test_set.templates_by_prompt[prompt_idx] == "BABA":  # type: ignore
                prompt_str = ["Sarah", "Luke", "Sarah", "end"]
            else:
                prompt_str = ["Luke", "Sarah", "Sarah", "end"]
            prompt_str = prompt_str[: end_idx + 1]  # type: ignore

        # png
        plt.figure(figsize=(10 * model.cfg.n_layers, 10 * model.cfg.n_heads))
        plt.rcParams.update({"font.size": 20})
        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                for_chart = cache["pattern", layer][0][  # type: ignore
                    head, : end_idx + 1, : end_idx + 1  # type: ignore
                ].cpu()
                ax = plt.subplot(
                    model.cfg.n_heads,
                    model.cfg.n_layers,
                    head * model.cfg.n_layers + layer + 1,
                )

                cax = ax.imshow(
                    for_chart,
                    cmap="bwr_r",
                    norm=colors.TwoSlopeNorm(vcenter=0),
                    aspect="auto",
                )

                plt.title(f"{layer}.{head}")
                ax.set_xticks(range(len(prompt_str)))
                ax.set_yticks(range(len(prompt_str)))
                ax.set_xticklabels(prompt_str, rotation=90)
                ax.set_yticklabels(prompt_str)

                if head == 0:
                    plt.ylabel("Destination (Query)")
                if layer == model.cfg.n_layers - 1:
                    plt.xlabel("Source (Keys)")

        plt.tight_layout()
        save_path = f"charts/patterns/{''.join(checkpoint_filename.split('/')[-1].split('.')[:-1])}/attnpng_p{prompt_idx}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.show()
        print(1)

    # %%
