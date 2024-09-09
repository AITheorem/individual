# %%
import math
import random
from types import SimpleNamespace

import numpy as np
import torch as t
from torch.optim.adam import Adam
from tqdm import tqdm

import wandb
from tr.ioi_types.data_prep import load_data_from_config
from tr.model_setup.model_settings import load_model, setup_model
from tr.training.train_funcs import (
    add_attention_hook,
    calculate_accuracy,
    calculate_attn_loss,
    combine_EU_grads,
    load_checkpoint,
    logits_with_patch,
    loss_against_teacher_model,
    predict,
    save_checkpoint,
)

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def train_main(model_config, train_config, suffix=""):
    t.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    random.seed(train_config.seed)

    if hasattr(model_config, "d_all"):
        for attr in ["d_head", "d_mlp", "d_model"]:
            setattr(model_config, attr, model_config.d_all)

    # Load Data
    ioi_dataset, data_loader, test_set, test_loader = load_data_from_config(
        train_config, model_config.d_vocab
    )
    aba_dataset = test_set.gen_flipped_prompts("ABB->ABA, BAB->BAA")
    # overlap_idxs = list(set(range(len(test_set))) - set(non_overlap_idxs))  # check

    # batch_tracking/loss_tracking removed
    batch_tracking_loader = None
    loss_tracking_loaders = None

    model = load_model(seed=train_config.seed, **model_config.__dict__).to(device)
    optim = Adam(model.parameters(), lr=train_config.lr)
    checkpoint_data = {}
    if train_config.checkpoint is not None:
        checkpoint_data = load_checkpoint(train_config.checkpoint)
        model.load_state_dict(checkpoint_data["model_state"])
        optim.load_state_dict(checkpoint_data["optimizer_state_dict"])
        for param_group in optim.param_groups:
            param_group["lr"] = train_config.lr  # force new LR
    if train_config.tie_weights:
        model.unembed.W_U = t.nn.Parameter(model.embed.W_E.data.T.clone())
    train_config.start_epoch = checkpoint_data.get("train_config", {}).get(
        "end_epoch", -1
    )

    teacher_model = None
    if train_config.teacher:
        teacher_model = setup_model(train_config.teacher)
        teacher_model.eval()

    attn_steer = None
    if train_config.attention_steering:
        attn_steer = add_attention_hook(
            model, train_config.attention_steering, train_config.attention_steer_weight
        )

    calculate_accuracy(model, test_loader, "Initial", train_config.train_type)
    epoch, batch_tracker, loss_trackers = train_loop(
        model,
        data_loader,
        train_config.start_epoch,
        train_config.num_epochs,
        optim,
        test_loader,
        train_config.train_type,
        train_config.early_stopping,
        train_config.tie_weights,
        train_config.full_batch,
        batch_tracking_loader,
        loss_tracking_loaders,
        train_config.corrupt_names,
        attn_steer,
        teacher_model,
        train_config.teacher_type,
        train_config.corrupt_proportion,
        model_config,
        train_config,
        aba_dataset,
        train_config.noise,
    )
    train_config.end_epoch = epoch
    final_test_acc, final_test_loss = calculate_accuracy(
        model, test_loader, "Final", train_type="io"
    )

    my_name, check_path = save_checkpoint(
        model, model_config, train_config, final_test_acc, optim, suffix
    )

    wandb.log(
        {
            "te_ac_final": final_test_acc,
            "te_ls_final": final_test_loss,
            "checkpoint": check_path,
        }
    )

    return final_test_acc


def train_loop(
    model,
    data_loader,
    start_epoch,
    num_epochs,
    optim,
    test_loader,
    train_type="io",
    es=False,
    tie_weights=False,
    full_batch=False,
    batch_tracking_loader=None,
    loss_tracking_loaders=None,
    corrupt_names=None,
    attn_steer=None,
    teacher_model=None,
    teacher_type=None,
    corrupt_proportion=0,
    model_config=None,
    train_config=None,
    aba_dataset=None,
    noise=0,
):
    loss_function = t.nn.CrossEntropyLoss(reduction="none")
    progress_bar = tqdm(range(start_epoch, start_epoch + num_epochs))

    batch_tracker = None
    loss_trackers = None

    for epoch in progress_bar:
        avg_loss = train_epoch(
            model,
            data_loader,
            loss_function,
            optim,
            train_type,
            tie_weights,
            full_batch,
            batch_tracker,
            loss_trackers,
            corrupt_names,
            attn_steer,
            teacher_model,
            teacher_type,
            corrupt_proportion,
            test_loader.dataset,
            aba_dataset,
            noise,
        )
        test_acc, test_loss = calculate_accuracy(model, test_loader, None, train_type)
        tqdm.write(
            f"epoch {epoch} trLs {avg_loss:.4f} teLs {test_loss:.4f} teAc {test_acc*100:.1f}"
        )
        io_loss, io_acc = test_loss, test_acc
        if train_type == "all":
            io_acc, io_loss = calculate_accuracy(model, test_loader, None, "io")
            tqdm.write(f"IO teLs {io_loss:.4f} teAc {io_acc*100:.1f}")

        wandb.log(
            {
                "epoch": epoch,
                "tr_ls": avg_loss,
                "te_ls": test_loss,
                "te_ac": test_acc,
                "io_te_ls": io_loss,
                "io_te_ac": io_acc,
            }
        )

        if epoch % 10 == 0:
            train_config.end_epoch = epoch  # type: ignore
            save_checkpoint(
                model,
                model_config,
                train_config,
                test_acc,
                optim,
                suffix=f"_ep{epoch}",
                folder="checkpoints",
            )

        if io_loss < 0.05:
            es = True
        if es:
            break
    return epoch, batch_tracker, loss_trackers


def optim_step_and_track(
    model, optim, tie_weights, batch_tracker=None, loss_trackers=None, train_type=None
):
    combine_EU_grads(model) if tie_weights else None
    optim.step()
    optim.zero_grad()


def train_epoch(
    model,
    data_loader,
    loss_function,
    optim,
    train_type,
    tie_weights,
    full_batch,
    batch_tracker,
    loss_trackers,
    corrupt_names,
    attn_steer,
    teacher_model,
    teacher_type,
    corrupt_proportion,
    test_dataset,
    aba_dataset,
    noise=0,
):
    optim.zero_grad()
    model.train()
    total_loss = 0
    if corrupt_names is not None:
        corrupt_caches = [{name: None for name in names} for names in corrupt_names]

    for i, (toks, context) in enumerate(data_loader):
        prompts = toks.to(device)

        if not corrupt_names or random.random() > corrupt_proportion:
            pred, targets = predict(model, prompts, train_type, context, noise=noise)
        else:
            patched_logits = logits_with_patch(
                model, prompts, context, corrupt_caches, data_loader.dataset.tokenizer
            )
            # still need to run predict function to select last token, although now the logits are fixed
            pred, targets = predict(model, prompts, train_type, context, patched_logits)
            if noise > 0:
                raise NotImplementedError("Noise not implemented for corrupt_names")

        if attn_steer:
            attn_loss = calculate_attn_loss(attn_steer, context)

        if not teacher_model:
            loss = loss_function(pred, targets.long())
            loss = loss.sum()
        else:
            loss = loss_against_teacher_model(
                pred, teacher_model, prompts, context, teacher_type, train_type
            )

        if attn_steer:
            loss += attn_loss * attn_steer["weight"]
        total_loss += loss.item()
        loss.backward()

        if not full_batch:
            optim_step_and_track(model, optim, tie_weights)

    if full_batch:
        optim_step_and_track(model, optim, tie_weights)

    average_loss = total_loss / len(data_loader.dataset)
    return average_loss


def sample_and_double_integers(n, seed):
    # make sure we select base templates randomly
    if n is None or n == 0 or isinstance(n, list):
        return n
    random.seed(seed)
    sampled_integers = random.sample(list(range(15)), n)
    doubled_list = []
    for value in sampled_integers:
        doubled_list.extend([2 * value, 2 * value + 1])
    return doubled_list


if __name__ == "__main__":
    for (
        d_model,
        d_mlp,
        nbt,
        n_heads,
        n_layers,
        tie_weights,
        train_type,
        seed,
        d_head,
        noise,
        syn_proportion,
    ) in [
        (64, 0, 0, 1, 3, False, "io", 1, 64, 0, 0.5),
    ]:
        print(d_mlp, nbt)

        d_head = d_head if d_head else math.floor(d_model / n_heads)
        model_config = SimpleNamespace(
            d_vocab=50257,
            n_layers=n_layers,
            n_heads=n_heads,  # 1
            d_head=d_head,  # 25 # 16 4 # 5 # 768
            d_mlp=d_mlp,
            d_model=d_model,  # 16 4 # 5 # 768
            attn_only=False,
            normalization_type="LNPre",  # None,  # "LN"
            extra_hooks=False,
            gpt2_inherit=None,  # "EUpos"
        )
        train_config = SimpleNamespace(
            num_epochs=500,  # 25,  # 400,  # 12000,  # 2500 # 500
            batch_size=32,  # 256
            batch_size_test=256,
            lr=5e-5,  # 5e-4,  # 5e-3
            dataset_size=16000,  # 16000
            dataset_symmetric=False,
            early_stopping=False,
            seed=seed,
            train_type=train_type,
            prepend_bos=False,
            tie_weights=tie_weights,
            ioi_type="full",  # "full",  # min full
            template_type="mixed",  # "mixed",  # wto mixed
            nb_templates=sample_and_double_integers(nbt, seed),  # None
            combined_type=None,
            syn_proportion=0,  # 0.5
            weight_decay=0,
            corrupt_names=[
                # ["blocks.0.attn.hook_z", "blocks.1.attn.hook_z"],
                # ["blocks.0.attn.hook_z"],
                # ["blocks.1.attn.hook_z"],
                # ["blocks.2.attn.hook_k"],  #  Don't forget the proportion
            ],  # []
            corrupt_proportion=0,
            attention_steering=[
                # (0, "s2_idx", "s1_idx"),
                # (0, "end_idx", "start_idx"),
                # # # (0, "end_idx", "end-1_idx"),
                # (1, "end_idx", "s2_idx"),
                # (2, "end_idx", "io_idx"),
            ],  # layer, q, k
            attention_steer_weight=0,  # 0.5,
            teacher=None,  # "gpt2",
            teacher_type="kl",
            full_batch=False,  # False
            data_files=None,
            checkpoint=None,
            noise=noise,
        )  #

        # use_minwto = True
        # if use_minwto:
        #     train_config.ioi_type = "min"
        #     train_config.template_type = "wto"
        #     model_config.d_vocab = 1000
        # use_minumc = True
        # if use_minumc:
        #     train_config.ioi_type = "min"
        #     train_config.template_type = "umc"
        #     model_config.d_vocab = 1000

        NAME = f"minwto_rnbt{nbt}_thresh05_nh{n_heads}_dm{d_model}_mlp{d_mlp}_{'n' if not tie_weights else 'y'}_tw_tt{train_type}_sd{seed}"

        if model_config.d_mlp == 0:
            model_config.attn_only = True
        if train_config.nb_templates == 0:
            train_config.nb_templates = None
        wandb.init(
            project="template_freq",
            name=NAME,
            config=model_config.__dict__ | train_config.__dict__,
        )

        print(model_config.__dict__)
        print(train_config.__dict__)
        train_main(model_config, train_config, suffix="")
        wandb.finish()
