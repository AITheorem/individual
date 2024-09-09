import inspect
import os
import pickle
import re
from functools import partial

import numpy as np
import torch as t
import torch.nn.functional as F
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformer_lens.utils import get_act_name

import tr.ioi_types.min_ioi as min_ioi
from tr.ioi_types.syn_ioi import randomize_names
from tr.model_setup.model_settings import load_model

device = t.device("cuda" if t.cuda.is_available() else "cpu")


webtext_loader = None


def predict(model, prompts, train_type, context, force_logits=None, noise=0.0):
    global webtext_loader
    if force_logits is not None:
        logits = force_logits
    elif noise > 0.0 and np.random.rand() < noise:
        if webtext_loader is None:
            webtext_loader = load_preprocessed_webtext(prompts.size(0))
        webtext_batch = next(iter(webtext_loader))
        alt_prompts = webtext_batch["input_ids"].to(device)
        logits = model(alt_prompts, return_type="logits")
        # train_type = "all"
    #     toks = t.nn.functional.one_hot(prompts, num_classes=model.cfg.d_vocab).float()
    #     toks_embed = toks @ model.W_E
    #     pos_embed = model.pos_embed(toks_embed[:, :, 0])
    #     residual = toks_embed + pos_embed
    #     std_dev = residual.std(dim=0, keepdim=True)
    #     proportional_noise = t.randn_like(residual) * std_dev * noise
    #     residual_with_noise = residual + proportional_noise
    #     logits = model(residual_with_noise, return_type="logits", start_at_layer=0)
    else:
        logits = model(prompts, return_type="logits")

    if train_type == "io":
        end_idx_by_batch = context["end_idx"].to(device)
        end_idx_by_b_seq_v = (
            end_idx_by_batch.unsqueeze(1).unsqueeze(2).expand(-1, 1, logits.size(2))
        )
        predictions = t.gather(logits, 1, end_idx_by_b_seq_v).squeeze(1)
        target_idx_by_b_seq = (end_idx_by_batch + 1).unsqueeze(1)
        targets = t.gather(prompts, 1, target_idx_by_b_seq).squeeze(1)
    elif train_type == "all":
        targets = prompts[:, 1:]
        targets = targets.contiguous().view(-1)
        predictions = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    return predictions, targets


def combine_EU_grads(model):
    with t.no_grad():
        combined_grad = model.embed.W_E.grad + model.unembed.W_U.grad.T
        model.embed.W_E.grad = combined_grad
        model.unembed.W_U.grad = combined_grad.T


def calculate_accuracy(model, data_loader, print_message=None, train_type="io"):
    correct, count, total_loss = 0, 0, 0
    loss_function = t.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    with t.inference_mode():
        for toks, context in data_loader:
            prompts = toks.to(device)
            pred, targets = predict(model, prompts, train_type, context)

            loss = loss_function(pred, targets.long())
            total_loss += loss.item()
            count += targets.size(0)
            correct += (pred.argmax(dim=-1) == targets).sum().item()  # type: ignore
    model.train()
    acc = correct / count
    avg_loss = total_loss / count
    if print_message is not None:
        print(
            print_message,
            f"Accuracy: {acc:.3f} Test_Loss: {avg_loss:.4f}",
        )
    return acc, avg_loss


def load_checkpoint(checkpoint):
    print("Loading checkpoint")
    if not checkpoint.startswith("model"):
        checkpoint = os.path.join("model", f"{checkpoint}")
    if not checkpoint.endswith(".pt"):
        checkpoint += ".pt"
    checkpoint_data = t.load(checkpoint)
    return checkpoint_data


def save_checkpoint(
    model, model_config, train_config, final_test_acc, optim, suffix="", folder="model"
):
    with open(__file__, "r") as file:
        current_source_code = file.read()
    source_code = {
        "main": current_source_code,
        "toy_ioi_dataset": inspect.getsource(min_ioi),
        "load_model": inspect.getsource(load_model),
    }
    data_to_save = {
        "model_state": model.state_dict(),
        "model_config": model_config.__dict__,
        "train_config": train_config.__dict__,
        "accuracy": final_test_acc,
        "end_epoch": train_config.end_epoch,
        "optimizer_state_dict": optim.state_dict(),
        "optimizer": optim.__class__.__name__,
        "source_code": source_code,
    }
    save_name = (
        f"dv{model_config.d_vocab}_"
        f"ds{train_config.dataset_size}_"
        f"l{model_config.n_layers}_"
        f"h{model_config.n_heads}_"
        f"{model_config.d_head}{model_config.d_mlp}{model_config.d_model}_"
        f"{'n' if model_config.attn_only else 'y'}mlp_"
        f"{'nln' if model_config.normalization_type is None else 'yln' if model_config.normalization_type == 'LN' else 'pln'}_"
        f"{data_to_save['end_epoch'] / 1000:.1f}k_"
        f"ac{final_test_acc * 100:.0f}_"
        f"lr{train_config.lr:.0e}_"
        f"{'y' if train_config.early_stopping else 'n'}es_"
        f"bs{train_config.batch_size}"
        f"_tt{train_config.template_type}{('with'+train_config.combined_type) if train_config.combined_type else ''}"
        f"_{'y' if train_config.prepend_bos else 'n'}bos"
        f"_{'y' if train_config.tie_weights else 'n'}tw"
        f"_{'y' if train_config.full_batch else 'n'}fb"
        f"_pred{train_config.train_type}"
        f"_i{train_config.ioi_type}"
        f"_sp{train_config.syn_proportion}"
        f"_wd{train_config.weight_decay}"
        f"_crpt{len(train_config.corrupt_names)}p{train_config.corrupt_proportion}"
        f"_sd{train_config.seed}"
        f"_atnst{len(train_config.attention_steering) if train_config.attention_steering else 0}w{train_config.attention_steer_weight}"
        f"_tch{train_config.teacher + train_config.teacher_type if train_config.teacher else 'n'}"
        f"_nbt{train_config.nb_templates if train_config.nb_templates and isinstance(train_config.nb_templates, int) else 0 if not train_config.nb_templates else 'L' + str(len(train_config.nb_templates))}"
        f"_inh{model_config.gpt2_inherit if model_config.gpt2_inherit else 'n'}"
        f"_noise{train_config.noise}"
        f"{suffix}"
    )
    save_path = os.path.join(folder, f"{save_name}.pt")
    t.save(data_to_save, save_path)

    return save_name, save_path


def load_preprocessed_webtext(batch_size):
    tokenized_dataset = load_from_disk("data/webtext/tokenized_openwebtext_subset")
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)  # type: ignore
    return loader


def add_attention_hook(model, attention_steering, weight):
    attn_steer = {
        "attn_score_cache": {},
        "qk_pairs": {},
        "weight": weight,
    }
    for layer, q, k in attention_steering:
        name = get_act_name("attn_scores", layer)
        attn_steer["qk_pairs"].setdefault(name, []).append((q, k))

    def attn_score_names_filter(name):
        return name in attn_steer["qk_pairs"]

    def attn_score_store(attn_score, hook):
        attn_steer["attn_score_cache"][hook.name] = attn_score

    model.add_hook(attn_score_names_filter, attn_score_store, is_permanent=True)

    return attn_steer


def extract_layer_number(name):
    match = re.search(r"blocks\.(\d+)\.", name)
    if match:
        return int(match.group(1))


def max_layer_plus1_from_hook_names(hook_names):
    layer_nums = [extract_layer_number(name) for name in hook_names]
    if not layer_nums or any(["ln_final" in name for name in hook_names]):
        return None
    elif layer_nums:
        return max(layer_nums) + 1  # type: ignore
    elif all(["_embed_" in name for name in hook_names]):
        return 0
    else:
        raise ValueError("Layer names not expected.")


def logits_with_patch(model, prompts, context, corrupt_caches, tokenizer):

    for corrupt_cache in corrupt_caches:

        def is_name_in_keys(name):
            return name in corrupt_cache

        def store_corrupt(midway_values, hook):
            corrupt_cache[hook.name] = midway_values

        corrupt_prompts = randomize_names(
            prompts, context, tokenizer, preserve_mf=False
        )

        max_layer_plus1 = max_layer_plus1_from_hook_names(corrupt_cache.keys())
        model.run_with_hooks(
            corrupt_prompts,
            return_type=None,
            fwd_hooks=[(is_name_in_keys, store_corrupt)],
            stop_at_layer=max_layer_plus1,  # works like [:stop] and 0=E, so want plus1
        )

    flat_corrupt_cache = {k: v for cache in corrupt_caches for k, v in cache.items()}

    def is_name_in_keys(name):
        return name in flat_corrupt_cache

    def patch_with_corrupt(midway_values, hook):
        return flat_corrupt_cache[hook.name]

    patched_logits = model.run_with_hooks(
        prompts,
        return_type="logits",
        fwd_hooks=[(is_name_in_keys, patch_with_corrupt)],
    )
    return patched_logits


def pattern_hook_fn(variable, hook, pattern_cache, q_dict, k_dict):
    for q_label, q_idxs in q_dict.items():
        q_data = variable[range(variable.shape[0]), range(variable.shape[1]), q_idxs]
        entropy = -t.sum(q_data * t.log(q_data.clamp(1e-9)), dim=-1)
        for stat in ["mean", "var"]:
            stat_list = (
                pattern_cache.setdefault(hook.name, {})
                .setdefault(q_label, {})
                .setdefault("entropy", {})
                .setdefault(stat, [])
            )
            stat_list.append(getattr(entropy, stat)(0).item())

        for k_label, k_idxs in k_dict.items():
            qk_data = q_data[range(q_data.shape[0]), k_idxs]
            for stat in ["mean", "var"]:
                stat_list = (
                    pattern_cache.setdefault(hook.name, {})
                    .setdefault(q_label, {})
                    .setdefault(k_label, {})
                    .setdefault(stat, [])
                )
                stat_list.append(getattr(qk_data, stat)(0).item())


def track_patterns(model, train_set):
    pattern_name_filter = lambda name: name.endswith(".attn.hook_pattern")
    q_dict = {k: train_set.word_idx[k] for k in ["end", "S2"]}
    k_dict = {k: train_set.word_idx[k] for k in ["end", "S2", "IO", "S1"]}
    pattern_cache = {}

    with t.no_grad():
        model.run_with_hooks(
            train_set.toks,
            return_type=None,
            fwd_hooks=[
                (
                    pattern_name_filter,
                    partial(
                        pattern_hook_fn,
                        pattern_cache=pattern_cache,
                        q_dict=q_dict,
                        k_dict=k_dict,
                    ),
                ),
            ],
        )

    for block_key, block_val in pattern_cache.items():
        for q_key, q_val in block_val.items():
            for k_key, k_val in q_val.items():
                for stat_key, stat_val in k_val.items():
                    wandb.log({f"{block_key}>{q_key}>{k_key}>{stat_key}": stat_val[-1]})


def calculate_attn_loss(attn_steer, context):
    # note - could make this more efficient by only storing the attn_score for relevant queries
    attn_losses = []
    for name, qk_pairs in attn_steer["qk_pairs"].items():
        attn_score = attn_steer["attn_score_cache"][name]
        for q, k in qk_pairs:
            q_idx = context[q]
            pred_attn_score = attn_score[t.arange(len(attn_score)), :, q_idx]
            pred_attn_score = pred_attn_score.view(-1, pred_attn_score.size(-1))
            # need to repeat the target for each head
            k_idx = context[k].to(device)
            num_heads = attn_score.size(1)
            target_expanded = k_idx.unsqueeze(1).repeat(1, num_heads).view(-1)

            attn_loss = F.cross_entropy(
                pred_attn_score, target_expanded, reduction="sum"
            )
            attn_losses.append(attn_loss)

    return sum(attn_losses)


def loss_against_teacher_model(
    pred, teacher_model, prompts, context, teacher_type, train_type
):
    with t.no_grad():
        teacher_pred, targets = predict(teacher_model, prompts, train_type, context)
    teacher_pred = teacher_pred.detach()

    if teacher_type == "kl":
        kd_loss = F.kl_div(
            t.nn.functional.log_softmax(pred, dim=-1),
            t.nn.functional.softmax(teacher_pred, dim=-1),
            reduction="sum",
        )
        loss = kd_loss
    elif teacher_type == "mse":
        teacher_io_logits = teacher_pred[:, targets]
        student_io_logits = pred[:, targets]
        io_loss = F.mse_loss(student_io_logits, teacher_io_logits, reduction="sum")
        teacher_s_logits = teacher_pred[:, context["s"]]
        student_s_logits = pred[:, context["s"]]
        s_loss = F.mse_loss(student_s_logits, teacher_s_logits, reduction="sum")
        loss = io_loss + s_loss

    return loss
