# %%
from copy import deepcopy
from functools import partial
from typing import Callable

import numpy as np
import torch as t

from tr.analysis.patching_funcs import a_minus_b_logitdiff, patch_by_head_and_pos

device = t.device("cuda" if t.cuda.is_available() else "cpu")
t.set_printoptions(precision=3)
t.manual_seed(1)
np.random.seed(1)


def freeze_hook(z, hook, orig_cache, dont_freeze_filter):
    if dont_freeze_filter(hook.name):
        # usually this branch is only relevant if receiver is z or mlp_out
        # in future could be more careful here and just unfreeze the receiver position / heads, while freezing the rest
        return z

    z[...] = orig_cache[hook.name][...]
    return z


def run_patch_sender_freeze_intermediates_cache_receiver(
    model,
    orig_dataset,
    new_cache,
    orig_cache,
    receiver_names_filter,
    sender_pos=None,
    send_filter=None,
    freeze_filter=None,
    heads_to_patch={},
):
    freeze_hook_fn = partial(
        freeze_hook,
        orig_cache=orig_cache,
        dont_freeze_filter=receiver_names_filter,
    )
    patch_sender_fn = partial(
        patch_by_head_and_pos,
        new_cache=new_cache,
        heads_to_patch=heads_to_patch,
        pos_to_patch=sender_pos,
    )
    fwd_hooks = [
        (freeze_filter, freeze_hook_fn),
        (send_filter, patch_sender_fn),
    ]
    with model.hooks(fwd_hooks) as hooked_model:
        _, receiver_cache = hooked_model.run_with_cache(
            orig_dataset.toks,
            names_filter=receiver_names_filter,
            return_type=None,
        )

    return receiver_cache


def run_patch_receiver_heads(
    model,
    patched_cache,
    orig_dataset,
    receiver_names_filter,
    receiver_pos=None,
    receiver_heads={},
):
    patch_receiver_fn = partial(
        patch_by_head_and_pos,
        new_cache=patched_cache,
        heads_to_patch=receiver_heads,
        pos_to_patch=receiver_pos,
    )
    patched_logits = model.run_with_hooks(
        orig_dataset.toks,
        fwd_hooks=[(receiver_names_filter, patch_receiver_fn)],
        return_type="logits",
    )  # (batch, seq, d_vocab)
    # only care about end logit
    end_pos = orig_dataset.word_idx["end"]
    patched_end_logits = patched_logits[range(len(patched_logits)), end_pos]
    return patched_end_logits


def create_name_filter(filter_list):
    if isinstance(filter_list, list):
        return lambda name: name in filter_list
    else:
        return filter_list


def convert_pos(pos, data):
    if isinstance(pos, str):
        return data.word_idx[pos]
    elif isinstance(pos, dict):
        return {
            k: (data.word_idx[v] if isinstance(v, str) else v) for k, v in pos.items()
        }
    else:
        return pos


def double_path_patch(
    model,
    orig_data,
    aba_data,
    sender_name_filter=[],
    sender_heads={},
    sender_pos=None,
    receiver_name_filter=[],
    receiver_heads={},
    receiver_pos=None,
    freeze_name_filter=[],
    label="",
    logit_diff_fn: Callable = None,  # type: ignore
    receiver_name_filter_2=[],
    receiver_heads_2={},
    receiver_pos_2=None,
    freeze_name_filter_2=[],
):
    # must convert to functions because later we use fwd_hooks=[(filter, hook_fn)]
    # which requires filter to be a single str or function, not a list
    actual_sender_name_filter = create_name_filter(sender_name_filter)
    actual_receiver_name_filter = create_name_filter(receiver_name_filter)
    actual_freeze_name_filter = create_name_filter(freeze_name_filter)

    actual_receiver_name_filter_2 = create_name_filter(receiver_name_filter_2)
    actual_freeze_name_filter_2 = create_name_filter(freeze_name_filter_2)

    # Combine sender and freeze filters into a single filter
    send_and_freeze_filter = lambda name: actual_sender_name_filter(
        name
    ) or actual_freeze_name_filter(name)

    sender_pos = convert_pos(sender_pos, orig_data)
    receiver_pos = convert_pos(receiver_pos, orig_data)
    sender_pos_2 = receiver_pos
    receiver_pos_2 = convert_pos(receiver_pos_2, orig_data)

    ioi_logits, orig_cache = model.run_with_cache(
        orig_data.toks, names_filter=send_and_freeze_filter, return_type="logits"
    )
    aba_logits, new_cache = model.run_with_cache(
        aba_data.toks, names_filter=send_and_freeze_filter, return_type="logits"
    )
    ioi_logit_diffs = logit_diff_fn(
        ioi_logits, per_prompt=True, end_idx=orig_data.word_idx["end"]
    )
    aba_logit_diffs = logit_diff_fn(
        aba_logits, per_prompt=True, end_idx=orig_data.word_idx["end"]
    )

    receiver_cache = run_patch_sender_freeze_intermediates_cache_receiver(
        model,
        orig_data,
        new_cache,
        orig_cache,
        sender_pos=sender_pos,
        send_filter=actual_sender_name_filter,
        receiver_names_filter=actual_receiver_name_filter,
        freeze_filter=actual_freeze_name_filter,
        heads_to_patch=sender_heads,
    )
    receiver_cache_2 = run_patch_sender_freeze_intermediates_cache_receiver(
        model,
        orig_data,
        receiver_cache,
        orig_cache,
        sender_pos=sender_pos_2,
        send_filter=actual_receiver_name_filter,
        receiver_names_filter=actual_receiver_name_filter_2,
        freeze_filter=actual_freeze_name_filter_2,
        heads_to_patch=receiver_heads,
    )
    patched_end_logits = run_patch_receiver_heads(
        model,
        receiver_cache_2,
        orig_data,
        receiver_names_filter=actual_receiver_name_filter_2,
        receiver_pos=receiver_pos_2,
        receiver_heads=receiver_heads_2,
    )

    patched_logit_diffs = logit_diff_fn(
        patched_end_logits, per_prompt=True, end_idx=None  # end logits already selected
    )
    # what fraction of the deterioration in performance can be attributed to this path
    # path_performance_impact_per_prompt = (
    #     ioi_logit_diffs - patched_logit_diffs
    # ) / (ioi_logit_diffs - aba_logit_diffs)
    path_performance_impact = (ioi_logit_diffs.mean() - patched_logit_diffs.mean()) / (
        ioi_logit_diffs.mean() - aba_logit_diffs.mean()
    )
    # this version deals better with outliers that have close to 0 denominator
    # (compared to path_performance_impact_per_prompt.mean())
    # +1 means patching this path damages performance a lot, 0 means not at all, negative means patching helps
    # note previously we used the negative of this

    return (
        path_performance_impact.item(),
        patched_logit_diffs,
        ioi_logit_diffs,
        aba_logit_diffs,
    )


def calculate_path_patch(
    model,
    orig_data,
    aba_data,
    sender_name_filter=[],
    sender_heads={},
    sender_pos=None,
    receiver_name_filter=[],
    receiver_heads={},
    receiver_pos=None,
    freeze_name_filter=[],
    label="",
    logit_diff_fn: Callable = None,  # type: ignore
):
    # must convert to functions because later we use fwd_hooks=[(filter, hook_fn)]
    # which requires filter to be a single str or function, not a list
    if isinstance(receiver_name_filter, list):
        actual_receiver_name_filter = lambda name: name in receiver_name_filter
    else:
        actual_receiver_name_filter = receiver_name_filter
    if isinstance(sender_name_filter, list):
        actual_sender_name_filter = lambda name: name in sender_name_filter
    else:
        actual_sender_name_filter = sender_name_filter
    if isinstance(freeze_name_filter, list):
        actual_freeze_name_filter = lambda name: name in freeze_name_filter
    else:
        actual_freeze_name_filter = freeze_name_filter
    send_and_freeze_filter = lambda name: actual_sender_name_filter(
        name
    ) or actual_freeze_name_filter(name)
    # also convert str pos to tensors
    if isinstance(sender_pos, str):
        sender_pos = orig_data.word_idx[sender_pos]
    if isinstance(receiver_pos, str):
        receiver_pos = orig_data.word_idx[receiver_pos]
    if isinstance(receiver_pos, dict):
        receiver_pos = {
            k: (orig_data.word_idx[v] if isinstance(v, str) else v)
            for k, v in receiver_pos.items()
        }

    ioi_logits, orig_cache = model.run_with_cache(
        orig_data.toks, names_filter=send_and_freeze_filter, return_type="logits"
    )
    aba_logits, new_cache = model.run_with_cache(
        aba_data.toks, names_filter=send_and_freeze_filter, return_type="logits"
    )
    ioi_logit_diffs = logit_diff_fn(
        ioi_logits, per_prompt=True, end_idx=orig_data.word_idx["end"]
    )
    aba_logit_diffs = logit_diff_fn(
        aba_logits, per_prompt=True, end_idx=orig_data.word_idx["end"]
    )

    receiver_cache = run_patch_sender_freeze_intermediates_cache_receiver(
        model,
        orig_data,
        new_cache,
        orig_cache,
        sender_pos=sender_pos,
        send_filter=actual_sender_name_filter,
        receiver_names_filter=actual_receiver_name_filter,
        freeze_filter=actual_freeze_name_filter,
        heads_to_patch=sender_heads,
    )
    patched_end_logits = run_patch_receiver_heads(
        model,
        receiver_cache,
        orig_data,
        receiver_names_filter=actual_receiver_name_filter,
        receiver_pos=receiver_pos,
        receiver_heads=receiver_heads,
    )

    patched_logit_diffs = logit_diff_fn(
        patched_end_logits, per_prompt=True, end_idx=None  # end logits already selected
    )
    # what fraction of the deterioration in performance can be attributed to this path
    # path_performance_impact_per_prompt = (
    #     ioi_logit_diffs - patched_logit_diffs
    # ) / (ioi_logit_diffs - aba_logit_diffs)
    path_performance_impact = (ioi_logit_diffs.mean() - patched_logit_diffs.mean()) / (
        ioi_logit_diffs.mean() - aba_logit_diffs.mean()
    )
    # this version deals better with outliers that have close to 0 denominator
    # (compared to path_performance_impact_per_prompt.mean())
    # +1 means patching this path damages performance a lot, 0 means not at all, negative means patching helps
    # note previously we used the negative of this

    return (
        path_performance_impact.item(),
        patched_logit_diffs,
        ioi_logit_diffs,
        aba_logit_diffs,
    )


def freeze_z(name):
    return name.endswith("z")


def freeze_mlp(name):
    return name.endswith("mlp_out")


def freeze_z_and_mlp(name):
    return freeze_z(name) or freeze_mlp(name)
