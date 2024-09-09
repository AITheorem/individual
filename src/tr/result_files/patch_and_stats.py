from functools import partial

import torch as t

from tr.analysis.patching_funcs import a_minus_b_logitdiff
from tr.analysis.path_patch_pos import calculate_path_patch, double_path_patch


def aba_patching(result_store, model, orig_data, patch_configs):
    aba_set = orig_data.gen_flipped_prompts("ABB->ABA, BAB->BAA")
    io_minus_s_logitdiff = partial(
        a_minus_b_logitdiff, io_toks=orig_data.io_tokenIDs, s_toks=orig_data.s_tokenIDs
    )
    for config in patch_configs:
        result_store[config["label"]] = calculate_path_patch(
            model,
            orig_data,
            aba_set,
            logit_diff_fn=io_minus_s_logitdiff,
            **config,  # type: ignore
        )[0]


def cbb_patching(result_store, model, orig_data, patch_configs):
    cbb_set = orig_data.gen_flipped_prompts("ABB->CBB, BAB->BCB")
    io_minus_cio_logitdiff = partial(
        a_minus_b_logitdiff, io_toks=orig_data.io_tokenIDs, s_toks=cbb_set.io_tokenIDs
    )
    for config in patch_configs:
        result_store[config["label"]] = calculate_path_patch(
            model,
            orig_data,
            cbb_set,
            logit_diff_fn=io_minus_cio_logitdiff,
            **config,  # type: ignore
        )[0]


def create_doubles(n):
    layers = list(range(n))
    doubles_int = []
    for first_end in range(1, n):
        first = layers[:first_end]
        second = layers[first_end:]
        doubles_int.append((first, second))
    list_of_doubles = []
    for first, second in doubles_int:
        z = [f"blocks.{i}.attn.hook_v" for i in first]
        v = [f"blocks.{i}.attn.hook_q" for i in second]
        list_of_doubles.append((z, v))
    return list_of_doubles


def single_path_patch_search(
    result_store,
    model,
    orig_data,
):
    acc_set = orig_data.gen_flipped_prompts("ABB->ABA, BAB->BAA")
    io_minus_s_logitdiff = partial(
        a_minus_b_logitdiff, io_toks=orig_data.io_tokenIDs, s_toks=orig_data.s_tokenIDs
    )
    patch_configs = [
        dict(
            label="S2_v -> end_q",
            sender_pos="S2",
            receiver_pos="end",
        )
    ]
    list_of_doubles = create_doubles(model.cfg.n_layers)
    for config in patch_configs:
        max_logit_diff = -500
        for v, q in list_of_doubles:
            config["sender_name_filter"] = v
            config["receiver_name_filter"] = q
            logit_diff = calculate_path_patch(
                model,
                orig_data,
                acc_set,
                logit_diff_fn=io_minus_s_logitdiff,
                **config,  # type: ignore
            )[0]
            max_logit_diff = max(max_logit_diff, logit_diff)
        result_store[config["label"]] = max_logit_diff


def layer_triplets(n):
    layers = list(range(n))
    result = []
    for first_end in range(1, n - 1):
        for second_end in range(first_end + 1, n):
            first = layers[:first_end]
            second = layers[first_end:second_end]
            third = layers[second_end:]
            result.append((first, second, third))
    return result


def create_triplets(n):
    tiplets_int = layer_triplets(n)
    list_of_triplets = []
    for first, second, third in tiplets_int:
        z = [f"blocks.{i}.attn.hook_z" for i in first]
        v = [f"blocks.{i}.attn.hook_v" for i in second]
        q = [f"blocks.{i}.attn.hook_q" for i in third]
        list_of_triplets.append((z, v, q))
    return list_of_triplets


def double_path_patch_search(result_store, model, orig_data):
    acc_set = orig_data.gen_flipped_prompts("ABB->ABA, BAB->BAA")
    io_minus_s_logitdiff = partial(
        a_minus_b_logitdiff, io_toks=orig_data.io_tokenIDs, s_toks=orig_data.s_tokenIDs
    )
    patch_configs = [
        dict(
            label="S2_z -> S2_v -> end_q",
            sender_pos="S2",
            receiver_pos="S2",
            receiver_pos_2="end",
        )
    ]
    list_of_triplets = create_triplets(model.cfg.n_layers)
    for config in patch_configs:
        max_logit_diff = -500
        for z, v, q in list_of_triplets:
            config["sender_name_filter"] = z
            config["receiver_name_filter"] = v
            config["receiver_name_filter_2"] = q
            logit_diff = double_path_patch(
                model,
                orig_data,
                acc_set,
                logit_diff_fn=io_minus_s_logitdiff,
                **config,  # type: ignore
            )[0]
            max_logit_diff = max(max_logit_diff, logit_diff)
        result_store[config["label"]] = max_logit_diff


def aba_patching_maxh_search(
    result_store,
    model,
    orig_data,
    sender_heads=None,
    receiver_heads=None,
    max_head_by_prompt=False,
    patch_configs=[{}],
):
    aba_set = orig_data.gen_flipped_prompts("ABB->ABA, BAB->BAA")
    io_minus_s_logitdiff = partial(
        a_minus_b_logitdiff, io_toks=orig_data.io_tokenIDs, s_toks=orig_data.s_tokenIDs
    )
    for config in patch_configs:
        if sender_heads == "max" and receiver_heads is None:
            sender_results = []
            max_diffs_by_prompt = -t.ones(len(aba_set)).to(aba_set.device)
            head_indices = t.zeros(len(aba_set)).long().to(aba_set.device)
            for head in range(model.cfg.n_heads):
                config["sender_heads"] = {
                    sender_name: [head]
                    for sender_name in config["sender_name_filter"]
                    if ".attn." in sender_name
                }
                (
                    path_performance_impact,
                    patched_logit_diffs,
                    ioi_logit_diffs,
                    aba_logit_diffs,
                ) = calculate_path_patch(
                    model,
                    orig_data,
                    aba_set,
                    logit_diff_fn=io_minus_s_logitdiff,
                    **config,  # type: ignore
                )
                diffs_by_prompt = (ioi_logit_diffs - patched_logit_diffs) / (
                    ioi_logit_diffs - aba_logit_diffs
                )
                capped_diffs_by_prompt = t.clamp(diffs_by_prompt, -1, 1)
                new_max = t.maximum(max_diffs_by_prompt, capped_diffs_by_prompt)
                update_indices = new_max != max_diffs_by_prompt
                head_indices[update_indices] = head
                max_diffs_by_prompt = new_max
                sender_results.append(capped_diffs_by_prompt.mean().item())
            result_store[config["label"]] = (
                t.mean(max_diffs_by_prompt).item()
                if max_head_by_prompt
                else max(sender_results)
            )
            # print("sender", model.cfg.n_heads, head_indices)

        elif sender_heads is None and receiver_heads == "max":
            receiver_results = []
            max_diffs_by_prompt = -t.ones(len(aba_set)).to(aba_set.device)
            head_indices = t.zeros(len(aba_set)).long().to(aba_set.device)
            for head in range(model.cfg.n_heads):
                config["receiver_heads"] = {
                    receiver_name: [head]
                    for receiver_name in config["receiver_name_filter"]
                    if ".attn." in receiver_name
                }
                (
                    path_performance_impact,
                    patched_logit_diffs,
                    ioi_logit_diffs,
                    aba_logit_diffs,
                ) = calculate_path_patch(
                    model,
                    orig_data,
                    aba_set,
                    logit_diff_fn=io_minus_s_logitdiff,
                    **config,  # type: ignore
                )
                diffs_by_prompt = (ioi_logit_diffs - patched_logit_diffs) / (
                    ioi_logit_diffs - aba_logit_diffs
                )
                capped_diffs_by_prompt = t.clamp(diffs_by_prompt, -1, 1)
                new_max = t.maximum(max_diffs_by_prompt, capped_diffs_by_prompt)
                update_indices = new_max != max_diffs_by_prompt
                head_indices[update_indices] = head
                max_diffs_by_prompt = new_max
                receiver_results.append(capped_diffs_by_prompt.mean().item())
            result_store[config["label"]] = (
                t.mean(max_diffs_by_prompt).item()
                if max_head_by_prompt
                else max(receiver_results)
            )
            # print("receiver", model.cfg.n_heads, head_indices)

        elif sender_heads == "max" and receiver_heads == "max":
            sender_receiver_results = []
            max_diffs_by_prompt = -t.ones(len(aba_set)).to(aba_set.device)
            for sender_head in range(model.cfg.n_heads):
                for receiver_head in range(model.cfg.n_heads):
                    config["sender_heads"] = {
                        sender_name: [sender_head]
                        for sender_name in config["sender_name_filter"]
                        if ".attn." in sender_name
                    }
                    config["receiver_heads"] = {
                        receiver_name: [receiver_head]
                        for receiver_name in config["receiver_name_filter"]
                        if ".attn." in receiver_name
                    }
                    (
                        path_performance_impact,
                        patched_logit_diffs,
                        ioi_logit_diffs,
                        aba_logit_diffs,
                    ) = calculate_path_patch(
                        model,
                        orig_data,
                        aba_set,
                        logit_diff_fn=io_minus_s_logitdiff,
                        **config,  # type: ignore
                    )
                    diffs_by_prompt = (ioi_logit_diffs - patched_logit_diffs) / (
                        ioi_logit_diffs - aba_logit_diffs
                    )
                    capped_diffs_by_prompt = t.clamp(diffs_by_prompt, -1, 1)
                    max_diffs_by_prompt = t.maximum(
                        max_diffs_by_prompt, capped_diffs_by_prompt
                    )
                    sender_receiver_results.append(capped_diffs_by_prompt.mean().item())
            result_store[config["label"]] = (
                t.mean(max_diffs_by_prompt).item()
                if max_head_by_prompt
                else max(sender_receiver_results)
            )

        else:
            (
                path_performance_impact,
                patched_logit_diffs,
                ioi_logit_diffs,
                aba_logit_diffs,
            ) = calculate_path_patch(
                model,
                orig_data,
                aba_set,
                logit_diff_fn=io_minus_s_logitdiff,
                **config,  # type: ignore
            )
            diff_by_prompt = (ioi_logit_diffs - patched_logit_diffs) / (
                ioi_logit_diffs - aba_logit_diffs
            )
            capped_diff_by_prompt = t.clamp(diff_by_prompt, -1, 1)
            result_store[config["label"]] = t.mean(capped_diff_by_prompt).item()
