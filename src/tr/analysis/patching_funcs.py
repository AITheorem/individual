import torch as t


def patch_by_head_and_pos(
    z,
    hook,
    new_cache,
    heads_to_patch={},
    pos_to_patch=None,  # either dim='batch' or dim='batch several_seq_idxs'
):
    if heads_to_patch and ".attn." in hook.name:  # type: ignore
        relevant_heads = heads_to_patch[hook.name]
        # might want to allow some hooks with all heads and some with specific heads
        # in which case go back to relevant_heads = heads_to_patch.get(hook.name, None)
        # but that should be a conscious choice. more likely to deref error here due to typo
    else:
        relevant_heads = None
    batch_range = range(z.shape[0])
    if isinstance(pos_to_patch, dict):
        pos_to_patch = pos_to_patch[hook.name]
    if pos_to_patch is not None and pos_to_patch.dim() > 1:
        batch_range = t.arange(z.shape[0]).unsqueeze(1).expand_as(pos_to_patch)
    new_z = new_cache[hook.name]

    if relevant_heads is None and pos_to_patch is None:
        z[...] = new_z[...]
    elif relevant_heads is None and pos_to_patch is not None:
        z[batch_range, pos_to_patch] = new_z[batch_range, pos_to_patch]
    elif relevant_heads is not None and pos_to_patch is None:
        z[:, :, relevant_heads] = new_z[:, :, relevant_heads]
    elif relevant_heads is not None and pos_to_patch is not None:
        if isinstance(pos_to_patch, list):
            pos_to_patch = t.tensor(pos_to_patch)
        if isinstance(relevant_heads, list):
            relevant_heads = t.tensor(relevant_heads)
        if pos_to_patch.dim() < relevant_heads.dim():
            while pos_to_patch.dim() < relevant_heads.dim():
                pos_to_patch = pos_to_patch.unsqueeze(-1)
            batch_range = t.arange(z.shape[0])
            while batch_range.dim() < pos_to_patch.dim():
                batch_range = batch_range.unsqueeze(-1)
            batch_range = batch_range.expand_as(pos_to_patch)
            z[batch_range, pos_to_patch, relevant_heads] = new_z[
                batch_range, pos_to_patch, relevant_heads
            ]
        else:
            z[batch_range, pos_to_patch, relevant_heads] = new_z[
                batch_range, pos_to_patch, relevant_heads
            ]
    return z


def a_minus_b_logitdiff(
    logits,  #  "batch seq d_vocab" or "batch d_vocab" if end_logits_only
    io_toks,
    s_toks,  # for tokIDs, crucial NOTE - use IOI dataset, even for the ABC
    per_prompt=False,
    end_idx=None,
):
    batch_range = range(logits.size(0))
    if end_idx is not None:
        io_logits = logits[batch_range, end_idx, io_toks]
        s_logits = logits[batch_range, end_idx, s_toks]
    else:
        io_logits = logits[batch_range, io_toks]
        s_logits = logits[batch_range, s_toks]

    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()
