import os
from types import SimpleNamespace

import torch as t
from transformer_lens import ActivationCache

from tr.ioi_types.data_prep import load_data_from_config
from tr.model_setup.model_settings import load_model
from tr.training.train_funcs import calculate_accuracy

device = t.device("cuda" if t.cuda.is_available() else "cpu")


def import_checkpoint(file_name):
    prefix = (
        "model"
        if not file_name.startswith("model") and not file_name.startswith("checkpoints")
        else ""
    )
    suffix = ".pt" if not file_name.endswith(".pt") else ""
    checkpoint_path = os.path.join(prefix, f"{file_name}{suffix}")
    checkpoint_data = t.load(checkpoint_path)

    train_config = dict(train_type="io", prepend_bos=False, template_type="wto")
    train_config.update(checkpoint_data["train_config"])
    train_config = SimpleNamespace(**train_config)

    model_state = checkpoint_data.pop("model_state")
    for key in ["optimizer_state_dict", "source_code"]:
        checkpoint_data.pop(key, None)
    checkpoint_data["pt_file"] = file_name
    print(checkpoint_data)

    return checkpoint_data, train_config, model_state


def check_io_overlap(train_set, test_set, abc_set=None):
    train_ios = t.tensor([train_set.io_tokenIDs, train_set.s_tokenIDs]).T
    test_ios = t.tensor([test_set.io_tokenIDs, test_set.s_tokenIDs]).T
    matches = (test_ios.unsqueeze(1) == train_ios).all(dim=2)

    if abc_set is not None:
        C_toks = abc_set.toks[range(abc_set.N), abc_set.word_idx["S2"]].tolist()
        io_C = t.tensor([test_set.io_tokenIDs, C_toks]).T
        s_C = t.tensor([test_set.s_tokenIDs, C_toks]).T
        io_Cs_matches = (io_C.unsqueeze(1) == train_ios).all(dim=2)
        s_Cs_matches = (s_C.unsqueeze(1) == train_ios).all(dim=2)
        C_matches = t.logical_or(io_Cs_matches, s_Cs_matches)
        matches = t.logical_or(matches, C_matches)

    overlap = matches.any(dim=1)
    non_overlap_idxs = (
        (overlap == False).nonzero(as_tuple=True)[0].tolist()
        if not overlap.all()
        else None
    )
    overlap_count = matches.any(dim=1).sum().item()
    percentage_overlap = (overlap_count / test_ios.size(0)) * 100
    print(
        f"Percentage of test set IO/S pairs also in the train set: {percentage_overlap:.2f}%"
    )
    return non_overlap_idxs


def create_and_cache_model(
    model_config,
    model_state,
    train_config=None,
    cache_model=False,
    load_train=True,
    load_test=True,
):
    model = load_model(**vars(model_config)).to(device)
    model.load_state_dict(model_state)

    if train_config is not None:
        ioi_dataset, data_loader, test_set, test_loader = load_data_from_config(
            train_config, model_config.d_vocab, load_train, load_test
        )
        if load_test:
            calculate_accuracy(model, test_loader, print_message="", train_type="io")

        if load_train and load_test:
            train_prompts = ioi_dataset.toks
            test_prompts = test_set.toks
            matches = (test_prompts.unsqueeze(1) == train_prompts).all(dim=2)
            overlap_count = matches.any(dim=1).sum().item()
            percentage_overlap = (overlap_count / test_prompts.size(0)) * 100
            print(
                f"Percentage of test set prompts also in the train set: {percentage_overlap:.2f}%"
            )

            if train_config.ioi_type == "full":
                check_io_overlap(ioi_dataset, test_set)
    else:
        ioi_dataset = None
        test_set = None

    # def back_hook(pattern, hook):
    #     print(pattern)
    # model.add_hook("blocks.0.hook_resid_pre", back_hook, dir="bwd")
    if cache_model:
        cache_dict = model.add_caching_hooks(incl_bwd=True)
        cache = ActivationCache(cache_dict, model)
    else:
        cache = None

    return model, cache, test_set, ioi_dataset  # type: ignore


def pos_between(pos1: list, pos2: list, include_pos1=False, include_pos2=False):
    # will repeat the last position
    # eg  [1,2,1] and [3,5,2] -> [[2,3,3], [3,4,5], [2,2,2]]
    start_offset = 0 if include_pos1 else 1
    end_offset = 1 if include_pos2 else 0
    ranges = [
        t.arange(pos1[i] + start_offset, pos2[i] + end_offset) for i in range(len(pos1))
    ]
    max_len = max(len(r) for r in ranges)
    padded_ranges = [
        t.cat([r, r[-1].repeat(max_len - len(r))]) if len(r) < max_len else r
        for r in ranges
    ]
    return t.stack(padded_ranges)
