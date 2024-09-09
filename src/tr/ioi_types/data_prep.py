from typing import Tuple

import torch as t
from torch.utils.data import DataLoader

import tr.ioi_types.min_ioi as min_ioi
from tr.ioi_types.min_ioi import DummyTokenizer, MinimalIOIDataset
from tr.ioi_types.syn_ioi import CombinedDataset, SynIOIDataset, replace_synonyms


def load_data_from_config(
    train_config, d_vocab, load_train=True, load_test=True
) -> Tuple[SynIOIDataset, DataLoader, SynIOIDataset, DataLoader]:
    try:
        train_config.nb_templates
    except AttributeError:
        train_config.nb_templates = None

    if train_config.data_files:
        ioi_dataset, data_loader, test_set, test_loader = load_data_from_files(
            train_config
        )
    elif train_config.ioi_type == "full" and train_config.combined_type is None:
        ioi_dataset, data_loader, test_set, test_loader = prepare_full_ioi_dataset(
            train_config, train_config.seed, load_train, load_test
        )
    elif train_config.ioi_type == "full":
        ioi_dataset, data_loader, test_set, test_loader = (
            prepare_fullcombined_ioi_dataset(train_config, train_config.seed)
        )

    elif train_config.ioi_type == "min":
        ioi_dataset, data_loader, test_set, test_loader, non_overlap_idxs = (
            prepare_min_ioi_data(d_vocab, train_config, train_config.seed)
        )
    elif train_config.ioi_type == "fullwithbacba":
        train_config.combined_type = "BACBA"
        ioi_dataset, data_loader, test_set, test_loader = (
            prepare_fullcombined_ioi_dataset(train_config, train_config.seed)
        )

    # assert d_vocab > max(ioi_dataset.toks.max(), test_set.toks.max())

    return ioi_dataset, data_loader, test_set, test_loader  # type: ignore


def prepare_min_ioi_data(d_vocab, train_config, seed, load_train=True, load_test=True):
    min_ioi.NAMES = [str(x) for x in range(3, d_vocab)]  # 0 is reserved for BOS
    dummy_tokenizer = DummyTokenizer()

    if load_train:
        train_set = MinimalIOIDataset(
            ("mixed" if train_config.template_type not in min_ioi.DONT_MIX else "BABA"),
            N=train_config.dataset_size,
            tokenizer=dummy_tokenizer,
            prepend_bos=train_config.prepend_bos,
            symmetric=train_config.dataset_symmetric,
            seed=seed,
            device="cpu",
            base_template_type=train_config.template_type,
        )
        data_loader = DataLoader(
            train_set,
            batch_size=train_config.batch_size,
            shuffle=True,
            pin_memory=True,
            # num_workers=2,
        )  # type: ignore
    else:
        train_set = None
        data_loader = None

    if load_test:
        test_set = MinimalIOIDataset(
            ("mixed" if train_config.template_type not in min_ioi.DONT_MIX else "BABA"),
            N=train_config.batch_size_test,
            tokenizer=dummy_tokenizer,
            prepend_bos=train_config.prepend_bos,
            symmetric=True,
            seed=seed + 100,
            device="cpu",
            base_template_type=train_config.template_type,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=train_config.batch_size_test,
            shuffle=False,
            pin_memory=True,
            # num_workers=2,
        )  # type: ignore
    else:
        test_set = None
        test_loader = None

    if load_train and load_test:
        train_prompts = train_set.toks  # type: ignore
        test_prompts = test_set.toks  # type: ignore
        matches = (test_prompts.unsqueeze(1) == train_prompts).all(dim=2)
        overlap = matches.any(dim=1)
        non_overlap_idxs = (
            (overlap == False).nonzero(as_tuple=True)[0].tolist()
            if not overlap.all()
            else None
        )
        overlap_count = matches.any(dim=1).sum().item()
        percentage_overlap = (overlap_count / test_prompts.size(0)) * 100
        print(
            f"Percentage of test set prompts also in the train set: {percentage_overlap:.2f}%"
        )

    return train_set, data_loader, test_set, test_loader, non_overlap_idxs


def prepare_full_ioi_dataset(train_config, seed, load_train, load_test):

    if load_train:
        train_set = SynIOIDataset(
            N=train_config.dataset_size,
            # AutoTokenizer
            prompt_type=train_config.template_type,
            prepend_bos=train_config.prepend_bos,
            symmetric=train_config.dataset_symmetric,
            seed=seed,
            device="cpu",
            nb_templates=train_config.nb_templates,
        )
        train_set = replace_synonyms(train_set, train_config.syn_proportion)
        data_loader = DataLoader(
            train_set,
            batch_size=train_config.batch_size,
            shuffle=True,
            pin_memory=True,
            # num_workers=2,
        )  # type: ignore
    else:
        train_set = None
        data_loader = None

    if load_test:
        test_set = SynIOIDataset(
            N=train_config.batch_size_test,
            # AutoTokenizer
            prompt_type=train_config.template_type,
            prepend_bos=train_config.prepend_bos,
            symmetric=True,
            seed=seed + 100,
            device="cpu",
            nb_templates=train_config.nb_templates,
        )
        test_set = replace_synonyms(test_set, train_config.syn_proportion)
        test_loader = DataLoader(
            test_set,
            batch_size=train_config.batch_size_test,
            shuffle=False,
            pin_memory=True,
            # num_workers=2,
        )  # type: ignore
    else:
        test_set = None
        test_loader = None

    return train_set, data_loader, test_set, test_loader


def prepare_fullcombined_ioi_dataset(train_config, seed):
    main_set = SynIOIDataset(
        N=train_config.dataset_size,
        # AutoTokenizer
        prompt_type=train_config.template_type,
        prepend_bos=train_config.prepend_bos,
        symmetric=train_config.dataset_symmetric,
        seed=seed,
        device="cpu",
        nb_templates=train_config.nb_templates,
    )
    main_set = replace_synonyms(main_set, train_config.syn_proportion)
    bacba_set = SynIOIDataset(
        N=train_config.dataset_size,
        # AutoTokenizer
        prompt_type=train_config.combined_type,
        prepend_bos=train_config.prepend_bos,
        symmetric=train_config.dataset_symmetric,
        seed=seed + 200,
        device="cpu",
    )
    train_set = CombinedDataset([main_set, bacba_set])

    test_set = SynIOIDataset(
        N=train_config.batch_size_test,
        # AutoTokenizer
        prompt_type=train_config.template_type,
        prepend_bos=train_config.prepend_bos,
        symmetric=True,
        seed=seed + 100,
        device="cpu",
        nb_templates=train_config.nb_templates,
    )
    test_set = replace_synonyms(test_set, train_config.syn_proportion)

    data_loader = DataLoader(
        train_set,
        batch_size=train_config.batch_size,
        shuffle=True,
        pin_memory=True,
        # num_workers=2,
    )  # type: ignore

    test_loader = DataLoader(
        test_set,
        batch_size=train_config.batch_size_test,
        shuffle=False,
        pin_memory=True,
        # num_workers=2,
    )  # type: ignore
    return train_set, data_loader, test_set, test_loader


def load_data_from_files(train_config):
    train_data, test_data = train_config.data_files
    train_set = t.load(train_data)
    test_set = t.load(test_data)
    data_loader = DataLoader(
        train_set,
        batch_size=train_config.batch_size,
        shuffle=True,
        pin_memory=True,
        # num_workers=2,
    )  # type: ignore

    test_loader = DataLoader(
        test_set,
        batch_size=train_config.batch_size_test,
        shuffle=False,
        pin_memory=True,
        # num_workers=2,
    )  # type: ignore
    return train_set, data_loader, test_set, test_loader
