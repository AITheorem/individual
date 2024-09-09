# this code has been adapted from Wang et al. (2022), wang2022interpretability
# we also use the same code with slightly different adaptations in syn_ioi.py

# %%
import copy
import random
import re
import warnings
from typing import List, Optional, Union

import numpy as np
import torch as t
from transformers import AutoTokenizer

# NAMES is used in exactly 2 places, the init of the IOI dataset and the gen_flipped_prompts method
# so it's safe to change it, generate a new dataset, change it again, etc.
NAMES = [str(x) for x in range(3, 8)]

base_BABA_templates = dict(
    # predict tok 1
    aa0=[" [A] [A] . [B] ."],
    # tok 2
    a1a=[" [A] 1 [A] . [B] ."],
    aba=[" [A] [B] [A] ."],
    baa=[" [B] [A] [A] ."],
    raa=[" [A] 1 [A] . [B] .", " 1 [A] [A] . [B] ."],
    # tok 3
    umc=[" [B] [A] [B] [A] ."],
    # tok 4
    wto=[" [B] [A] [B] 1 [A] ."],
    rto=[" [B] [A] [B] 1 [A] .", " [B] [A] 1 [B] [A] ."],
    # tok 4 with flipped task
    fto=[" [B] [A] [B] 1 [A] . . .", " [B] [A] [B] 2 [B] . [A] ."],
    # tok 5
    spa=[" [B] [A] [B] 1 [A] . .", " [B] [A] [B] 2 [A] . .", " [B] [A] [B] 1 2 [A] ."],
)
DONT_MIX = ["aa0", "a1a", "aba", "baa", "raa"]

ABC_TEMPLATES = [" [A] [B] [C] [A] .", " [A] [B] [A] [C] ."]
BAC_TEMPLATES = [
    template.replace("[B]", "[A]", 1).replace("[A]", "[B]", 1)
    for template in ABC_TEMPLATES
]


def BABA_to_ABBA(BABA_templates):
    templates = BABA_templates.copy()
    for i in range(len(templates)):
        first_clause = True
        for j in range(1, len(templates[i]) - 1):
            if templates[i][j - 1 : j + 2] == "[B]" and first_clause:
                templates[i] = templates[i][:j] + "A" + templates[i][j + 1 :]
            elif templates[i][j - 1 : j + 2] == "[A]" and first_clause:
                first_clause = False
                templates[i] = templates[i][:j] + "B" + templates[i][j + 1 :]
    return templates


VERBS = [" "]
PLACES = [""]
OBJECTS = [""]


def gen_prompt_uniform(
    templates: list[str], names, nouns_dict, N, symmetric, prefixes=None, abc=False
):
    nb_gen = 0
    ioi_prompts = []
    while nb_gen < N:
        temp = random.choice(templates)
        temp_id = templates.index(temp)
        name_1 = ""
        name_2 = ""
        name_3 = ""
        while len(set([name_1, name_2, name_3])) < 3:
            name_1 = random.choice(names)
            name_2 = random.choice(names)
            name_3 = random.choice(names)

        nouns = {}
        ioi_prompt = {}
        for k in nouns_dict:
            nouns[k] = random.choice(nouns_dict[k])
            ioi_prompt[k] = nouns[k]
        prompt = temp
        for k in nouns_dict:
            prompt = prompt.replace(k, nouns[k])

        if prefixes is not None:
            L = random.randint(30, 40)
            pref = ".".join(random.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""

        prompt1 = prompt.replace("[A]", name_1)
        prompt1 = prompt1.replace("[B]", name_2)
        if abc:
            prompt1 = prompt1.replace("[C]", name_3)
        prompt1 = pref + prompt1
        ioi_prompt["text"] = prompt1
        ioi_prompt["IO"] = name_1
        ioi_prompt["S"] = name_2
        ioi_prompt["TEMPLATE_IDX"] = temp_id
        ioi_prompts.append(ioi_prompt)
        if abc:
            ioi_prompts[-1]["C"] = name_3

        nb_gen += 1

        if symmetric and nb_gen < N:
            prompt2 = prompt.replace("[A]", name_2)
            prompt2 = prompt2.replace("[B]", name_1)
            prompt2 = pref + prompt2
            ioi_prompts.append(
                {"text": prompt2, "IO": name_2, "S": name_1, "TEMPLATE_IDX": temp_id}
            )
            nb_gen += 1
    return ioi_prompts


def flip_words_in_prompt(
    prompt: str,
    word1: str,
    word2: str,
    instances: Optional[Union[int, List[int]]] = None,
):
    """
    Flips instances of word `word1` with `word2` in the string `string`.

    By default it flips all instances, but the optional `instances` argument specifies which
    instances to flip (e.g. if instances = 0, then it only flips the 0th instance of either
    word1 or word2.

    Examples of (arguments) -> return value:

        ("ABA", "A", "B") -> "BAB"
        ("ABA", "A", "B", 1) -> "AAA"
        ("ABA", "A", "B", [0, 1]) -> "BAA
    """
    split_prompt = re.split("({}|{})".format(word1, word2), prompt)
    indices_of_names = [i for i, s in enumerate(split_prompt) if s in (word1, word2)]
    indices_to_flip = [indices_of_names[i] for i in instances]
    for i in indices_to_flip:
        split_prompt[i] = word1 if split_prompt[i] == word2 else word1
    prompt = "".join(split_prompt)
    return prompt


def gen_flipped_prompts(
    prompts: List[dict],
    templates_by_prompt: List[str],
    flip: str,
    names: List[str],
    seed: int,
) -> List[dict]:
    """
    Flip prompts in a way described by the flip argument. Returns new prompts.

    prompts: List[dict]
        list of prompts, each prompt is a dict with keys "S", "IO", "text", etc

    templates_by_prompt: List[str]
        each element is "ABBA" or "BABA"

    flip: str
        "ABB -> XYZ, BAB -> XYZ" means that the prompt "A and B went to [place], B gave [object] to A" becomes "X and Y went to [place], Z gave [object] to A" (and equivalent for the BABA case)

    names: List[str]
        list of names, for when flip involves random tokens

    seed: int
        provides reproducibility

    Note that we don't bother flipping the last token in the prompt (IO2), since
    we don't use it for anything (intuitively, we use this function to create
    datasets to provide us with corrupted signals, but we still use the IO2 from
    the original uncorrupted IOI database as our "correct answer", so we don't
    care about what the correct answer (IO2) for the corrupted set is).
    """
    random.seed(seed)
    np.random.seed(seed)
    abba_flip, baba_flip = flip.split(",")
    flip_dict = {
        "ABB": [flip.strip() for flip in abba_flip.split("->")],
        "BAB": [flip.strip() for flip in baba_flip.split("->")],
    }

    new_prompts = []

    for idx, (prompt, template) in enumerate(zip(prompts, templates_by_prompt)):

        flip_orig, flip_new = flip_dict[template[:-1]]

        prompt = copy.copy(prompt)

        # Get indices and original values of first three names int the prompt
        prompt_split = prompt["text"].split(" ")
        orig_names_and_posns = [
            (i, s) for i, s in enumerate(prompt_split) if s in names
        ][:3]
        orig_names = list(zip(*orig_names_and_posns))[1]

        # Get a dictionary of the correspondence between orig names and letters in flip_orig
        # (and get a subdict for those names which are kept in flip_new)
        orig_names_key = {letter: s for s, letter in zip(orig_names, flip_orig)}
        kept_names_key = {k: v for k, v in orig_names_key.items() if k in flip_new}
        # This line will throw an error if flip_orig is wrong (e.g. if it says "SOS" but the
        # S1 and S2 tokens don't actually match
        assert len(orig_names_key) == len(set(flip_orig))

        # Get all random names we'll need, in the form of a dictionary
        sorted_names = sorted(set(names) - set(orig_names), key=names.index)
        sorted_flips = sorted(set(flip_new) - set(flip_orig), key=flip_new.index)
        # TR edit - sample without replacement to avoid duplicate names explicitly requested
        # rand_names = {letter: np.random.choice(sorted_names) for letter in sorted_flips} # with replacement
        selected_names = random.sample(sorted_names, len(sorted_flips))
        rand_names = {
            letter: name for letter, name in zip(sorted_flips, selected_names)
        }

        # Get a "full dictionary" which maps letters in flip_new to the new values they will have
        name_replacement_dict = {**kept_names_key, **rand_names}
        assert len(name_replacement_dict) == len(set(flip_new)), (
            name_replacement_dict,
            flip_new,
        )

        # Populate the new names, with either random names or with the corresponding orig names
        for (i, s), letter in zip(orig_names_and_posns, flip_new):
            prompt_split[i] = name_replacement_dict[letter]

        # Join the prompt back together
        prompt["text"] = " ".join(prompt_split)

        # Change the identity of the S and IO tokens.
        # S token is just same as S2, but IO is a bit messier because it might not be
        # well-defined (it's defined as the unique non-duplicated name of the first
        # two). If it's ill-defined, WLOG set it to be the second name.
        prompt["S"] = name_replacement_dict[flip_new[-1]]
        possible_IOs = [
            name_replacement_dict[letter]
            for letter in flip_new[:2]
            if list(flip_new).count(letter) == 1
        ]
        # Case where IO is well-defined
        if len(possible_IOs) == 1:
            prompt["IO"] = possible_IOs[0]
        # Case where it isn't well-defined
        else:
            prompt["IO"] = name_replacement_dict[flip_new[1]]

        new_prompts.append(prompt)

    return new_prompts


def get_name_idxs(prompts, tokenizer, idx_types=["IO", "S1", "S2"], prepend_bos=False):
    name_idx_dict = dict((idx_type, []) for idx_type in idx_types)
    for prompt in prompts:
        text_split = prompt["text"].split(" ")
        toks = tokenizer.tokenize(" ".join(text_split[:-1]))
        # Get the first instance of IO token
        name_idx_dict["IO"].append(
            toks.index(tokenizer.tokenize(" " + prompt["IO"])[0])
        )
        # Get the first instance of S token
        name_idx_dict["S1"].append(toks.index(tokenizer.tokenize(" " + prompt["S"])[0]))
        # Get the last instance of S token
        name_idx_dict["S2"].append(
            len(toks) - toks[::-1].index(tokenizer.tokenize(" " + prompt["S"])[0]) - 1
        )

    return [
        int(prepend_bos) + t.tensor(name_idx_dict[idx_type]) for idx_type in idx_types
    ]


def get_word_idxs(prompts, word_list, tokenizer):
    """Get the index of the words in word_list in the prompts. Exactly one of the word_list word has to be present in each prompt"""
    idxs = []
    tokenized_words = [
        tokenizer.decode(tokenizer(word)["input_ids"][0]) for word in word_list
    ]
    for prompt in prompts:
        toks = [
            tokenizer.decode(t)
            for t in tokenizer(prompt["text"], return_tensors="pt", padding=True)[
                "input_ids"
            ][0]
        ]
        idx = None
        for i, w_tok in enumerate(tokenized_words):
            if word_list[i] in prompt["text"]:
                try:
                    idx = toks.index(w_tok)
                    if toks.count(w_tok) > 1:
                        idx = len(toks) - toks[::-1].index(w_tok) - 1
                except:
                    idx = toks.index(w_tok)
                    # raise ValueError(toks, w_tok, prompt["text"])
        if idx is None:
            raise ValueError(f"Word {word_list} and {i} not found {prompt}")
        idxs.append(idx)
    return t.tensor(idxs)


def get_end_idxs(toks, tokenizer, name_tok_len=1, prepend_bos=False):
    relevant_idx = int(prepend_bos)
    # if the sentence begins with an end token
    # AND the model pads at the end with the same end token,
    # then we need make special arrangements

    pad_token_id = tokenizer.pad_token_id

    end_idxs_raw = []
    for i in range(toks.shape[0]):
        if pad_token_id not in toks[i][1:]:
            end_idxs_raw.append(toks.shape[1])
            continue
        nonzers = (toks[i] == pad_token_id).nonzero()[relevant_idx][0].item()
        end_idxs_raw.append(nonzers)
    end_idxs = t.tensor(end_idxs_raw)
    end_idxs = end_idxs - 1 - name_tok_len

    for i in range(toks.shape[0]):
        assert toks[i][end_idxs[i] + 1] != 0 and (
            toks.shape[1] == end_idxs[i] + 2 or toks[i][end_idxs[i] + 2] == pad_token_id
        ), (
            toks[i],
            end_idxs[i],
            toks[i].shape,
            "the END idxs aren't properly formatted",
        )

    return end_idxs


def get_idx_dict(ioi_prompts, tokenizer, prepend_bos=False, toks=None):
    (
        IO_idxs,
        S1_idxs,
        S2_idxs,
    ) = get_name_idxs(
        ioi_prompts,
        tokenizer,
        idx_types=["IO", "S1", "S2"],
        prepend_bos=prepend_bos,
    )

    end_idxs = get_end_idxs(
        toks,
        tokenizer,
        name_tok_len=1,
        prepend_bos=prepend_bos,
    )

    punct_idxs = get_word_idxs(ioi_prompts, [",", "."], tokenizer)

    return {
        "IO": IO_idxs,
        "IO-1": IO_idxs - 1,
        "IO+1": IO_idxs + 1,
        "S1": S1_idxs,
        "S1-1": S1_idxs - 1,
        "S1+1": S1_idxs + 1,
        "S2": S2_idxs,
        "end": end_idxs,
        "starts": t.zeros_like(end_idxs),
        "punct": punct_idxs,
    }


class MinimalIOIDataset(t.utils.data.Dataset):
    def __init__(
        self,
        prompt_type: Union[
            str, List[str]
        ],  # if list, then it will be a list of templates
        N=500,
        tokenizer=None,
        prompts=None,
        symmetric=False,
        prefixes=None,
        nb_templates=None,
        prepend_bos=False,
        manual_word_idx=None,
        has_been_flipped: bool = False,
        seed=0,
        device="cuda",
        base_template_type="umc",
    ):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        if not (
            N == 1
            or prepend_bos == False
            or tokenizer.bos_token_id == tokenizer.eos_token_id
        ):
            warnings.warn(
                "Probably word_idx will be calculated incorrectly due to this formatting"
            )
        self.has_been_flipped = has_been_flipped
        assert not (symmetric and prompt_type == "ABC")
        assert (
            (prompts is not None) or (not symmetric) or (N % 2 == 0)
        ), f"{symmetric} {N}"
        self.prompt_type = prompt_type

        # TR edit - deleted use of nb_templates, replaced global variables with my own global variable / to_abba function
        if prompt_type == "ABBA":
            self.templates = BABA_to_ABBA(base_BABA_templates[base_template_type])
        elif prompt_type == "BABA":
            self.templates = base_BABA_templates[base_template_type].copy()
        elif prompt_type == "mixed":
            assert base_template_type not in DONT_MIX
            base_BABA = base_BABA_templates[base_template_type].copy()
            self.templates = base_BABA + BABA_to_ABBA(base_BABA)
            random.shuffle(self.templates)
        elif prompt_type == "ABC":
            self.templates = ABC_TEMPLATES.copy()
        elif prompt_type == "BAC":
            self.templates = BAC_TEMPLATES.copy()
        elif prompt_type == "ABC mixed":
            self.templates = ABC_TEMPLATES.copy() + BAC_TEMPLATES.copy()
            random.shuffle(self.templates)
        elif isinstance(prompt_type, list):
            self.templates = prompt_type
        else:
            raise ValueError(prompt_type)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        self.prefixes = prefixes
        self.prompt_type = prompt_type
        if prompts is None:
            self.ioi_prompts = gen_prompt_uniform(  # list of dict of the form {"text": "Alice and Bob bla bla. Bob gave bla to Alice", "IO": "Alice", "S": "Bob"}
                self.templates,
                NAMES,
                nouns_dict={"[PLACE]": PLACES, "[OBJECT]": OBJECTS},
                N=N,
                symmetric=symmetric,
                prefixes=self.prefixes,
                abc=(prompt_type in ["ABC", "ABC mixed", "BAC"]),
            )
        else:
            assert N == len(prompts), f"{N} and {len(prompts)}"
            self.ioi_prompts = prompts

        all_ids = [prompt["TEMPLATE_IDX"] for prompt in self.ioi_prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])

        small_groups = []
        for group in self.groups:
            if len(group) < 5:
                small_groups.append(len(group))

        self.sentences = [
            prompt["text"] for prompt in self.ioi_prompts
        ]  # a list of strings. Renamed as this should NOT be forward passed

        self.templates_by_prompt = []  # for each prompt if it's ABBA or BABA
        for i in range(N):
            if self.sentences[i].index(self.ioi_prompts[i]["IO"]) < self.sentences[
                i
            ].index(self.ioi_prompts[i]["S"]):
                self.templates_by_prompt.append("ABBA")
            else:
                self.templates_by_prompt.append("BABA")

        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
            for prompt in self.ioi_prompts
        ]
        self.toks = t.Tensor(self.tokenizer(texts, padding=True).input_ids).long()

        self.word_idx = get_idx_dict(
            self.ioi_prompts,
            self.tokenizer,
            prepend_bos=prepend_bos,
            toks=self.toks,
        )
        self.prepend_bos = prepend_bos
        if manual_word_idx is not None:
            self.word_idx = manual_word_idx

        self.N = N
        self.max_len = max(
            [
                len(self.tokenizer(prompt["text"]).input_ids)
                for prompt in self.ioi_prompts
            ]
        )

        self.io_tokenIDs = [
            self.tokenizer.encode(" " + prompt["IO"])[0] for prompt in self.ioi_prompts
        ]
        self.s_tokenIDs = [
            self.tokenizer.encode(" " + prompt["S"])[0] for prompt in self.ioi_prompts
        ]

        self.tokenized_prompts = []

        for i in range(self.N):
            self.tokenized_prompts.append(
                "|".join([self.tokenizer.decode(tok) for tok in self.toks[i]])
            )

        self.device = device
        self.to(device)

    def gen_flipped_prompts(self, flip):
        # Check if it's already been flipped (shouldn't string 2 flips together)
        if self.has_been_flipped:
            warnings.warn(
                "This dataset has already been flipped. Generally, you should try and apply flips in one step, because this can lead to errors."
            )

        # Redefine seed (so it's different depending on what the flip is, e.g. we don't want (IO, RAND) then (S, RAND) to give us the same rand names)
        seed = self.seed + sum(map(ord, list("".join(flip))))

        # Get flipped prompts
        flipped_prompts = gen_flipped_prompts(
            self.ioi_prompts, self.templates_by_prompt, flip, NAMES, seed
        )

        flipped_ioi_dataset = MinimalIOIDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=flipped_prompts,
            prefixes=self.prefixes,
            prepend_bos=self.prepend_bos,
            manual_word_idx=self.word_idx,
            has_been_flipped=True,
            seed=seed,
        )
        return flipped_ioi_dataset

    def copy(self):
        copy_ioi_dataset = MinimalIOIDataset(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=self.ioi_prompts.copy(),
            prefixes=(
                self.prefixes.copy() if self.prefixes is not None else self.prefixes
            ),
        )
        return copy_ioi_dataset

    def __getitem__(self, key):
        # TR edit to support DataLoader
        if isinstance(key, int):
            return self.toks[key], {
                "io": self.io_tokenIDs[key],
                "s": self.s_tokenIDs[key],
                "end_idx": self.word_idx["end"][key],
            }

        # TR edit added in ability to pass a list of idxs
        if isinstance(key, list) or isinstance(key, range):
            sliced_prompts = [self.ioi_prompts[i] for i in key]
        else:
            sliced_prompts = self.ioi_prompts[key]
        sliced_dataset = MinimalIOIDataset(
            prompt_type=self.prompt_type,
            N=len(sliced_prompts),
            tokenizer=self.tokenizer,
            prompts=sliced_prompts,
            prefixes=self.prefixes,
            prepend_bos=self.prepend_bos,
        )
        return sliced_dataset

    def get_batch(self, idxs):
        datas = []
        contexts = []
        for i in idxs:
            data, context = self[i]
            datas.append(data)
            contexts.append(context)

        batch_tensors = t.stack(datas)
        context_keys = contexts[0].keys()
        batch_dicts = {
            key: t.stack([t.tensor(context[key]) for context in contexts])
            for key in context_keys
        }
        return batch_tensors, batch_dicts

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __len__(self):
        return self.N

    def tokenized_prompts(self):
        return self.toks

    def to(self, device):
        self.toks = self.toks.to(device)
        return self


# AttrDict is a dict which allows you to access the keys as attributes
class AttrDict(dict):
    """Custom dictionary class that supports both dictionary and attribute style access."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")

    def __setattr__(self, key, value):
        self[key] = value


# previously I set this up to inherit from hf BatchEncoding, removed to work with Dataloader num_workers
class DummyTokenizer:
    def __init__(self, max_len=5):
        self.max_len = max_len
        self.pad_token_id = 0
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.pad_token = "0"
        self.bos_token = "0"
        self.eos_token = "0"
        self.input_ids = []

    def encode(self, text):
        if isinstance(text, str) and " " not in text[1:]:
            text = text.strip()
            return [int(text) if text.isdigit() else self.pad_token_id]

        if isinstance(text, str):
            text = [text]

        tokens = []
        for prompt in text:
            tokens.append(
                [
                    (int(char) if char.isdigit() else self.pad_token_id)
                    for char in prompt.split()
                ]
            )
        return tokens

    def decode(self, tok):
        if isinstance(tok, int):
            return str(tok)
        if isinstance(tok, t.Tensor):
            return str(tok.item())
        if len(tok) == 1:
            return str(tok[0])

    def tokenize(self, text):
        return [char if char.isnumeric() else self.pad_token for char in text.split()]

    # def __call__(self, text, return_tensors=None, padding=False):
    #     if isinstance(text, str):
    #         text = [text]
    #     encoded_tokens = self.encode(text)
    #     if padding:
    #         padded_tokens = []
    #         for i, encoded in enumerate(encoded_tokens):
    #             padded_tokens.append(
    #                 encoded + [self.pad_token_id] * (self.max_len - len(encoded))
    #             )
    #         input_ids = padded_tokens
    #     else:
    #         input_ids = encoded_tokens

    # return BatchEncoding({"input_ids": input_ids}, tensor_type=return_tensors)
    def __call__(self, text, return_tensors=None, padding=False):
        encoded_tokens = self.encode(text)
        if padding:
            padded_tokens = []
            for encoded in encoded_tokens:
                padded_tokens.append(
                    encoded + [self.pad_token_id] * (self.max_len - len(encoded))
                )
            input_ids = padded_tokens
        else:
            input_ids = encoded_tokens

        # Check if tensors are requested, return them as such
        if return_tensors == "pt":
            input_ids = t.tensor(input_ids, dtype=t.long)

        return AttrDict({"input_ids": input_ids})


if __name__ == "__main__":
    from pprint import pprint

    ioi_dataset = MinimalIOIDataset(
        "mixed",
        N=20,
        tokenizer=DummyTokenizer(),
        prepend_bos=False,
        # symmetric=True,
        base_template_type="spa",
    )
    # ioi_dataset = ioi_dataset.gen_flipped_prompts("ABB -> ABZ, BAB -> BAZ")
    # print("sentences", ioi_dataset.sentences)
    print("word_idx", ioi_dataset.word_idx)
    print("word_idx:")
    pprint(ioi_dataset.word_idx)
    print("io_tokenIDs", ioi_dataset.io_tokenIDs)
    print("s_tokenIDs", ioi_dataset.s_tokenIDs)
    print("tokenized_prompts", ioi_dataset.tokenized_prompts)
    print("groups", ioi_dataset.groups)
    print("max_len", ioi_dataset.max_len)
    print("N", ioi_dataset.N)
    print("templates", ioi_dataset.templates)
    print("templates_by_prompt", ioi_dataset.templates_by_prompt)
    print("prompt_type", ioi_dataset.prompt_type)
    print("prefixes", ioi_dataset.prefixes)
    print("prepend_bos", ioi_dataset.prepend_bos)
    print("device", ioi_dataset.device)
    print("to", ioi_dataset.to(ioi_dataset.device))
    print("copy", ioi_dataset.copy())
    print("len", len(ioi_dataset))
    print("ioi_prompts", ioi_dataset.ioi_prompts)
    print("token strings for display", ioi_dataset.tokenized_prompts)
    print("tokens", ioi_dataset.toks)

    # getitem always expects a range slice, not a single index
    # and it doesn't actually return a nice sentence, it returns a whole dataset
    # Edited behaviour: to support DataLoader, I actually have added an if branche for a single int key dereference
    # which returns: toks, io_tokenIDs, s_tokenIDs
    print("getitem", ioi_dataset[0:1])
    print("getbatch", ioi_dataset.get_batch(range(2)))
    print("TR edited getitem", ioi_dataset[0])
    print("ioi[0]", ioi_dataset.ioi_prompts[0])

    # # Flipping prompts
    # abz_dataset = ioi_dataset.gen_flipped_prompts("ABB -> ABZ, BAB -> BAZ")
    # print("abz", abz_dataset.sentences)

    # %%

    # # Note about the default flipping behaviour
    # # Default Behaviour: xyz can look like xyza as desired, but also xyya, xxya, xyxa, maybe even xxxa
    # # Edited Behaviour: sample without replacement, so xyz will give 3 different names
    # xyz_dataset = ioi_dataset.gen_flipped_prompts("ABB -> XYZ, BAB -> XYZ")
    # print("xyz", xyz_dataset.sentences)

    # # Alternatively if you  want to be strict about the flip without my code change
    # # then changing one letter at a time means you won't get the double letter patterns
    # abz_dataset = ioi_dataset.gen_flipped_prompts("ABB -> ABZ, BAB -> BAZ")
    # xbz_dataset = abz_dataset.gen_flipped_prompts("ABZ -> XBZ, BAZ -> BXZ")
    # print("xbz_alt", xbz_dataset.sentences)
