# this code has been adapted from Wang et al. (2022), wang2022interpretability
# we also use the same code with slightly different adaptations in min_ioi.py

# %%
import copy
import random
import re
import warnings
from typing import List, Optional, Union

import numpy as np
import torch as t
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer

men = [
    "Adam",
    "Alan",
    "Alex",
    "Anderson",
    "Andrew",
    "Anthony",
    "Arthur",
    "Carter",
    "Charles",
    "Charlie",
    "Christian",
    "Christopher",
    "Crew",
    "Grant",
    "Henry",
    "Jacob",
    "Jamie",
    "Jeremy",
    "Jordan",
    "Joseph",
    "Joshua",
    "Justin",
    "Kyle",
    "Leon",
    "Louis",
    "Matthew",
    "Patrick",
    "Russell",
]  # more men's names were used in original, I just picked length to match women's names
women = [
    "Alice",
    "Amy",
    "Anna",
    "Collins",
    "Crystal",
    "Elizabeth",
    "Emily",
    "Eva",
    "Georgia",
    "Jane",
    "Jennifer",
    "Jessica",
    "Kate",
    "Kelly",
    "Laura",
    "Lisa",
    "Madison",
    "Maria",
    "Mary",
    "Michelle",
    "Morgan",
    "Rachel",
    "River",
    "Rose",
    "Ruby",
    "Sarah",
    "Taylor",
    "Victoria",
]
NAMES = men + women
syn_options_m = ["he"]  # tok 339
syn_options_w = ["she"]  # tok 673

unused_men = [
    "Aaron",
    "Andre",
    "Andy",
    "Austin",
    "Blake",
    "Brandon",
    "Brian",
    "Clark",
    "Cole",
    "Connor",
    "Daniel",
    "David",
    "Dean",
    "Edward",
    "Eric",
    "Ford",
    "Frank",
    "George",
    "Graham",
    "Ian",
    "Jack",
    "Jake",
    "James",
    "Jason",
    "Jay",
    "John",
    "Jonathan",
    "Kevin",
    "Lewis",
    "Luke",
    "Marco",
    "Marcus",
    "Mark",
    "Martin",
    "Max",
    "Michael",
    "Paul",
    "Peter",
    "Prince",
    "Richard",
    "Robert",
    "Roman",
    "Ryan",
    "Scott",
    "Sean",
    "Simon",
    "Stephen",
    "Steven",
    "Sullivan",
    "Thomas",
    "Tyler",
    "Warren",
    "William",
]

# NOTE - going back to using all names now that synonym analysis is done
NAMES = NAMES + unused_men
men = NAMES
women = NAMES

BACBA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [C] gave a [OBJECT] to [B] and [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [C] gave a [OBJECT] to [B] and [A]",
    "Then, [B] and [A] were working at the [PLACE]. [C] decided to give a [OBJECT] to [B] and [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [C] wanted to give a [OBJECT] to [B] and [A]",
    "Then, [B] and [A] had a long argument, and afterwards [C] said to [B] and [A]",
    "After [B] and [A] went to the [PLACE], [C] gave a [OBJECT] to [B] and [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [C] decided to give it to [B] and [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [C] decided to give the [OBJECT] to [B] and [A]",
    "While [B] and [A] were working at the [PLACE], [C] gave a [OBJECT] to [B] and [A]",
    "While [B] and [A] were commuting to the [PLACE], [C] gave a [OBJECT] to [B] and [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [C] gave a [OBJECT] to [B] and [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [C] gave a [OBJECT] to [B] and [A]",
    "Then, [B] and [A] had a long argument. Afterwards [C] said to [B] and [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [C] gave it to [B] and [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [C] gave it to [B] and [A]",
]

ABC_TEMPLATES = [
    "Then, [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "Afterwards [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "When [A], [B] and [C] arrived at the [PLACE], [B] and [C] gave a [OBJECT] to [A]",
    "Friends [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
]

BAC_TEMPLATES = [
    template.replace("[B]", "[A]", 1).replace("[A]", "[B]", 1)
    for template in ABC_TEMPLATES
]

BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BSPACEABA_TEMPLATES = [
    "Then, [B] went with [A] to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] had a lot of fun with [A] at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] worked with [A] at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] thought about going to the [PLACE] with [A]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] had a long argument with [A], and afterwards [B] said to [A]",
    "After [B] went to the [PLACE] with [A], [B] gave a [OBJECT] to [A]",
    # "When [B] got a [OBJECT] at the [PLACE] with [A], [B] decided to give it to [A]", # del
    # "When [B] got a [OBJECT] at the [PLACE] with [A], [B] decided to give the [OBJECT] to [A]", # del
    "While [B] was working with [A] at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] was commuting to the [PLACE] with [A], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] went to the [PLACE] with [A]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] went to the [PLACE] with [A]. [B] gave a [OBJECT] to [A]",
    "Then, [B] had a long argument with [A]. Afterwards [B] said to [A]",
    "The [PLACE] [B] went to with [A] had a [OBJECT]. [B] gave it to [A]",
    # "Friends [B] found a [OBJECT] at the [PLACE] with [A]. [B] gave it to [A]", # del
]

BABA_LONG_TEMPLATES = [
    "Then in the morning, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After taking a long break [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While spending time together [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While spending time together [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch in the afternoon, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, while spending time together [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning afterwards, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The local big [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends separated at birth [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_LATE_IOS = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument and after that [B] said to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
]

BABA_EARLY_IOS = [
    "Then [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] had a lot of fun at the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] were working at the [PLACE], and [B] decided to give a [OBJECT] to [A]",
    "Then [B] and [A] were thinking about going to the [PLACE], and [B] wanted to give a [OBJECT] to [A]",
    "Then [B] and [A] had a long argument, and after that [B] said to [A]",
    "After the lunch [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Afterwards [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] had a long argument, and afterwards [B] said to [A]",
]

ABBA_TEMPLATES = BABA_TEMPLATES[:]
ASPACEBBA_TEMPLATES = BSPACEABA_TEMPLATES[:]
ABBA_LATE_IOS = BABA_LATE_IOS[:]
ABBA_EARLY_IOS = BABA_EARLY_IOS[:]

for TEMPLATES in [ABBA_TEMPLATES, ABBA_LATE_IOS, ABBA_EARLY_IOS, ASPACEBBA_TEMPLATES]:
    for i in range(len(TEMPLATES)):
        first_clause = True
        for j in range(1, len(TEMPLATES[i]) - 1):
            if TEMPLATES[i][j - 1 : j + 2] == "[B]" and first_clause:
                TEMPLATES[i] = TEMPLATES[i][:j] + "A" + TEMPLATES[i][j + 1 :]
            elif TEMPLATES[i][j - 1 : j + 2] == "[A]" and first_clause:
                first_clause = False
                TEMPLATES[i] = TEMPLATES[i][:j] + "B" + TEMPLATES[i][j + 1 :]

combined_templates = (
    BABA_TEMPLATES
    + ABBA_TEMPLATES
    + BABA_LATE_IOS
    + ABBA_LATE_IOS
    + BABA_EARLY_IOS
    + ABBA_EARLY_IOS
)


VERBS = [" tried", " said", " decided", " wanted", " gave"]

PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]

OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]


def gen_prompt_uniform(
    templates: list[str],
    names,
    nouns_dict,
    N,
    symmetric,
    prefixes=None,
    abc=False,
):
    nb_gen = 0
    ioi_prompts = []
    while nb_gen < N:
        temp = random.choice(templates)
        temp_id = templates.index(temp)
        name_1 = ""
        name_2 = ""
        name_3 = ""
        ioi_prompt = {}
        while len(set([name_1, name_2, name_3])) < 3:
            name_3 = random.choice(names)
            # TR edit - force m / f pairing of names
            if nb_gen % 2 == 0:
                name_1 = random.choice(men)  # io
                name_2 = random.choice(women)  # s
                mf_io = 0  # m
                mf_s = 1  # w
            elif nb_gen % 2 == 1:
                name_1 = random.choice(women)  # io
                name_2 = random.choice(men)  # s
                mf_io = 1  # w
                mf_s = 0  # m

        nouns = {}
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
        ioi_prompt["mf_io"] = mf_io
        ioi_prompt["mf_s"] = mf_s
        ioi_prompts.append(ioi_prompt)
        if abc:
            ioi_prompts[-1]["C"] = name_3

        nb_gen += 1

        if symmetric and nb_gen < N:
            prompt2 = prompt.replace("[A]", name_2)
            prompt2 = prompt2.replace("[B]", name_1)
            prompt2 = pref + prompt2
            prompt2_dict = copy.copy(ioi_prompt)
            prompt2_dict["text"] = prompt2
            prompt2_dict["IO"] = name_2
            prompt2_dict["S"] = name_1
            prompt2_dict["mf_io"] = mf_s
            prompt2_dict["mf_s"] = mf_io
            ioi_prompts.append(prompt2_dict)
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
        rand_names = {letter: np.random.choice(sorted_names) for letter in sorted_flips}

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
        "N1": t.min(IO_idxs, S1_idxs),
        "N2": t.max(IO_idxs, S1_idxs),
        "N1+1": t.min(IO_idxs, S1_idxs) + 1,
    }


class SynIOIDataset(Dataset):
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

        if nb_templates is None:
            nb_templates = (
                len(BABA_TEMPLATES) + len(ABBA_TEMPLATES)
                if "mixed" in prompt_type
                else len(BABA_TEMPLATES)
            )

        if isinstance(nb_templates, list):
            index_list = nb_templates
        else:
            index_list = list(range(nb_templates))

        if prompt_type == "ABBA":
            self.templates = [ABBA_TEMPLATES[i] for i in index_list]
        elif prompt_type == "BABA":
            self.templates = [BABA_TEMPLATES[i] for i in index_list]
        elif prompt_type == "mixed":
            self.templates = [
                (BABA_TEMPLATES[i // 2] if i % 2 == 0 else ABBA_TEMPLATES[i // 2])
                for i in index_list
            ]
            random.shuffle(self.templates)
        elif prompt_type == "shortandlong":
            self.templates = [combined_templates[i] for i in index_list]
            random.shuffle(self.templates)
        elif prompt_type == "ABC":
            self.templates = [ABC_TEMPLATES[i] for i in index_list]
        elif prompt_type == "BAC":
            self.templates = [BAC_TEMPLATES[i] for i in index_list]
        elif prompt_type == "ABC mixed":
            self.templates = [
                (ABC_TEMPLATES[i // 2] if i % 2 == 0 else BAC_TEMPLATES[i // 2])
                for i in index_list
            ]
            random.shuffle(self.templates)
        elif prompt_type == "BACBA":
            self.templates = [BACBA_TEMPLATES[i] for i in index_list]
        elif isinstance(prompt_type, list):
            self.templates = [
                prompt_type[i] for i in index_list if i < len(prompt_type)
            ]
        else:
            raise ValueError(prompt_type)  # combined may need to change to shortandlong

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
                abc="C" in prompt_type,
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

        flipped_ioi_dataset = SynIOIDataset(
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
        flipped_ioi_dataset = replace_synonyms_in_matching_positions(
            flipped_ioi_dataset, self
        )

        return flipped_ioi_dataset

    def copy(self):
        copy_ioi_dataset = SynIOIDataset(
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
                "io": self.io_tokenIDs[key],  # type: ignore
                "s": self.s_tokenIDs[key],  # type: ignore
                "end_idx": self.word_idx["end"][key],
                "end-1_idx": self.word_idx["end"][key] - 1,
                "s2_idx": self.word_idx["S2"][key],
                "s1_idx": self.word_idx["S1"][key],
                "io_idx": self.word_idx["IO"][key],
                "start_idx": self.word_idx["starts"][key],
                "and_idx": max(self.word_idx["IO"][key], self.word_idx["S1"][key]) + 1,
                "mf_io": self.ioi_prompts[key]["mf_io"],
                "mf_s": self.ioi_prompts[key]["mf_s"],
                "abbababa": self.templates_by_prompt[key],
            }

        # TR edit added in ability to pass a list of idxs
        if isinstance(key, list) or isinstance(key, range):
            sliced_prompts = [self.ioi_prompts[i] for i in key]
        else:
            sliced_prompts = self.ioi_prompts[key]
        sliced_dataset = SynIOIDataset(
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


def replace_synonyms(ioi_dataset, syn_proportion):
    """This will work with the standard IOI Dataset, HOWEVER
    The SynIOIDataset ensures that each prompt has one m / one f name
    This all happens outside the actual dataset (I needed to change gen_prompt_uniform and Names)
    So it wouldn't be hard to incorporate this into the standard IOI Dataset.
    """
    ioi_dataset.syn_tokenIDs = [None] * ioi_dataset.N
    for prompt_info in ioi_dataset.ioi_prompts:
        prompt_info["syn"] = None

    for i in range(int(ioi_dataset.N * syn_proportion)):
        s = ioi_dataset.ioi_prompts[i]["S"]
        io = ioi_dataset.ioi_prompts[i]["IO"]
        if s in men:
            syn = random.choice(syn_options_m)
            assert io not in men
        elif s in women:
            syn = random.choice(syn_options_w)
            assert io not in women
        else:
            raise ValueError(f"Name {s} not found in m or f list")
        ioi_dataset.ioi_prompts[i]["syn"] = syn
        syn_token = ioi_dataset.tokenizer.encode(" " + syn)[0]
        s2_idx = ioi_dataset.word_idx["S2"][i].item()
        ioi_dataset.toks[i][s2_idx] = syn_token
        ioi_dataset.syn_tokenIDs[i] = syn_token

    return ioi_dataset


def replace_synonyms_in_matching_positions(new_dataset, old_dataset):
    # check if the old_dataset has the syn_tokenIDs attribute
    if not hasattr(old_dataset, "syn_tokenIDs"):
        return new_dataset

    new_dataset.syn_tokenIDs = [None] * new_dataset.N
    for prompt_info in new_dataset.ioi_prompts:
        prompt_info["syn"] = None

    for i in range(new_dataset.N):
        if old_dataset.syn_tokenIDs[i] is None:
            continue

        s = new_dataset.ioi_prompts[i]["S"]
        io = new_dataset.ioi_prompts[i]["IO"]
        if s in men:
            syn = random.choice(syn_options_m)
            assert io not in men
        elif s in women:
            syn = random.choice(syn_options_w)
            assert io not in women
        else:
            raise ValueError(f"Name {s} not found in m or f list")
        new_dataset.ioi_prompts[i]["syn"] = syn
        syn_token = new_dataset.tokenizer.encode(" " + syn)[0]
        s2_idx = new_dataset.word_idx["S2"][i].item()
        new_dataset.toks[i][s2_idx] = syn_token
        new_dataset.syn_tokenIDs[i] = syn_token

    return new_dataset


def randomize_names(toks, context, tokenizer, preserve_mf=True):
    corrupt_toks = toks.clone()
    for prompt, io, s, mf_io, mf_s in zip(
        corrupt_toks, context["io"], context["s"], context["mf_io"], context["mf_s"]
    ):
        name_io = ""
        name_s = ""
        if not preserve_mf:
            while len(set([name_io, name_s])) < 2:
                name_io = random.choice(NAMES)
                name_s = random.choice(NAMES)
        elif mf_io == 0:
            name_io = random.choice(men)
            name_s = random.choice(women)
            assert mf_s == 1
        elif mf_io == 1:
            name_io = random.choice(women)
            name_s = random.choice(men)
            assert mf_s == 0
        name_io_tok = tokenizer.encode(" " + name_io)[0]
        name_s_tok = tokenizer.encode(" " + name_s)[0]
        prompt[prompt == io] = name_io_tok
        prompt[prompt == s] = name_s_tok
    return corrupt_toks


class CombinedDataset(Dataset):
    def __init__(self, datasets: list):
        self.datasets = datasets
        self.N = sum([len(dataset) for dataset in datasets])
        self.tokenizer = datasets[0].tokenizer
        self.prefixes = datasets[0].prefixes
        self.prepend_bos = datasets[0].prepend_bos
        self.prompt_type = [dataset.prompt_type for dataset in datasets]
        self.ioi_prompts = []
        for dataset in datasets:
            assert dataset.prefixes == self.prefixes
            assert dataset.prepend_bos == self.prepend_bos
            self.ioi_prompts += dataset.ioi_prompts
        toks = [dataset.toks for dataset in datasets]
        max_size = max([tok.size(1) for tok in toks])
        padding_value = self.tokenizer.pad_token_id
        padded_toks = []
        for tok in toks:
            padded_toks.append(
                F.pad(tok, (0, max_size - tok.size(1)), value=padding_value)
            )
        self.toks = t.cat(padded_toks, dim=0)

    def __getitem__(self, key):
        if isinstance(key, int):
            toks = self.toks[key]
            for dataset in self.datasets:
                if key < len(dataset):
                    _, context = dataset[key]
                    return toks, context
                key -= len(dataset)

        if isinstance(key, list) or isinstance(key, range):
            sliced_prompts = [self.ioi_prompts[i] for i in key]
        else:
            sliced_prompts = self.ioi_prompts[key]
        sliced_dataset = SynIOIDataset(
            prompt_type=self.prompt_type,
            N=len(sliced_prompts),
            tokenizer=self.tokenizer,
            prompts=sliced_prompts,
            prefixes=self.prefixes,
            prepend_bos=self.prepend_bos,
        )
        return sliced_dataset

    def __len__(self):
        return self.N


if __name__ == "__main__":
    ioi_dataset = SynIOIDataset("BACBA", N=10)
    # replace_synonyms(ioi_dataset, 0.5)
    # print("syn_tokenIDs", ioi_dataset.syn_tokenIDs)
    print("sentences", ioi_dataset.sentences)
    print("word_idx", ioi_dataset.word_idx)
    print("io_tokenIDs", ioi_dataset.io_tokenIDs)
    print("s_tokenIDs", ioi_dataset.s_tokenIDs)
    print("tokenized_prompts", ioi_dataset.tokenized_prompts)
    print("tokenized_prompts splitting", ioi_dataset.tokenized_prompts[0].split("|"))
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
    print("toks", ioi_dataset.toks)
    print("ioi_prompts", ioi_dataset.ioi_prompts[:2])
    # print(
    #     "gen_flipped_prompts", ioi_dataset.gen_flipped_prompts("ABB -> XYZ, BAB -> XYZ")
    # )
    # print("copy", ioi_dataset.copy())
    # print("getitem", ioi_dataset[0:1])
    # print("len", len(ioi_dataset))
    # print("tokenized_prompts", ioi_dataset.tokenized_prompts)

    # toks = ioi_dataset.toks[:2]
    # context = {
    #     "mf_io": [0, 1],
    #     "mf_s": [1, 0],
    #     "io": ioi_dataset.io_tokenIDs[:2],
    #     "s": ioi_dataset.s_tokenIDs[:2],
    # }
    # tokenizer = ioi_dataset.tokenizer

    # corrupt_toks = randomize_names(toks, context, tokenizer, preserve_mf=True)
    print("get_item", ioi_dataset[0])

    mixed_dataset = SynIOIDataset("mixed", N=10)
    combined_dataset = CombinedDataset([ioi_dataset, mixed_dataset])
    print("combined_dataset", combined_dataset[0])


# %%
