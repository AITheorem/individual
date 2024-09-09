# %%
import os
from types import SimpleNamespace

import torch as t
from transformers import AutoTokenizer

from tr.analysis.path_patch_pos import calculate_path_patch
from tr.model_setup.model_settings import setup_model

while "\\src" in os.getcwd():
    os.chdir("..")

model = setup_model("gpt2-small")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# %%

io = "Mary"
s = "John"

prompt = f"When {io} and {s} went to the store, {s} gave a drink to "
corrupt = f"When {io} and {s} went to the store, {io} gave a drink to "


toks = tokenizer(prompt, return_tensors="pt")["input_ids"]
toks_corrupt = tokenizer(corrupt, return_tensors="pt")["input_ids"]


orig_data = SimpleNamespace(toks=toks, word_idx={"end": t.tensor([len(toks[0]) - 1])})
corrupt_data = SimpleNamespace(
    toks=toks_corrupt, word_idx={"end": t.tensor([len(toks_corrupt[0]) - 1])}
)

sender_name_filter = ["blocks.7.attn.hook_z"]
sender_heads = {"blocks.7.attn.hook_z": t.tensor([3])}
sender_pos = "end"

receiver_name_filter = ["blocks.9.attn.hook_q"]
receiver_heads = {"blocks.9.attn.hook_q": t.tensor([9])}
receiver_pos = "end"

(path_performance_impact, patched_logit_diffs, ioi_logit_diffs, aba_logit_diffs) = (
    calculate_path_patch(
        model,
        orig_data,
        corrupt_data,
        sender_name_filter,
        sender_heads,
        sender_pos,
        receiver_name_filter,
        receiver_heads,
        receiver_pos,
        label="dummy",
        logit_diff_fn=lambda x, **y: x,
    )
)


io_tok = tokenizer(io)["input_ids"]
io_clean = ioi_logit_diffs[0][-1][io_tok]
io_corrupt = aba_logit_diffs[0][-1][io_tok]
io_patch = patched_logit_diffs[0][io_tok]

s_tok = tokenizer(s)["input_ids"]
s_clean = ioi_logit_diffs[0][-1][s_tok]
s_corrupt = aba_logit_diffs[0][-1][s_tok]
s_patch = patched_logit_diffs[0][s_tok]

print(
    "IO clean",
    io_clean[0].item(),
    "IO corrupt",
    io_corrupt[0].item(),
    "IO patch",
    io_patch[0].item(),
)
print(
    "S clean",
    s_clean[0].item(),
    "S corrupt",
    s_corrupt[0].item(),
    "S patch",
    s_patch[0].item(),
)
print(
    "IO-S clean",
    (io_clean[0] - s_clean[0]).item(),
    "IO-S corrupt",
    (io_corrupt[0] - s_corrupt[0]).item(),
    "IO-S patch",
    (io_patch[0] - s_patch[0]).item(),
)
print("IO patch - IO clean", (io_patch[0] - io_clean[0]).item())
print("S patch - S clean", (s_patch[0] - s_clean[0]).item())

# %%
