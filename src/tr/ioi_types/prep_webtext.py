from datasets import load_dataset
from transformers import GPT2Tokenizer


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=21
    )


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("Skylion007/openwebtext", split="train[:100000]")
tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
tokenized_dataset.save_to_disk("data/webtext/tokenized_openwebtext_subset")
