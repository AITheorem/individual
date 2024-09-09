# %%
import torch as t
from torch import device
from transformer_lens import HookedTransformer, HookedTransformerConfig


def load_model(
    extra_hooks=False,
    n_layers=1,
    d_head=6,
    d_mlp=6,
    d_model=6,
    n_heads=3,
    d_vocab=8,
    attn_only=False,
    normalization_type="LN",
    dtype=t.float32,
    n_ctx=32,
    gpt2_inherit=None,  # "EU", "EUpos"
    seed=1,
):
    cfg_dict = {
        "act_fn": "relu",  # gpt2 uses gelu_new
        "attention_dir": "causal",
        "attn_only": attn_only,
        "attn_types": None,  # for local attention
        "checkpoint_index": None,
        "checkpoint_label_type": None,
        "checkpoint_value": None,
        "d_head": d_head,
        "d_mlp": d_mlp,  # defaults to 4 * d_model in usual transformers
        "d_model": d_model,
        "d_vocab": d_vocab,
        "d_vocab_out": d_vocab,
        "default_prepend_bos": True,
        "device": device("cuda" if t.cuda.is_available() else "cpu"),
        "dtype": dtype,
        "eps": 1e-05,
        "final_rms": False,  # related to SoLU
        "from_checkpoint": False,
        "gated_mlp": False,
        "init_mode": "gpt2",
        "init_weights": True,
        # "initializer_range": 0.02886751345948129, # default 0.8 / sqrt(d_model)
        # "model_name": "custom",
        "n_ctx": n_ctx,  # context ie. max seq length usually 1024, reduced for efficiency
        "n_devices": 1,
        "n_heads": n_heads,
        "n_key_value_heads": None,
        "n_layers": n_layers,
        # "n_params": 84934656,
        "normalization_type": normalization_type,  # options available for LNPre (no w and b) or None
        # "original_architecture": "GPT2LMHeadModel",
        "parallel_attn_mlp": False,
        "positional_embedding_type": "standard",  # "shortformer" only adds pos to K and Q, not V and MLP
        "post_embedding_ln": False,
        # "rotary_adjacent_pairs": False,  # only relevant for rotary embeddings
        # "rotary_base": 10000,
        # "rotary_dim": None,
        "scale_attn_by_inverse_layer_idx": False,  # used by Stanford-mistral models (not Mistral-AI)
        "seed": seed,
        # "tokenizer_name": "gpt2",
        # "tokenizer_prepends_bos": False,  # this is not a setting but a description of how tokenizer behaves
        "trust_remote_code": False,
        "use_attn_in": extra_hooks,
        "use_attn_result": extra_hooks,
        "use_attn_scale": True,
        "use_hook_mlp_in": extra_hooks,
        "use_hook_tokens": extra_hooks,
        "use_local_attn": False,
        "use_split_qkv_input": extra_hooks,
        "window_size": None,
    }

    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    model = HookedTransformer(cfg, tokenizer=None)

    if gpt2_inherit:
        gpt2_small = HookedTransformer.from_pretrained("gpt2-small")
        if "E" in gpt2_inherit:
            assert model.W_E.data.shape == gpt2_small.W_E.data.shape
            model.W_E.data = gpt2_small.W_E.data.clone()
            model.W_E.requires_grad = False
        if "U" in gpt2_inherit:
            assert model.W_U.data.shape == gpt2_small.W_U.data.shape
            model.W_U.data = gpt2_small.W_U.data.clone()
            model.W_U.requires_grad = False
        if "pos" in gpt2_inherit:
            assert model.W_pos.data.shape == gpt2_small.W_pos.data[:n_ctx].shape
            model.W_pos.data = gpt2_small.W_pos.data[:n_ctx].clone()
            model.W_pos.requires_grad = False
        # Confirm all the inputs were actually valid
        assert gpt2_inherit.replace("pos", "").replace("E", "").replace("U", "") == ""

    return model


def setup_model(
    name="gpt2-small",
    dtype="float32",
    refactored=True,
    checkpoint_index=None,
    # use_attn_result=False,
):
    model = HookedTransformer.from_pretrained(
        name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=refactored,
        device=device("cuda" if t.cuda.is_available() else "cpu"),
        dtype=dtype,
        default_prepend_bos=False,  # this argument doesn't actually seem to do anything when True anyway
        checkpoint_index=checkpoint_index,
        # use_attn_result=use_attn_result,
    )

    return model
