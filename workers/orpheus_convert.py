"""Convert an Orpheus GGUF (CrispASR layout, `orpheus` architecture) into a plain HF Llama checkpoint.

The talker is a stock Llama-3.2-3B; only the GGUF architecture name and tensor prefix are custom,
which is why transformers can't load the file directly. Runs in envs/orpheus.
"""
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors.torch import save_file

TEXT_TOKENIZER = "unsloth/Llama-3.2-3B-Instruct"  # ungated copy of the Llama 3.2 tokenizer

NAMES = {
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
}


def _field(reader, key):
    return reader.fields[key].contents()


def _unpermute(w, n_heads):
    # Undo llama.cpp's rotary interleave permutation of q/k rows.
    return w.reshape(n_heads, w.shape[0] // n_heads // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def convert(gguf_path, out_dir, unpermute_qk=False):
    from transformers import AutoTokenizer, LlamaConfig

    r = GGUFReader(str(gguf_path))
    p = "orpheus.talker."
    n_heads, n_kv = _field(r, p + "n_heads"), _field(r, p + "n_kv_heads")
    config = LlamaConfig(
        vocab_size=_field(r, p + "vocab_size"), hidden_size=_field(r, p + "d_model"),
        intermediate_size=_field(r, p + "ff_dim"), num_hidden_layers=_field(r, p + "n_layers"),
        num_attention_heads=n_heads, num_key_value_heads=n_kv, head_dim=_field(r, p + "head_dim"),
        max_position_embeddings=_field(r, p + "max_pos"), rope_theta=_field(r, p + "rope_theta"),
        rms_norm_eps=_field(r, p + "rms_norm_eps"), tie_word_embeddings=True,
        # Llama-3.2 rope scaling (Orpheus base is Llama-3.2-3B-Instruct).
        rope_scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                      "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
        bos_token_id=128000, eos_token_id=128258, torch_dtype="bfloat16",
    )

    state = {}
    for t in r.tensors:
        name = t.name.removeprefix("talker.")
        w = dequantize(t.data, t.tensor_type).reshape([int(d) for d in reversed(t.shape)])
        if name == "token_embd.weight":
            key = "model.embed_tokens.weight"
        elif name == "output_norm.weight":
            key = "model.norm.weight"
        else:
            _, layer, kind, _ = name.split(".")
            key = f"model.layers.{layer}.{NAMES[kind]}.weight"
            if unpermute_qk and kind in ("attn_q", "attn_k"):
                w = _unpermute(w, n_heads if kind == "attn_q" else n_kv)
        state[key] = torch.from_numpy(np.ascontiguousarray(w)).to(torch.bfloat16)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, out_dir / "model.safetensors", metadata={"format": "pt"})
    config.save_pretrained(out_dir)

    # Text tokens are stock Llama 3 (audio/control tokens are addressed by id). Rebuilding the tokenizer
    # from the GGUF fields loses the byte-level pre-tokenizer and silently drops all spaces.
    AutoTokenizer.from_pretrained(TEXT_TOKENIZER).save_pretrained(out_dir)
    (out_dir / "speakers.json").write_text(json.dumps(_field(r, "orpheus.spk_names")))
    return out_dir


if __name__ == "__main__":
    import sys
    convert(sys.argv[1], sys.argv[2], unpermute_qk="--unpermute" in sys.argv)
