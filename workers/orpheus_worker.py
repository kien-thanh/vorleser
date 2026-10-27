"""Kartoffel-Orpheus 3B (SebastianBodza, German "natural" fine-tune of Orpheus). Runs in envs/orpheus.

Uses the ungated GGUF re-upload, converted once into a plain HF checkpoint under models/ (see
orpheus_convert.py). No voice cloning: pick one of the built-in `speaker`s (Julian, Jakob, Sophie, ...).
"""
import json
import sys
from pathlib import Path

import torch
import soundfile as sf

from common import serve

REPO = "cstr/kartoffel-orpheus-3b-german-natural-GGUF"
GGUF = "kartoffel-orpheus-de-natural-q8_0.gguf"
LOCAL = Path(__file__).resolve().parent.parent / "models" / "kartoffel-orpheus-natural-q8"

START, BOS, END_TEXT, END_HUMAN = 128259, 128000, 128009, 128260
START_AUDIO, END_AUDIO, AUDIO_OFFSET = 128257, 128258, 128266


def _checkpoint():
    if not (LOCAL / "model.safetensors").exists():
        from huggingface_hub import hf_hub_download
        from orpheus_convert import convert
        print(f"Converting {GGUF} -> {LOCAL} (one-time)", file=sys.stderr)
        convert(hf_hub_download(REPO, GGUF), LOCAL)
    return LOCAL


def _codes_to_snac(codes):
    # Each 7-token frame interleaves SNAC's three codebooks as 1 + 2 + 4 slots.
    layers = [[], [], []]
    for i in range(len(codes) // 7):
        f = [codes[7 * i + j] - j * 4096 for j in range(7)]
        layers[0].append(f[0])
        layers[1] += [f[1], f[4]]
        layers[2] += [f[2], f[3], f[5], f[6]]
    return [torch.tensor(l, device="cuda").unsqueeze(0) for l in layers]


def load(opts):
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = _checkpoint()
    speakers = json.loads((ckpt / "speakers.json").read_text())
    speaker = opts.get("speaker", "Julian")
    if speaker not in speakers:
        raise SystemExit(f"Unknown speaker {speaker!r}. Available: {', '.join(speakers)}")
    if opts.get("voice"):
        print("orpheus: no voice cloning, ignoring --voice; use --opt speaker=NAME", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda").eval()
    gen_kwargs = {
        "temperature": float(opts.get("temperature", 0.6)),
        "top_p": float(opts.get("top_p", 0.95)),
        "repetition_penalty": float(opts.get("repetition_penalty", 1.1)),
    }
    default_seed = opts.get("seed")

    def synthesize(text, out, seed=None):
        seed = default_seed if seed is None else seed
        if seed is not None:
            torch.manual_seed(int(seed))
        ids = tokenizer(f"{speaker}: {text}", add_special_tokens=False).input_ids
        ids = torch.tensor([[START, BOS, *ids, END_TEXT, END_HUMAN]], device="cuda")
        with torch.inference_mode():
            gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=4000, do_sample=True,
                                 eos_token_id=END_AUDIO, pad_token_id=END_AUDIO, **gen_kwargs)[0, ids.shape[1]:].tolist()
            if START_AUDIO in gen:
                gen = gen[gen.index(START_AUDIO) + 1:]
            codes = [t - AUDIO_OFFSET for t in gen if t >= AUDIO_OFFSET]
            codes = codes[:len(codes) // 7 * 7]
            if not codes:
                raise RuntimeError("model produced no audio tokens")
            wav = snac.decode(_codes_to_snac(codes))
        sf.write(out, wav.squeeze().float().cpu().numpy(), 24000)

    return synthesize, 24000


if __name__ == "__main__":
    serve(load)
