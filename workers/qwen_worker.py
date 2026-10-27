"""Qwen3-TTS 1.7B worker. Runs in envs/qwen.

Modes (picked from options):
  voice=<wav> [+ ref_text]  -> Base model, clones the reference voice (ICL mode with transcript,
                               speaker-embedding-only without)
  design=<description>      -> VoiceDesign model (used by `vorlesen design-voice`)
  otherwise                 -> CustomVoice model with built-in `speaker` and optional `instruct`

Generation runs through qwen_fast (CUDA graphs + torch.compile) unless fast=off; fast=graph skips
compiling (faster startup, ~25% slower generation).
"""
import math
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch: avoids fragmentation OOMs

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from common import serve

REPO = "Qwen/Qwen3-TTS-12Hz-{size}-{variant}"
FRAMES_PER_SECOND = 12.5


def frame_budget(texts):
    """Generous cap on generated frames: ~2x what the core's duration QA still accepts (16 chars/s,
    1.8x + 0.5 s). Without it a babbling segment runs to the default 8192 frames (~11 min)."""
    return math.ceil(FRAMES_PER_SECOND * (2 * (0.3 + max(map(len, texts)) / 16) + 2))


def load(opts):
    from qwen_tts import Qwen3TTSModel

    size = opts.get("size", "1.7B")
    if opts.get("voice"):
        variant = "Base"
    elif opts.get("design"):
        variant = "VoiceDesign"
    else:
        variant = "CustomVoice"
    model = Qwen3TTSModel.from_pretrained(
        REPO.format(size=size, variant=variant),
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        attn_implementation=opts.get("attn", "sdpa"),
    )
    fast = opts.get("fast", "compile")
    if fast != "off" and torch.cuda.is_available():
        import qwen_fast
        qwen_fast.patch(model, mode=fast, talker=True)
    gen_kwargs = {k: opts[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty") if k in opts}
    default_seed = opts.get("seed")

    if variant == "Base":
        prompt = model.create_voice_clone_prompt(
            ref_audio=opts["voice"], ref_text=opts.get("ref_text"),
            x_vector_only_mode=not opts.get("ref_text"),
        )

        def generate(texts):
            return model.generate_voice_clone(text=texts, language="German", voice_clone_prompt=prompt,
                                              max_new_tokens=frame_budget(texts), **gen_kwargs)
    elif variant == "VoiceDesign":
        def generate(texts):
            return model.generate_voice_design(text=texts, language="German", instruct=opts["design"],
                                               max_new_tokens=frame_budget(texts), **gen_kwargs)
    else:
        def generate(texts):
            return model.generate_custom_voice(text=texts, language="German", speaker=opts.get("speaker", "Ryan"),
                                               instruct=opts.get("instruct") or None,
                                               max_new_tokens=frame_budget(texts), **gen_kwargs)

    def synthesize(text, out, seed=None):
        seed = default_seed if seed is None else seed
        if seed is not None:
            torch.manual_seed(int(seed))
        with torch.inference_mode():
            wavs, sr = generate([text])
        sf.write(out, wavs[0], sr)

    def synthesize_batch(texts, outs):
        # One generate() call for several segments: each step reads all weights once, whatever the batch size.
        if default_seed is not None:
            torch.manual_seed(int(default_seed))
        # Pad to a power of two (repeating the shortest text, which finishes early) so compiled code and
        # CUDA graphs only ever see a few batch shapes.
        padded = list(texts) + [min(texts, key=len)] * ((1 << (len(texts) - 1).bit_length()) - len(texts))
        with torch.inference_mode():
            wavs, sr = generate(padded)
        for wav, out in zip(wavs, outs):
            sf.write(out, wav, sr)

    # Sample rate is only known after a generation; the 12Hz tokenizer decodes at 24 kHz.
    return synthesize, 24000, synthesize_batch


if __name__ == "__main__":
    serve(load)
