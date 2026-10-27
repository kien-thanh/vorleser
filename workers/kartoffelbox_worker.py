"""Kartoffelbox (SebastianBodza): Chatterbox fine-tuned on German. Runs in envs/chatterbox."""
import torch
import torchaudio

from common import serve

REPO = "SebastianBodza/Kartoffelbox-v0.1"


def load(opts):
    from chatterbox.tts import ChatterboxTTS
    from huggingface_hub import snapshot_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = snapshot_download(REPO, allow_patterns=["*.safetensors", "*.json", "*.pt"])
    model = ChatterboxTTS.from_local(ckpt, device=device)
    exaggeration = float(opts.get("exaggeration", 0.5))
    if opts.get("voice"):
        # Condition once on the reference voice instead of on every call.
        model.prepare_conditionals(opts["voice"], exaggeration=exaggeration)
    gen_kwargs = {k: float(opts[k]) for k in ("cfg_weight", "temperature", "repetition_penalty", "min_p", "top_p")
                  if k in opts}
    default_seed = opts.get("seed")

    def synthesize(text, out, seed=None):
        seed = default_seed if seed is None else seed
        if seed is not None:
            torch.manual_seed(int(seed))
        with torch.inference_mode():
            wav = model.generate(text, exaggeration=exaggeration, **gen_kwargs)
        torchaudio.save(out, wav.cpu(), model.sr)

    return synthesize, model.sr


if __name__ == "__main__":
    serve(load)
