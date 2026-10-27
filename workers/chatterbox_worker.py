"""Chatterbox Multilingual (Resemble AI) worker. Runs in envs/chatterbox."""
import torch
import torchaudio

from common import serve


def load(opts):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
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
            wav = model.generate(text, language_id="de", exaggeration=exaggeration, **gen_kwargs)
        torchaudio.save(out, wav.cpu(), model.sr)

    return synthesize, model.sr


if __name__ == "__main__":
    serve(load)
