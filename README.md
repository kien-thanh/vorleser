# vorlesen – German EPUB → audiobook (.m4b) with local TTS

    ./vorlesen.sh list    BOOK.epub                      # chapters (* = rendered by default)
    ./vorlesen.sh preview BOOK.epub --chapter 8          # normalized text exactly as the model reads it
    ./vorlesen.sh sample  BOOK.epub --chapter 8 --voice voices/erzaehler.wav   # ~2 min per engine -> samples/
    ./vorlesen.sh render  BOOK.epub --engine chatterbox --voice voices/erzaehler.wav   # -> ~/<Title> - <Author> (<engine>).m4b
    ./vorlesen.sh design-voice NAME "Beschreibung der Stimme"   # new reference voice -> voices/NAME.wav + .txt

`render` is resumable: interrupt anytime, rerun the same command. Segments whose length doesn't fit
their text (cut off / babbling) are re-generated up to 3 times; leftovers are listed in `work/<book>/<engine>-<hash>/qa.log`.
qwen renders 8 segments per model call (`--batch-size N`, 1 = off) and generates through `workers/qwen_fast.py`
(CUDA graphs + torch.compile, ~4x faster than stock qwen_tts in batches). The first run per new batch shape compiles
for about a minute; it is cached on disk afterwards. `--opt fast=graph` skips compiling, `--opt fast=off` uses stock qwen_tts.

## Layout
- `vorlesen/` core pipeline (envs/core): `epub.py` extraction, `normalize_de.py` German text cleanup,
  `chunker.py` segments + pauses, `audio.py` trimming/joining/m4b, `engines.py` engine registry.
- `workers/` one worker per TTS model, each in its own venv (`envs/chatterbox`, `envs/qwen`, `envs/orpheus`) because
  their dependencies conflict. They talk JSON lines over stdin/stdout (see `engines.py`).
- `voices/` reference voices (wav + transcript txt). `erzaehler` was made with Qwen3 VoiceDesign.

## Engine options (`--opt key=value`)
- chatterbox: `exaggeration` (0.5; higher = more emotional), `cfg_weight` (0.5; lower = slower, calmer), `temperature`, `seed`
- kartoffelbox: German Chatterbox fine-tune, same options as chatterbox (defaults `cfg_weight` 0.3, `temperature` 0.6). Gated on HF: needs `hf auth login` + accepted terms
- orpheus: Kartoffel-Orpheus 3B German "natural"; no cloning (`--voice` ignored), `speaker` (Julian, Jakob, Jonas, Sophie, Marie, …; list in `models/kartoffel-orpheus-natural-q8/speakers.json`), `temperature`, `top_p`, `repetition_penalty`, `seed`. Weights come from an ungated GGUF and are converted once into `models/` by `workers/orpheus_convert.py`
- qwen: without `--voice`: `speaker` (Ryan, Aiden, …) and `instruct`; `temperature`, `top_p`, `seed` (`top_p` < 1 falls back to the stock, slower generation loop)

## Adding an engine
Write `workers/<name>_worker.py` with a `load(opts) -> (synthesize(text, out, seed=None), sample_rate)` and
`serve(load)`, create `envs/<name>`, and add an `EngineSpec` to `ENGINES` in `vorlesen/engines.py`.
