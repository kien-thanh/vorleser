"""Model-agnostic TTS engines.

Each engine runs as a worker process inside its own virtualenv (the models have
conflicting dependencies). Protocol, one JSON object per line:
  worker -> {"ready": true, "sample_rate": 24000}
  core   -> {"text": "...", "out": "/path/seg.wav", "seed": 3}   (seed optional, used for retries)
  core   -> {"batch": [{"text": ..., "out": ...}, ...]}           (only if ready said "batch": true)
  worker -> {"ok": true} | {"ok": false, "error": "..."}
To add an engine: write workers/<name>_worker.py using workers/common.py and
register it in ENGINES.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EngineSpec:
    venv: str
    worker: str
    max_chars: int          # longest text the model reliably handles in one go
    defaults: dict = field(default_factory=dict)
    batch_size: int = 1     # segments per generate() call, if the worker supports batching


ENGINES: dict[str, EngineSpec] = {
    "chatterbox": EngineSpec(
        venv="envs/chatterbox", worker="workers/chatterbox_worker.py", max_chars=300,
        defaults={"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
    ),
    "kartoffelbox": EngineSpec(
        venv="envs/chatterbox", worker="workers/kartoffelbox_worker.py", max_chars=300,
        defaults={"exaggeration": 0.5, "cfg_weight": 0.3, "temperature": 0.6},
    ),
    "orpheus": EngineSpec(
        venv="envs/orpheus", worker="workers/orpheus_worker.py", max_chars=300,
        defaults={"speaker": "Julian", "temperature": 0.6},
    ),
    "qwen": EngineSpec(
        venv="envs/qwen", worker="workers/qwen_worker.py", max_chars=400,
        defaults={"speaker": "Ryan"}, batch_size=8,
    ),
}


class Engine:
    def __init__(self, name: str, options: dict, batch_size: int | None = None):
        if name not in ENGINES:
            raise SystemExit(f"Unknown engine {name!r}. Available: {', '.join(ENGINES)}")
        self.name = name
        self.spec = ENGINES[name]
        self.options = {**self.spec.defaults, **{k: v for k, v in options.items() if v is not None}}
        self.proc: subprocess.Popen | None = None
        self.sample_rate: int | None = None
        self.batch_size = batch_size or self.spec.batch_size  # not an option: must not change the work dir hash

    def start(self):
        python = ROOT / self.spec.venv / "bin" / "python"
        env = {**os.environ, "TQDM_DISABLE": "1", "TRANSFORMERS_VERBOSITY": "error"}
        self.proc = subprocess.Popen(
            [str(python), str(ROOT / self.spec.worker), json.dumps(self.options)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, cwd=ROOT, env=env,
        )
        hello = self._read()
        self.sample_rate = hello["sample_rate"]
        if not hello.get("batch"):
            self.batch_size = 1
        return self

    def _read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name} worker exited (code {self.proc.wait()}); see messages above")
        return json.loads(line)

    def synthesize(self, text: str, out: Path, seed: int | None = None):
        req = {"text": text, "out": str(out)}
        if seed is not None:
            req["seed"] = seed
        self._call(req)

    def synthesize_batch(self, items: list[tuple[str, Path]]):
        self._call({"batch": [{"text": text, "out": str(out)} for text, out in items]})

    def _call(self, req: dict):
        self.proc.stdin.write(json.dumps(req) + "\n")
        reply = self._read()
        if not reply.get("ok"):
            raise RuntimeError(f"{self.name}: {reply.get('error')}")

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
