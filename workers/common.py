"""Shared worker loop. Runs inside an engine's own virtualenv; keep it dependency-free."""
import json
import os
import sys
import traceback


def serve(load):
    """load(options) -> (synthesize(text, out_path, seed=None), sample_rate[, synthesize_batch(texts, out_paths)])."""
    # Libraries print to stdout; keep the protocol channel clean by moving it aside.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    options = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    synthesize, sample_rate, *batch = load(options)
    synthesize_batch = batch[0] if batch else None
    proto.write(json.dumps({"ready": True, "sample_rate": sample_rate, "batch": bool(synthesize_batch)}) + "\n")

    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        try:
            if "batch" in req:
                synthesize_batch([r["text"] for r in req["batch"]], [r["out"] for r in req["batch"]])
            else:
                synthesize(req["text"], req["out"], seed=req.get("seed"))
            proto.write(json.dumps({"ok": True}) + "\n")
        except Exception as e:  # report and keep serving
            traceback.print_exc()
            proto.write(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}) + "\n")
