"""Speedups for Qwen3-TTS generation (runs in envs/qwen).

Per audio frame (12.5 Hz) the talker (28 layers) runs one step, then the code predictor runs a
full HF `generate()` for the 15 remaining codebooks. Both are dominated by Python/launch overhead,
so they are replaced by fixed-shape loops over static KV caches and captured as CUDA graphs:
  FastCodePredictor  the 15-token inner loop, one graph per batch shape
  FastTalker         the whole per-frame step (talker + sampling + code predictor), one graph,
                     after an ordinary HF prefill of the prompt
Same math and sampling as the original (verified token-identical with greedy decoding).
"""
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from qwen_tts.core.models.modeling_qwen3_tts import DynamicCache, apply_multimodal_rotary_pos_emb

torch._dynamo.config.cache_size_limit = 32  # one compiled variant per batch shape / sampling setting


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _capture(fn):
    # Warmup runs draw random numbers; restore the RNG so a seeded generation is reproducible.
    rng = torch.cuda.get_rng_state()
    g = _capture_graph(fn)
    torch.cuda.set_rng_state(rng)
    return g


def _capture_graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


class FastCodePredictor:
    def __init__(self, cp, mode: str = "graph"):
        self.cp, self.m = cp, cp.model
        cfg = self.m.config
        self.groups = cfg.num_code_groups - 1          # tokens to predict per frame (15)
        self.seq = self.groups + 1                     # 2 prefill positions + 14 decode steps
        self.head_dim = cfg.head_dim
        self.kv_heads = cfg.num_key_value_heads
        self.mode = mode
        pos = torch.arange(self.seq, device=cp.device)[None]
        x = torch.empty(1, dtype=cp.dtype, device=cp.device)
        cos, sin = self.m.rotary_emb(x, pos)           # [1, seq, head_dim]
        self.cos, self.sin = cos[0], sin[0]
        self.graphs = {}
        # compile: inductor fuses the many tiny kernels (norms, rope, residuals); the result is still graph-captured
        self.run = torch.compile(self._predict, fullgraph=True, dynamic=False) if mode == "compile" else self._predict

    def _layers(self, h, K, V, p0: int):
        B, n, _ = h.shape
        cos, sin = self.cos[p0:p0 + n], self.sin[p0:p0 + n]
        for i, layer in enumerate(self.m.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = a.q_norm(a.q_proj(x).view(B, n, -1, self.head_dim)).transpose(1, 2)
            k = a.k_norm(a.k_proj(x).view(B, n, -1, self.head_dim)).transpose(1, 2)
            v = a.v_proj(x).view(B, n, -1, self.head_dim).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            K[i][:, :, p0:p0 + n] = k
            V[i][:, :, p0:p0 + n] = v
            o = F.scaled_dot_product_attention(q, K[i][:, :, :p0 + n], V[i][:, :, :p0 + n],
                                               is_causal=n > 1, scale=a.scaling, enable_gqa=True)
            h = h + a.o_proj(o.transpose(1, 2).reshape(B, n, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.m.norm(h)

    @staticmethod
    def _sample(logits, do_sample, top_k, temperature):
        logits = logits.float()
        if not do_sample:
            return logits.argmax(-1)
        vals, idx = (logits / temperature).topk(top_k, dim=-1)
        probs = vals.softmax(-1)
        # argmax(p / Exp(1)) draws from p, and unlike multinomial it is safe inside a CUDA graph
        choice = (probs / torch.empty_like(probs).exponential_()).argmax(-1, keepdim=True)
        return idx.gather(-1, choice)[:, 0]

    def _predict(self, inp, do_sample: bool, top_k: int, temperature: float):
        B = inp.shape[0]
        n_layers = len(self.m.layers)
        shape = (B, self.kv_heads, self.seq, self.head_dim)
        K = [torch.zeros(shape, dtype=inp.dtype, device=inp.device) for _ in range(n_layers)]
        V = [torch.zeros(shape, dtype=inp.dtype, device=inp.device) for _ in range(n_layers)]
        proj = self.cp.small_to_mtp_projection
        h = self._layers(proj(inp), K, V, 0)
        tok = self._sample(self.cp.lm_head[0](h[:, -1]), do_sample, top_k, temperature)
        toks = [tok]
        for g in range(1, self.groups):
            e = proj(self.m.codec_embedding[g - 1](tok)[:, None])
            h = self._layers(e, K, V, g + 1)
            tok = self._sample(self.cp.lm_head[g](h[:, -1]), do_sample, top_k, temperature)
            toks.append(tok)
        return torch.stack(toks, 1)

    def __call__(self, inputs_embeds, do_sample=True, top_k=50, top_p=1.0, temperature=0.9, **_):
        if top_p is not None and top_p < 1.0:
            raise NotImplementedError("fast code predictor supports top_k sampling only (subtalker_top_p=1.0)")
        key = (tuple(inputs_embeds.shape), bool(do_sample), int(top_k), float(temperature))
        if self.mode == "eager":
            return SimpleNamespace(sequences=self.run(inputs_embeds, *key[1:]))
        if key not in self.graphs:
            static_in, out = inputs_embeds.clone(), {}
            g = _capture(lambda: out.update(seq=self.run(static_in, *key[1:])))
            self.graphs[key] = (g, static_in, out["seq"])
        g, static_in, static_out = self.graphs[key]
        static_in.copy_(inputs_embeds)
        g.replay()
        return SimpleNamespace(sequences=static_out.clone())


class FastTalker:
    """Drop-in for talker.generate() as called by Qwen3TTSForConditionalGeneration.generate()."""
    BUCKET = 256  # KV capacity granularity; one captured graph per (batch, capacity, sampling params)
    MAX_CAP = 4096  # beyond this (huge max_new_tokens) the static cache would not fit: use the stock loop

    def __init__(self, outer, predictor: FastCodePredictor):
        self.t = outer.talker
        self.orig_generate = self.t.generate
        self.pred = predictor
        self.cfg = outer.config.talker_config
        self.graph = None  # (key, CUDAGraph, state) – only the latest is kept, to bound VRAM
        self.eager = False  # debug: run the step without the graph (hooks fire, same results)
        # KV capacity varies per batch; dynamic=None lets dynamo generalize that dim after the first recompile
        self.layers = torch.compile(self._layers, fullgraph=True) if predictor.mode == "compile" else self._layers

    def _layers(self, h, st):
        m, B = self.t.model, h.shape[0]
        pos = (st["cur"] + st["deltas"]).view(1, B, 1).expand(3, B, 1)
        cos, sin = m.rotary_emb(h, pos)
        st["mask"].index_fill_(-1, st["cur"], True)
        for i, layer in enumerate(m.layers):
            a = layer.self_attn
            x = layer.input_layernorm(h)
            q = a.q_norm(a.q_proj(x).view(B, 1, -1, a.head_dim)).transpose(1, 2)
            k = a.k_norm(a.k_proj(x).view(B, 1, -1, a.head_dim)).transpose(1, 2)
            v = a.v_proj(x).view(B, 1, -1, a.head_dim).transpose(1, 2)
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, a.rope_scaling["mrope_section"],
                                                   a.rope_scaling["interleaved"])
            st["K"][i].index_copy_(2, st["cur"], k)
            st["V"][i].index_copy_(2, st["cur"], v)
            # GQA with a mask makes SDPA fall back to its slow math kernel (~1 ms/layer). With one query token,
            # the query heads sharing a KV head can instead be passed as that head's query positions.
            kv_heads = st["K"][i].shape[1]
            o = F.scaled_dot_product_attention(q.reshape(B, kv_heads, -1, a.head_dim), st["K"][i], st["V"][i],
                                               attn_mask=st["mask"], scale=a.scaling)
            h = h + a.o_proj(o.reshape(B, 1, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return m.norm(h)

    def _next_token(self, logits, st, generated, p):
        """HF processors in HF order: repetition penalty, min_new_tokens, suppress_tokens, temperature/top-k."""
        logits = logits.float()
        if p["repetition_penalty"] != 1.0:
            rp = p["repetition_penalty"]
            logits = torch.where(st["seen"], torch.where(logits < 0, logits * rp, logits / rp), logits)
        logits = logits.masked_fill(st["eos_mask"] & (generated < p["min_new_tokens"]), float("-inf"))
        logits = logits + st["suppress"]
        tok = FastCodePredictor._sample(logits, p["do_sample"], p["top_k"], p["temperature"])
        tok = torch.where(st["finished"], st["eos"], tok)  # finished rows emit pad (= eos), like HF
        st["finished"] |= tok == st["eos"]
        st["seen"].scatter_(1, tok[:, None], True)
        return tok

    def _step(self, st, p):
        tok = st["tok"]
        last = self.t.get_input_embeddings()(tok)[:, None]
        codes = self.pred.run(torch.cat((st["past_hidden"], last), 1), p["subtalker_dosample"],
                                   p["subtalker_top_k"], p["subtalker_temperature"])
        cp_emb = self.t.code_predictor.get_input_embeddings()
        h = torch.cat([last] + [cp_emb[i](codes[:, i:i + 1]) for i in range(codes.shape[1])], 1).sum(1, keepdim=True)
        h = h + st["trailing"].index_select(1, st["step"])
        st["frames"].index_copy_(1, st["step"], torch.cat((tok[:, None], codes), 1)[:, None])
        h = self.layers(h, st)
        st["past_hidden"].copy_(h)
        st["step"].add_(1)
        st["tok"].copy_(self._next_token(self.t.codec_head(h[:, -1]), st, st["step"], p))
        st["cur"].add_(1)

    def _state(self, B, cap, dev, dt):
        a = self.t.model.layers[0].self_attn
        kv = (B, self.cfg.num_key_value_heads, cap, a.head_dim)
        H, V = self.cfg.hidden_size, self.cfg.vocab_size
        eos = self.cfg.codec_eos_token_id
        st = dict(
            K=[torch.zeros(kv, dtype=dt, device=dev) for _ in self.t.model.layers],
            V=[torch.zeros(kv, dtype=dt, device=dev) for _ in self.t.model.layers],
            mask=torch.zeros(B, 1, 1, cap, dtype=torch.bool, device=dev),
            trailing=torch.zeros(B, cap, H, dtype=dt, device=dev),
            frames=torch.zeros(B, cap, self.cfg.num_code_groups, dtype=torch.long, device=dev),
            past_hidden=torch.zeros(B, 1, H, dtype=dt, device=dev),
            tok=torch.zeros(B, dtype=torch.long, device=dev),
            seen=torch.zeros(B, V, dtype=torch.bool, device=dev),
            finished=torch.zeros(B, dtype=torch.bool, device=dev),
            deltas=torch.zeros(B, dtype=torch.long, device=dev),
            cur=torch.zeros(1, dtype=torch.long, device=dev),
            step=torch.zeros(1, dtype=torch.long, device=dev),
            eos=torch.tensor(eos, device=dev),
            eos_mask=torch.arange(V, device=dev) == eos,
            suppress=torch.zeros(V, device=dev),
        )
        return st

    def __call__(self, inputs_embeds, attention_mask, trailing_text_hidden, tts_pad_embed, **kw):
        p = dict(do_sample=kw["do_sample"], top_k=int(kw["top_k"]), temperature=float(kw["temperature"]),
                 repetition_penalty=float(kw["repetition_penalty"]), min_new_tokens=int(kw.get("min_new_tokens", 0)),
                 subtalker_dosample=kw["subtalker_dosample"], subtalker_top_k=int(kw["subtalker_top_k"]),
                 subtalker_temperature=float(kw["subtalker_temperature"]))
        if kw.get("top_p", 1.0) < 1.0 or kw.get("subtalker_top_p", 1.0) < 1.0 \
                or kw.get("eos_token_id", self.cfg.codec_eos_token_id) != self.cfg.codec_eos_token_id \
                or inputs_embeds.shape[1] + int(kw["max_new_tokens"]) > self.MAX_CAP:
            return self.orig_generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                      trailing_text_hidden=trailing_text_hidden, tts_pad_embed=tts_pad_embed, **kw)
        B, L0, _ = inputs_embeds.shape
        max_frames = int(kw["max_new_tokens"])
        cap = -(-(L0 + max_frames) // self.BUCKET) * self.BUCKET
        key = (B, cap, tuple(sorted(p.items())), tuple(kw.get("suppress_tokens") or ()))

        if self.graph is None or self.graph[0] != key:
            self.graph = None
            torch.cuda.empty_cache()
            st = self._state(B, cap, inputs_embeds.device, inputs_embeds.dtype)
            if kw.get("suppress_tokens"):
                st["suppress"][torch.tensor(kw["suppress_tokens"], device=st["suppress"].device)] = float("-inf")
            self.graph = (key, _capture(lambda: self._step(st, p)), st)
        _, graph, st = self.graph

        # Prefill with the stock model (padding mask + per-row rope offsets, exactly as in talker.forward).
        delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        position_ids, deltas = self.t.get_rope_index(attention_mask)
        cache = DynamicCache()
        out = self.t.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids,
                           past_key_values=cache, use_cache=True)
        for i, layer in enumerate(cache.layers):
            st["K"][i][:, :, :L0].copy_(layer.keys)
            st["V"][i][:, :, :L0].copy_(layer.values)
        del cache
        st["mask"].zero_()
        st["mask"][:, 0, 0, :L0] = attention_mask.bool()
        st["deltas"].copy_((deltas - delta0).view(-1))
        st["trailing"].copy_(tts_pad_embed.expand_as(st["trailing"]))
        T = min(trailing_text_hidden.shape[1], cap)
        st["trailing"][:, :T] = trailing_text_hidden[:, :T]
        st["seen"].zero_()
        st["finished"].zero_()
        st["cur"].fill_(L0)
        st["step"].zero_()
        h = out.last_hidden_state[:, -1:]
        st["past_hidden"].copy_(h)
        st["tok"].copy_(self._next_token(self.t.codec_head(h[:, -1]), st, st["step"], p))

        n = 0
        while n < max_frames - 1:
            graph.replay() if not self.eager else self._step(st, p)
            n += 1
            if n % 8 == 0 and bool(st["finished"].all()):
                break
        frames = st["frames"][:, :n].clone()
        dummy = (frames.new_zeros(B, 1, 1, dtype=inputs_embeds.dtype),)
        # Shape of HF's GenerateOutput.hidden_states as consumed by the caller: one prefill entry, then per step
        # (hidden_states, codec_ids). Only codec_ids matter to generate_voice_clone & co.
        return SimpleNamespace(hidden_states=[(dummy, None)] + [(dummy, frames[:, j]) for j in range(n)])


def patch(model, mode: str = "graph", talker: bool = False, decode_batch: int = 2):
    """model: qwen_tts.Qwen3TTSModel. mode: graph | compile | eager. talker=True also graphs the talker step."""
    t = model.model.talker
    pred = FastCodePredictor(t.code_predictor, mode)
    t.code_predictor.generate = pred
    if talker:
        t.generate = FastTalker(model.model, pred)

    # The vocoder's memory grows with the number of segments decoded at once; batch 8 would not fit in 8 GB.
    tok = model.model.speech_tokenizer
    decode = tok.decode

    def chunked_decode(items, chunk=decode_batch):
        wavs, fs = [], None
        for i in range(0, len(items), chunk):
            w, fs = decode(items[i:i + chunk])
            wavs += list(w)
        return wavs, fs
    tok.decode = chunked_decode
