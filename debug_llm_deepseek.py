"""Standalone debug + fix runner for the DeepSeek LLM zero-shot path.

Use this when the notebook scoring failed (all source='error' or 'fallback'
in outputs/llm_zeroshot/llm_raw_cache.csv).

It tries 3 increasingly-conservative API call shapes:

A) logprobs=True, top_logprobs=10  (verbalizer trick, best signal)
B) plain text generation, max_tokens=2 (just the digit)
C) plain text + 'rationale' prefix removal heuristic

The script picks the first shape that returns a sensible response on a
3-row probe, then re-scores all rows that have source in {error, fallback}.

Run:
    python debug_llm_deepseek.py --api-key sk-...   # or set env DEEPSEEK_API_KEY
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI, OpenAI

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw"
RUN_DIR = ROOT / "outputs" / "llm_zeroshot"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = RUN_DIR / "llm_raw_cache.csv"
ABSTRACTS = ROOT / "outputs" / "external" / "abstracts_merged_v2.csv"
if not ABSTRACTS.exists():
    ABSTRACTS = ROOT / "outputs" / "external" / "abstracts_merged_v3.csv"
BASE_URL = "https://api.deepseek.com"

SYSTEM = """You are a research assistant specialized in classifying scientific papers by their relevance to Answer Set Programming (ASP) and the broader AI-symbolic agenda (neuro-symbolic AI, neural-network verification, explainable AI built on logic).

Given a paper's title and abstract, you score its ASP / AI-symbolic relevance on an integer scale 1-5.

Scoring rubric:
- 1 = Not related to ASP. Proceedings volumes, workshop summaries, generic verification / type theory / theoretical-CS papers without an ASP component.
- 2 = Loose adjacency to ASP / KR. Argumentation, description logic, default reasoning, planning, formal verification with no ASP.
- 3 = Generic logic / declarative reasoning that touches ASP vocabulary (Prolog, Datalog, abductive reasoning, NMR, qualitative reasoning) but is not ASP-centric.
- 4 = Applied / extending ASP. Papers that use ASP to solve a domain problem, extend ASP with new constructs (probabilities, choice, preferences, learning), or build tools / encodings on top of an ASP solver.
- 5 = Core ASP advances or ASP × AI cross-overs. ASP solver / grounder algorithms (Clingo, DLV, ASP(Q)), formal semantics of ASP, ASP-driven learning of programs / heuristics, neuro-symbolic systems built on ASP. Inside CAV / LICS, label 5 also covers neural-network verification and ML-meets-formal-methods work.

Output ONLY a single digit 1, 2, 3, 4, or 5. Do not add any explanation, prefix, or suffix."""


def build_user(title, abstract):
    title = (str(title) if title is not None else "").strip()
    abstract = (str(abstract) if abstract is not None else "").strip()
    if len(abstract) > 1500:
        abstract = abstract[:1500] + "..."
    if not abstract:
        abstract = "(no abstract available)"
    return f"Title: {title}\nAbstract: {abstract}\nAnswer:"


def messages(title, abstract):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(title, abstract)}]


# ---------------------------------------------------------------------------
# 1. Probe API to decide which call shape works
# ---------------------------------------------------------------------------


def probe(api_key: str, model: str) -> str:
    """Return one of 'logprobs', 'text', or raise RuntimeError."""
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    msgs = messages(
        "Formally Explaining Decision Tree Models with Answer Set Programming.",
        "We propose a formal framework for explaining decision tree predictions using Answer Set Programming.",
    )

    print("[probe] trying with logprobs=True ...")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=4,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
        )
        content = (resp.choices[0].message.content or "").strip()
        lp = resp.choices[0].logprobs
        if lp and lp.content and lp.content[0].top_logprobs:
            tokens = [t.token for t in lp.content[0].top_logprobs]
            print("  logprobs OK. text =", repr(content), "  top tokens =", tokens)
            digits_seen = sum(1 for t in tokens if t.strip() in {"1", "2", "3", "4", "5"})
            if digits_seen >= 1:
                return "logprobs"
            print("  warning: no digits in top_logprobs — falling back to text mode")
        else:
            print("  no logprobs in response — falling back to text mode")
    except Exception as exc:
        print(f"  logprobs request failed: {exc!r}. trying plain text ...")

    print("[probe] trying plain text ...")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=4,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        print("  text OK. content =", repr(content))
        if re.search(r"[1-5]", content):
            return "text"
        raise RuntimeError(f"plain text response did not contain a digit 1-5: {content!r}")
    except Exception as exc:
        raise RuntimeError(f"both logprobs and text probes failed: {exc!r}")


# ---------------------------------------------------------------------------
# 2. Async scoring
# ---------------------------------------------------------------------------


def parse_logprobs(lp):
    if lp is None or not lp.content or not lp.content[0].top_logprobs:
        return None
    digit_logprobs = {d: None for d in "12345"}
    for t in lp.content[0].top_logprobs:
        tok = t.token.strip()
        if tok in digit_logprobs:
            cur = digit_logprobs[tok]
            if cur is None or t.logprob > cur:
                digit_logprobs[tok] = t.logprob
    if all(v is None for v in digit_logprobs.values()):
        return None
    arr = np.array([digit_logprobs[d] if digit_logprobs[d] is not None else -50.0 for d in "12345"], dtype=np.float64)
    arr = arr - arr.max()
    p = np.exp(arr); p = p / p.sum()
    return float(np.dot(p, np.arange(1, 6, dtype=float))), p.astype(np.float32)


def parse_text(text):
    text = (text or "").strip()
    m = re.search(r"[1-5]", text)
    if not m:
        return None
    d = int(m.group(0))
    p = np.zeros(5, dtype=np.float32); p[d - 1] = 1.0
    return float(d), p


async def call(client, mode, model, msgs):
    kwargs = dict(model=model, messages=msgs, max_tokens=8, temperature=0.0)
    if mode == "logprobs":
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 10
    return await client.chat.completions.create(**kwargs)


async def score_one(client, mode, model, row, sem, retries=5):
    msgs = messages(row["title"], row["abstract"])
    last_err = ""
    for attempt in range(retries):
        try:
            async with sem:
                resp = await call(client, mode, model, msgs)
            text_out = (resp.choices[0].message.content or "").strip()
            res = None
            if mode == "logprobs":
                res = parse_logprobs(resp.choices[0].logprobs)
            if res is None:
                res = parse_text(text_out)
            if res is None:
                # Last-resort: score=3, uniform
                p = np.full(5, 0.2, dtype=np.float32)
                return {"score": 3.0, **{f"p{i+1}": float(p[i]) for i in range(5)},
                        "source": "fallback", "text_out": text_out, "error": ""}
            score, p = res
            return {"score": score, **{f"p{i+1}": float(p[i]) for i in range(5)},
                    "source": mode, "text_out": text_out, "error": ""}
        except Exception as exc:
            last_err = repr(exc)
            await asyncio.sleep(2 ** attempt)
    p = np.full(5, 0.2, dtype=np.float32)
    return {"score": 3.0, **{f"p{i+1}": float(p[i]) for i in range(5)},
            "source": "error", "text_out": "", "error": last_err}


async def score_dataset(df, split, mode, api_key, model, concurrency, flush_every):
    if CACHE_PATH.exists():
        cache = pd.read_csv(CACHE_PATH)
    else:
        cache = pd.DataFrame()
    # Re-do rows where source in {error, fallback} OR not in cache yet
    if not cache.empty:
        bad = cache["source"].isin({"error", "fallback"}) | cache["source"].isna()
        cache_good = cache[~bad].copy()
        done_keys = set(zip(cache_good.get("source_split", pd.Series([])).astype(str),
                            cache_good.get("id", pd.Series([])).astype(int))) if len(cache_good) else set()
    else:
        cache_good = pd.DataFrame()
        done_keys = set()

    targets = df.assign(source_split=split)
    pending = targets[~targets.apply(lambda r: (split, int(r["id"])) in done_keys, axis=1)]
    print(f"{split}: cache_good={sum(1 for k in done_keys if k[0]==split)} pending={len(pending)}")
    if len(pending) == 0:
        return cache_good[cache_good["source_split"] == split].copy()

    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
    sem = asyncio.Semaphore(concurrency)
    rows_buffer = []
    t0 = time.time()
    coros = [score_one(client, mode, model, r, sem) for _, r in pending.iterrows()]
    for i in range(0, len(coros), flush_every):
        chunk = await asyncio.gather(*coros[i:i + flush_every])
        for (_, r), res in zip(list(pending.iloc[i:i + flush_every].iterrows()), chunk):
            rows_buffer.append({"source_split": split, "id": int(r["id"]),
                                "title": str(r["title"]), **res})
        partial = pd.DataFrame(rows_buffer)
        merged = pd.concat([cache_good, partial], ignore_index=True) if len(cache_good) else partial
        # Drop duplicates - keep last (newer good row beats older bad)
        merged = merged.drop_duplicates(["source_split", "id"], keep="last")
        merged.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
        rate = (i + flush_every) / max(time.time() - t0, 1e-6)
        n_err = sum(1 for x in rows_buffer if x["source"] == "error")
        n_fb = sum(1 for x in rows_buffer if x["source"] == "fallback")
        print(f"  {split}: flushed {min(i+flush_every, len(coros))}/{len(coros)}  "
              f"rate={rate:.1f} req/s  errors={n_err} fallback={n_fb}")
    final = pd.read_csv(CACHE_PATH)
    return final[final["source_split"] == split].copy()


# ---------------------------------------------------------------------------
# 3. Driver
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    p.add_argument("--model", default="deepseek-v4-flash",
                   help="alternative: deepseek-chat, deepseek-v4-pro")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--mode", default="auto", choices=["auto", "logprobs", "text"])
    return p.parse_args()


def main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set --api-key or DEEPSEEK_API_KEY env var.")

    if args.mode == "auto":
        mode = probe(args.api_key, args.model)
    else:
        mode = args.mode
    print(f"[main] using mode = {mode}")

    train = pd.read_csv(DATA / "train.csv")
    public = pd.read_csv(DATA / "public_test.csv")
    private = pd.read_csv(DATA / "private_test.csv")
    abstracts = pd.read_csv(ABSTRACTS)
    abs_map = abstracts[["source_split", "id", "abstract", "has_abstract"]]

    def attach(df, split):
        df = df.copy()
        df["source_split"] = split
        out = df.merge(abs_map, on=["source_split", "id"], how="left")
        out["abstract"] = out["abstract"].fillna("")
        return out

    train_full = attach(train, "train").reset_index(drop=True)
    public_full = attach(public, "public_test").reset_index(drop=True)
    private_full = attach(private, "private_test").reset_index(drop=True)

    asyncio.run(score_dataset(train_full, "train", mode, args.api_key, args.model,
                              args.concurrency, args.flush_every))
    asyncio.run(score_dataset(public_full, "public_test", mode, args.api_key, args.model,
                              args.concurrency, args.flush_every))
    asyncio.run(score_dataset(private_full, "private_test", mode, args.api_key, args.model,
                              args.concurrency, args.flush_every))

    print()
    print("=== Final cache breakdown ===")
    cache = pd.read_csv(CACHE_PATH)
    print(cache.groupby(["source_split", "source"]).size().unstack(fill_value=0))
    print()
    print("Score stats:")
    print(cache.groupby("source_split")["score"].agg(["count", "mean", "std", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
