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

SYSTEM_V1 = """You are a research assistant specialized in classifying scientific papers by their relevance to Answer Set Programming (ASP) and the broader AI-symbolic agenda (neuro-symbolic AI, neural-network verification, explainable AI built on logic).

Given a paper's title and abstract, you score its ASP / AI-symbolic relevance on an integer scale 1-5.

Scoring rubric:
- 1 = Not related to ASP. Proceedings volumes, workshop summaries, generic verification / type theory / theoretical-CS papers without an ASP component.
- 2 = Loose adjacency to ASP / KR. Argumentation, description logic, default reasoning, planning, formal verification with no ASP.
- 3 = Generic logic / declarative reasoning that touches ASP vocabulary (Prolog, Datalog, abductive reasoning, NMR, qualitative reasoning) but is not ASP-centric.
- 4 = Applied / extending ASP. Papers that use ASP to solve a domain problem, extend ASP with new constructs (probabilities, choice, preferences, learning), or build tools / encodings on top of an ASP solver.
- 5 = Core ASP advances or ASP × AI cross-overs. ASP solver / grounder algorithms (Clingo, DLV, ASP(Q)), formal semantics of ASP, ASP-driven learning of programs / heuristics, neuro-symbolic systems built on ASP. Inside CAV / LICS, label 5 also covers neural-network verification and ML-meets-formal-methods work.

Examples:

Title: Proceedings 41st International Conference on Logic Programming, ICLP 2025, Rende, Italy.
Abstract: (no abstract available)
Answer: 1

Title: Synthesizing Reactive Systems from Hyperproperties.
Abstract: We present an algorithm for synthesizing reactive systems from hyperproperty specifications. We focus on a fragment of HyperLTL and provide a synthesis algorithm based on bounded synthesis.
Answer: 2

Title: DatalogMTL over the Integer Timeline.
Abstract: We study DatalogMTL, an extension of Datalog with metric temporal operators. Our main contribution is a tight characterisation of the data complexity of reasoning in the integer timeline setting.
Answer: 3

Title: An Answer Set Programming Approach to Argumentative Reasoning in the ASPIC+ Framework.
Abstract: We present an ASP encoding of argumentative reasoning in ASPIC+. The encoding is correct and complete, and we evaluate it on benchmark instances.
Answer: 4

Title: Formally Explaining Decision Tree Models with Answer Set Programming.
Abstract: We propose a formal framework for explaining decision tree predictions using Answer Set Programming. We provide both abductive and contrastive explanations and demonstrate the approach on standard benchmarks.
Answer: 5

Output ONLY a single digit 1, 2, 3, 4, or 5. Do not add any explanation, prefix, or suffix."""


SYSTEM_V2 = """You are a research assistant specialized in classifying scientific papers by their relevance to Answer Set Programming (ASP) and the broader AI-symbolic agenda (neuro-symbolic AI, neural-network verification, explainable AI built on logic).

Score each paper on an integer scale 1-5:
- 1 = NOT related to ASP. Pure verification, type theory, generic logic without ASP. Proceedings volumes, workshop summaries, tutorials, short or extended-abstract entries.
- 2 = Loose adjacency. Verification or knowledge representation with no ASP component. Logic-adjacent but core method is unrelated to Answer Set Programming.
- 3 = Generic logic / declarative reasoning that overlaps ASP vocabulary (Prolog, Datalog, NMR, abductive reasoning, qualitative reasoning, argumentation) but does not center on ASP.
- 4 = Applies or extends ASP. Uses ASP as a tool for a domain problem (planning, scheduling, configuration, etc.), extends ASP with new constructs (probabilities, choice, preferences, learning), or builds tools / encodings on top of an ASP solver.
- 5 = Core ASP advances OR ASP times AI cross-overs. Solver / grounder / formal-semantics work directly on ASP. ASP times neural networks, ASP times LLMs, neuro-symbolic systems built on ASP. Inside CAV / LICS venues, also covers neural-network verification and ML-meets-formal-methods work.

Reasoning chain to follow internally before answering:
1. Scan the title for explicit ASP cues: "asp", "answer set", "clingo", "dlv", "stable model", "aspq", "grounder", "ilasp".
2. Scan title + abstract for AI-symbolic cues: "neural", "neuro-symbolic", "deep learning", "llm", "transformer", "explainability".
3. Scan for low-signal cues: "proceedings", "workshop summary", "(short paper)", "(extended abstract)".
4. Pick the highest-matching label using this priority: explicit ASP cue > AI-symbolic cue > logic-adjacent > generic.

Examples below illustrate the rubric. The "Reasoning" line is for your internal calibration only - do not output it; output only the digit.

Example 1
Title: Branching Bisimulation Learning.
Abstract: We present a learning-based algorithm for inferring branching bisimulation equivalences over labelled transition systems. Our approach handles both finite and parametric systems through a modular tower construction.
Reasoning: Verification / formal-methods topic. No ASP, no neural networks, no neuro-symbolic. Generic verification.
Answer: 1

Example 2
Title: Synthesizing Reactive Systems from Hyperproperties.
Abstract: We present an algorithm for synthesizing reactive systems from hyperproperty specifications. We focus on a fragment of HyperLTL and provide a bounded-synthesis algorithm.
Reasoning: Synthesis is verification with an applied flavour. No ASP. Closer to applied verification than pure type theory.
Answer: 2

Example 3
Title: A Principle-based Analysis of Abstract Agent Argumentation Semantics.
Abstract: We extend Dung's argumentation theory with agents and study four types of semantics including agent-defense and social-agent semantics.
Reasoning: Argumentation theory, KR-adjacent. Uses logic-programming-style semantics but no Answer Set Programming or solver work.
Answer: 3

Example 4
Title: ASP and PDDL+ Applications in Urban Traffic Distribution and Control.
Abstract: Answer Set Programming (ASP) and the mixed discrete-continuous variant PDDL+ are well-known knowledge representation methodologies. We focus on recent problems modeled with ASP and PDDL+ in the context of urban traffic management.
Reasoning: Title says ASP. Abstract uses ASP for an urban-traffic application. Applied ASP problem-solving.
Answer: 4

Example 5
Title: Neural-Probabilistic Answer Set Programming.
Abstract: We combine the robustness of neural networks with the expressivity of symbolic methods, in a deep probabilistic logical programming framework that carries out probabilistic logical programming via the probability estimations of deep neural networks.
Reasoning: Title combines Neural and Answer Set Programming explicitly. Pure neuro-symbolic ASP work.
Answer: 5

Example 6
Title: Leveraging Large Language Models to Generate Answer Set Programs.
Abstract: Large language models have demonstrated strong performance in natural language processing but their reasoning capabilities are limited. We use LLMs to translate natural-language descriptions into Answer Set Programs that can then be solved by a logic engine.
Reasoning: Title has both LLM and Answer Set Programs. ASP times LLM cross-over.
Answer: 5

Example 7
Title: Extending Answer Set Programs with Neural Networks.
Abstract: We extend answer set programs with neural network components, integrating low-level perception with high-level reasoning. The framework supports end-to-end training with neural classifiers grounded inside the ASP solver.
Reasoning: Title says "Extending Answer Set Programs with Neural Networks". Core ASP + neural cross-over.
Answer: 5

Output ONLY a single digit 1, 2, 3, 4, or 5. Do not add the reasoning, any prefix, or any suffix."""


SYSTEM = SYSTEM_V1   # default; overridden by --prompt-version


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
    extra_body = {"thinking": {"type": "disabled"}} if model.startswith("deepseek-v4") else None

    print("[probe] trying with logprobs=True ...")
    try:
        kwargs = dict(model=model, messages=msgs, max_tokens=16, temperature=0.0,
                      logprobs=True, top_logprobs=10)
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        content = get_response_text(resp.choices[0].message)
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
        kwargs = dict(model=model, messages=msgs, max_tokens=16, temperature=0.0)
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        content = get_response_text(resp.choices[0].message)
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


def get_response_text(message):
    """Pull visible text from a DeepSeek chat message.

    DeepSeek-v4 thinking mode returns the visible answer in `content`, but the
    SDK may also expose `reasoning_content`. We concatenate both as a last
    resort so that even thinking-mode replies surface a digit.
    """
    parts = []
    content = getattr(message, "content", None) or ""
    if content:
        parts.append(str(content))
    rc = getattr(message, "reasoning_content", None) or ""
    if rc:
        parts.append(str(rc))
    return " ".join(parts).strip()


async def call(client, mode, model, msgs):
    kwargs = dict(model=model, messages=msgs, max_tokens=16, temperature=0.0)
    if mode == "logprobs":
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 10
    # deepseek-v4-flash defaults to thinking. Disable so the visible digit is
    # emitted within max_tokens.
    if model.startswith("deepseek-v4"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return await client.chat.completions.create(**kwargs)


async def score_one(client, mode, model, row, sem, retries=5):
    msgs = messages(row["title"], row["abstract"])
    last_err = ""
    for attempt in range(retries):
        try:
            async with sem:
                resp = await call(client, mode, model, msgs)
            text_out = get_response_text(resp.choices[0].message)
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
    p.add_argument("--model", default="deepseek-chat",
                   help="default deepseek-chat (non-thinking, compat alias). "
                        "Alternatives: deepseek-v4-flash (uses extra_body=thinking:disabled), "
                        "deepseek-v4-pro, deepseek-reasoner.")
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--flush-every", type=int, default=100)
    p.add_argument("--mode", default="auto", choices=["auto", "logprobs", "text"])
    p.add_argument("--finalize", action="store_true",
                   help="After scoring, also write oof_scores.csv, public_scores.csv, "
                        "private_scores.csv, metrics.json and llm_zeroshot_submission.csv "
                        "into outputs/llm_zeroshot/ (or outputs/llm_zeroshot_v2/). "
                        "Defaults off so you can inspect the cache first.")
    p.add_argument("--finalize-only", action="store_true",
                   help="Skip API calls; just rebuild the artefacts from the existing cache.")
    p.add_argument("--prompt-version", default="v1", choices=["v1", "v2"],
                   help="v1 = original 5-example prompt; v2 = 7-example prompt with internal "
                        "reasoning lines that combat the under-rating bias on labels 4-5. "
                        "v2 writes its cache and artefacts under outputs/llm_zeroshot_v2/.")
    return p.parse_args()


def main():
    args = parse_args()

    # Redirect cache + run dir if using a different prompt version
    global SYSTEM, RUN_DIR, CACHE_PATH
    if args.prompt_version == "v2":
        SYSTEM = SYSTEM_V2
        RUN_DIR = ROOT / "outputs" / "llm_zeroshot_v2"
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_PATH = RUN_DIR / "llm_raw_cache.csv"
    else:
        SYSTEM = SYSTEM_V1

    print(f"[main] prompt_version = {args.prompt_version}, RUN_DIR = {RUN_DIR}")

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

    if args.finalize_only:
        if not CACHE_PATH.exists():
            raise SystemExit(f"no cache at {CACHE_PATH}")
        cache = pd.read_csv(CACHE_PATH)
        finalize(cache, train_full, public_full, private_full)
        return

    if not args.api_key:
        raise SystemExit("Set --api-key or DEEPSEEK_API_KEY env var.")

    if args.mode == "auto":
        mode = probe(args.api_key, args.model)
    else:
        mode = args.mode
    print(f"[main] using mode = {mode}")

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

    if args.finalize:
        print("\n=== Finalizing artefacts ===")
        finalize(cache, train_full, public_full, private_full)


def finalize(cache: pd.DataFrame, train_full: pd.DataFrame,
             public_full: pd.DataFrame, private_full: pd.DataFrame) -> None:
    """Build oof_scores.csv, public_scores.csv, private_scores.csv,
    metrics.json, and llm_zeroshot_submission.csv from the (cleaned) cache."""
    from sklearn.metrics import cohen_kappa_score, mean_absolute_error, f1_score
    from scipy.optimize import differential_evolution

    sample = pd.read_csv(DATA / "Test_Submission.csv")
    train_full = train_full.sort_values("id").reset_index(drop=True)
    public_full = public_full.sort_values("id").reset_index(drop=True)
    private_full = private_full.sort_values("id").reset_index(drop=True)

    train_res = cache[cache["source_split"] == "train"].sort_values("id").reset_index(drop=True)
    public_res = cache[cache["source_split"] == "public_test"].sort_values("id").reset_index(drop=True)
    private_res = cache[cache["source_split"] == "private_test"].sort_values("id").reset_index(drop=True)

    assert (train_res["id"].to_numpy() == train_full["id"].to_numpy()).all(), "train id misalignment"
    assert (public_res["id"].to_numpy() == public_full["id"].to_numpy()).all(), "public id misalignment"
    assert (private_res["id"].to_numpy() == private_full["id"].to_numpy()).all(), "private id misalignment"

    y_class = train_full["Label"].astype(int).to_numpy()
    train_scores = train_res["score"].to_numpy()
    public_scores = public_res["score"].to_numpy()
    private_scores = private_res["score"].to_numpy()
    train_probs = train_res[["p1", "p2", "p3", "p4", "p5"]].to_numpy()
    public_probs = public_res[["p1", "p2", "p3", "p4", "p5"]].to_numpy()
    private_probs = private_res[["p1", "p2", "p3", "p4", "p5"]].to_numpy()

    print("Mean LLM score per true label (train):")
    print(pd.DataFrame({"true": y_class, "score": train_scores}).groupby("true")["score"]
          .agg(["mean", "std", "count"]).round(3).to_string())

    train_dist = pd.Series(y_class).value_counts(normalize=True).reindex(
        [1, 2, 3, 4, 5], fill_value=0).to_numpy()

    def s2l(s, t):
        return np.digitize(s, np.sort(np.asarray(t, dtype=float))) + 1

    def pred_dist(labels):
        return pd.Series(labels).value_counts(normalize=True).reindex(
            [1, 2, 3, 4, 5], fill_value=0).to_numpy()

    def tune(y, scores, lambd=0.5, seed=42):
        def obj(raw):
            thr = np.sort(raw)
            gap = np.min(np.diff(thr))
            gap_pen = 0.0 if gap >= 0.03 else (0.03 - gap) * 5.0
            labels = s2l(scores, thr)
            qwk = cohen_kappa_score(y, labels, weights="quadratic")
            dp = float(np.sum(np.abs(pred_dist(labels) - train_dist)))
            return -qwk + gap_pen + lambd * dp
        bounds = [(1.4, 2.5), (1.8, 2.9), (2.2, 3.4), (2.6, 4.2)]
        res = differential_evolution(obj, bounds, seed=seed, maxiter=120, popsize=15,
                                     polish=True, updating="immediate", workers=1)
        thr = np.sort(res.x)
        return thr, cohen_kappa_score(y, s2l(scores, thr), weights="quadratic")

    thresholds, oof_qwk = tune(y_class, train_scores)
    oof_pred = s2l(train_scores, thresholds)
    public_pred = s2l(public_scores, thresholds)
    private_pred = s2l(private_scores, thresholds)
    print(f"\nConstrained-tuned OOF QWK = {oof_qwk:.4f}")
    print(f"thresholds = {thresholds.tolist()}")

    metrics = {
        "method": "llm_zeroshot_deepseek",
        "oof_qwk": float(oof_qwk),
        "oof_mae": float(mean_absolute_error(y_class, oof_pred)),
        "oof_macro_f1": float(f1_score(y_class, oof_pred, average="macro")),
        "thresholds": [float(v) for v in thresholds],
        "label_distribution_combined": {int(k): int(v) for k, v in pd.Series(
            np.concatenate([public_pred, private_pred])).value_counts().sort_index().items()},
        "label_distribution_public": {int(k): int(v) for k, v in pd.Series(public_pred).value_counts().sort_index().items()},
        "label_distribution_private": {int(k): int(v) for k, v in pd.Series(private_pred).value_counts().sort_index().items()},
    }
    (RUN_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n", json.dumps(metrics, indent=2))

    pd.DataFrame({"id": train_full["id"], "Label": y_class,
                  "oof_score": train_scores, "oof_pred": oof_pred,
                  **{f"p_label_{i+1}": train_probs[:, i] for i in range(5)}}).to_csv(
        RUN_DIR / "oof_scores.csv", index=False)
    pd.DataFrame({"id": public_full["id"], "score": public_scores, "pred": public_pred,
                  **{f"p_label_{i+1}": public_probs[:, i] for i in range(5)}}).to_csv(
        RUN_DIR / "public_scores.csv", index=False)
    pd.DataFrame({"id": private_full["id"], "score": private_scores, "pred": private_pred,
                  **{f"p_label_{i+1}": private_probs[:, i] for i in range(5)}}).to_csv(
        RUN_DIR / "private_scores.csv", index=False)

    combo = pd.concat([
        pd.DataFrame({"id": public_full["id"], "Label": public_pred}),
        pd.DataFrame({"id": private_full["id"], "Label": private_pred}),
    ], ignore_index=True)
    submission = sample[["id"]].merge(combo, on="id", how="left")
    submission["Label"] = submission["Label"].astype(int)
    submission.to_csv(RUN_DIR / "llm_zeroshot_submission.csv", index=False)
    print(f"\nWrote artefacts under {RUN_DIR}")


if __name__ == "__main__":
    main()
