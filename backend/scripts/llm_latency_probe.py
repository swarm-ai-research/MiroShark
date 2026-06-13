#!/usr/bin/env python3
"""Latency + cost probe for batched LLM tick calls.

Closes bd-0t5. The first LLM smoke (PR #1) ran at ~7s/tick with
Gemini 2.5-flash via the OpenAI-compatible endpoint. Frontend auto-poll
is 1.5s — way faster than the world advances. For continuous-play
demos we need ~1s/tick.

This probe sends the exact same batched-LLM payload (3 personas, 1
observation each, "respond with JSON only") to each configured
provider/model and records wall-clock + token counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx


_BACKEND = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _BACKEND / "runs/llm_latency"

logger = logging.getLogger("llm_latency_probe")


# Mimics the phase-3 batched system prompt.
_SYSTEM = (
    "You are the decision layer for 3 workers in a gather-trade-build "
    "economy. For EACH worker, choose ONE action this tick. RESPONSE "
    "FORMAT: reply with a single raw JSON object mapping agent_id to "
    "action object. No markdown. No commentary. Start with { end with }. "
    "Action schema: {\"action_type\": one of move/gather/build/trade_buy/"
    "trade_sell/shift_income/misreport/noop, \"direction\": up/down/left/right}."
)

# Realistic observation payload, sized like the actual phase-3 payload.
_USER = (
    "Decide actions for these workers. Return one entry per worker.\n\n"
    + json.dumps({
        "open_markets": [
            {"market_id": "gtb-0000", "question": "Will welfare exceed 3.5 by epoch 5?",
             "metric": "welfare", "op": ">", "threshold": 3.5, "deadline_epoch": 5},
            {"market_id": "gtb-0001", "question": "Will Gini fall below 0.3 by epoch 5?",
             "metric": "gini_coefficient", "op": "<", "threshold": 0.3, "deadline_epoch": 5},
        ],
        "workers": {
            "worker_0": {"persona": {"name": "Ada", "personality": "aggressive builder"},
                         "obs": {"position": [3, 4], "inventory": {"wood": 2, "stone": 1, "coin": 9.5},
                                 "energy": 100, "epoch": 1, "step": 0, "frozen": False}},
            "worker_1": {"persona": {"name": "Bo", "personality": "cautious trader"},
                         "obs": {"position": [7, 8], "inventory": {"wood": 0, "stone": 0, "coin": 10},
                                 "energy": 100, "epoch": 1, "step": 0, "frozen": False}},
            "worker_2": {"persona": {"name": "Cy", "personality": "speculator"},
                         "obs": {"position": [1, 2], "inventory": {"wood": 1, "stone": 0, "coin": 10},
                                 "energy": 100, "epoch": 1, "step": 0, "frozen": False}},
        },
    }, indent=2)
)


_PROVIDERS = [
    # Gemini family via the OpenAI-compatible endpoint.
    {"name": "gemini-2.5-flash",
     "base": "https://generativelanguage.googleapis.com/v1beta/openai",
     "model": "gemini-2.5-flash", "key_env": "GEMINI_API_KEY"},
    {"name": "gemini-2.5-flash-lite",
     "base": "https://generativelanguage.googleapis.com/v1beta/openai",
     "model": "gemini-2.5-flash-lite", "key_env": "GEMINI_API_KEY"},
    {"name": "gemini-1.5-flash-8b",
     "base": "https://generativelanguage.googleapis.com/v1beta/openai",
     "model": "gemini-1.5-flash-8b", "key_env": "GEMINI_API_KEY"},
    # Groq — known fast inference.
    {"name": "groq/llama-3.3-70b",
     "base": "https://api.groq.com/openai/v1",
     "model": "llama-3.3-70b-versatile", "key_env": "GROQ_API_KEY"},
    {"name": "groq/llama-3.1-8b",
     "base": "https://api.groq.com/openai/v1",
     "model": "llama-3.1-8b-instant", "key_env": "GROQ_API_KEY"},
]


def _probe(provider: Dict[str, str], rounds: int) -> Optional[Dict[str, object]]:
    key = os.environ.get(provider["key_env"], "")
    if not key:
        logger.warning("no key for %s (env=%s); skipping", provider["name"], provider["key_env"])
        return None
    latencies: List[float] = []
    tokens_in: List[int] = []
    tokens_out: List[int] = []
    parse_success: List[bool] = []
    for r in range(rounds):
        body = {
            "model": provider["model"],
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER},
            ],
            "temperature": 0.4,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        t0 = time.time()
        try:
            resp = httpx.post(
                f"{provider['base']}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body, timeout=60,
            )
            elapsed = time.time() - t0
            data = resp.json()
            if "error" in data:
                logger.warning("%s round %d error: %s", provider["name"], r, data["error"])
                parse_success.append(False)
                continue
            content = data["choices"][0]["message"].get("content", "") or ""
            usage = data.get("usage") or {}
            latencies.append(elapsed)
            tokens_in.append(int(usage.get("prompt_tokens", 0) or 0))
            tokens_out.append(int(usage.get("completion_tokens", 0) or 0))
            try:
                # Strict JSON parse, no fence-stripping; if this fails the
                # actual `chat_json` would also fail without the bd-followup
                # extractor.
                json.loads(content.strip())
                parse_success.append(True)
            except json.JSONDecodeError:
                parse_success.append(False)
        except Exception as exc:
            logger.warning("%s round %d exception: %s", provider["name"], r, exc)
            parse_success.append(False)
    if not latencies:
        return None
    return {
        "provider": provider["name"],
        "rounds_attempted": rounds,
        "rounds_succeeded": len(latencies),
        "rounds_parsed": sum(parse_success),
        "latency_mean_s": statistics.mean(latencies),
        "latency_median_s": statistics.median(latencies),
        "latency_p10_s": min(latencies),
        "latency_p90_s": max(latencies),
        "tokens_in_mean": statistics.mean(tokens_in) if tokens_in else 0,
        "tokens_out_mean": statistics.mean(tokens_out) if tokens_out else 0,
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3, help="Probe calls per provider")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    results: List[dict] = []
    for prov in _PROVIDERS:
        logger.info("probing %s (%d rounds)...", prov["name"], args.rounds)
        r = _probe(prov, args.rounds)
        if r is not None:
            results.append(r)
            logger.info(
                "  -> mean %.2fs (n=%d/%d succeeded, %d parsed)",
                r["latency_mean_s"], r["rounds_succeeded"], r["rounds_attempted"],
                r["rounds_parsed"],
            )
    _write_csv(args.output / "results.csv", results)


if __name__ == "__main__":
    main()
