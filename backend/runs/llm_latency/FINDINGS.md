# Faster LLM model for continuous-play GTB demos

*Closes bd-0t5 (`Faster LLM model for live-demo continuous-play mode`).*

## Setup

Send the same batched-LLM prompt (3 personas, 2 open markets, JSON
response_format) to each available provider/model and measure
wall-clock + parse success rate.

The prompt is sized to match the actual phase-3 `BatchLLMDriver`
payload at 3-worker scale: 678 input tokens. Each provider was probed
5 times.

Probed providers (those that responded; OpenRouter account is dead per
PR #1 history; Groq key is expired; gemini-1.5-flash-8b 404s on the
OpenAI-compat endpoint):

## Results

| provider | mean latency | median | range | parse OK | tokens out |
|---|---:|---:|---:|---:|---:|
| **gemini-2.5-flash-lite** | **0.59s** | 0.59s | 0.46–0.72s | **5/5** | 94 |
| gemini-2.5-flash (baseline) | 8.81s | 9.02s | 6.13–12.11s | 5/5 | 79 |

## What the data says

1. **gemini-2.5-flash-lite is 15× faster than the current default at
   identical parse quality.** Median 0.59s vs 9.02s, range 0.46–0.72s
   (tight) vs 6.13–12.11s (wide). The lite variant doesn't burn
   thinking tokens before producing output.

2. **Parse success is 5/5 in both cases** at the GTB workload size.
   The post-smoke fixes from PR #1 (strict prompt + greedy `{…}`
   extractor + 4096 max_tokens) gave both models enough headroom to
   produce clean JSON. The lite model doesn't regress parse quality
   despite being smaller.

3. **At 0.59s/tick, the frontend's 1.5s auto-poll cadence finally
   makes sense.** With flash at 9s/tick, the frontend polled 6×
   between server ticks; with flash-lite, the server ticks ~3×
   between polls. Continuous-play UX is now viable.

4. **The current 4096 max_tokens default has plenty of headroom.**
   With completion=94 tokens (lite) and 79 tokens (flash), even the
   3-worker payload uses <3% of the budget. The PR #1 fix that
   bumped this from 2048 was the right call but doesn't need to grow
   further.

## Recommendation

Switch the default `LLM_MODEL_NAME` in `.env.example` from
`gemini-2.5-flash` (and `xiaomi/mimo-v2-flash` in the upstream
default) to **`gemini-2.5-flash-lite`** for any provider that
supports it. The latency win is huge and the parse quality is
unchanged at the GTB workload size.

For users who want a different model: the new
`backend/scripts/llm_latency_probe.py` runs the same probe against
any provider in 3-5 calls and writes a comparable results.csv.

## What's still unverified

- **No accuracy comparison.** Both models parse JSON but a smaller
  model may emit lower-quality persona-aligned action sequences. The
  proper test is bd-mit's Brier score against resolved markets,
  re-run with flash-lite to see if predictive accuracy holds.
- **Groq and OpenRouter** would likely be faster still; both have
  dead keys in the current environment. Re-probe when keys are
  refreshed.
- **No cost comparison.** Gemini 2.5 Flash and Flash-Lite have
  different per-token pricing; the 15× speed win may or may not also
  be a cost win depending on volume.

## Sibling questions worth filing

- **Re-run bd-mit (market predictiveness Brier score) with flash-lite**
  to confirm the speed win doesn't cost forecast quality.
- **Probe Anthropic claude-haiku-4-5 via the Anthropic API** when an
  ANTHROPIC_API_KEY is available.

## How to reproduce

```bash
cd backend
uv run python -m scripts.llm_latency_probe --rounds 5
# Results land in runs/llm_latency/results.csv
```

To swap the default model into the running backend:

```bash
sed -i.bak 's/^LLM_MODEL_NAME=.*$/LLM_MODEL_NAME=gemini-2.5-flash-lite/' .env
sed -i.bak 's/^SMART_MODEL_NAME=.*$/SMART_MODEL_NAME=gemini-2.5-flash-lite/' .env
# restart backend
```
