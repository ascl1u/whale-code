# Whale: An Agentic Harness for Terminal-Bench 2.0

A custom [Harbor](https://harborframework.com/) agent built on [Terminus 2](https://harborframework.com/docs/terminus-2) that targets strong performance on [Terminal-Bench 2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0), evaluated on a representative 18-task subset and compared against Claude Code.

## Approach

| Area | Base Terminus 2 | Whale |
|------|----------------|-------|
| **Tool interface** | ICL JSON/XML parsing | Native tool calling; `read_file` for text files; `image_read` for screenshots and image files |
| **Multimodal** | Text-only | `image_read` for vision |
| **Command execution** | Fixed wait times | Exact keystrokes preserved; safe inline completion polling for simple submitted commands, timed non-blocking sends for interactive or multiline input |
| **Completion** | Model self-reports done | Double-confirmation checklist |
| **Tool results in chat** | N/A / varies | One `tool` message per call to satisfy API sequencing; observations are fed back in the next turn, and the current implementation uses minimal `tool` stubs |
| **Context management** | Basic summarization | Terminus 2 summarization + `_limit_output_length` only (no extra truncation layer) |
| **Prompt caching** | None | Anthropic ephemeral caching |

Additional improvements: best-effort verifier script hint from the environment, short continuation user turns after tool results, and error recovery aligned with Terminus 2.

## Evaluation Design

- **Model:** Claude Sonnet 4.5 via OpenRouter — $3/M in, $15/M out
- **Subset:** 18 tasks (4 easy / 8 medium / 6 hard), frozen in [`eval/subset.txt`](eval/subset.txt) before development. Full TB 2.0 (89 tasks) exceeds the $200 budget for meaningful iteration.
- **Runs:** k=5 per task, matching the leaderboard standard
- **Baseline:** Claude Code + Sonnet 4.5 per-task results extracted from the [TB 2.0 leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0) submission detail view. Full-bench reference: ~42.7% on 89 tasks.

## Setup

**Prerequisites:** Python 3.12+, [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/ascl1u/whale-code.git
cd whale-code
uv sync
export OPENROUTER_API_KEY="sk-or-v1-..."
```

## Running

Harbor selects tasks with **`-i`** (short for `--include-task-name`). Pass **`-i` once per task name**. The 18 names below match [`eval/subset.txt`](eval/subset.txt).

**Smoke test (one task, one attempt):**

```bash
uv run harbor run -d terminal-bench@2.0 --agent-import-path agent.whale_agent:WhaleAgent -m openrouter/anthropic/claude-sonnet-4.5 -k 1 -i fix-git
```

**Full 18-task evaluation, Whale (k=5):**

```bash
uv run harbor run \
  -d terminal-bench@2.0 \
  --agent-import-path agent.whale_agent:WhaleAgent \
  -m openrouter/anthropic/claude-sonnet-4.5 \
  -k 5 \
  -i cobol-modernization \
  -i fix-git \
  -i overfull-hbox \
  -i prove-plus-comm \
  -i build-cython-ext \
  -i caffe-cifar-10 \
  -i crack-7z-hash \
  -i constraints-scheduling \
  -i hf-model-inference \
  -i qemu-startup \
  -i polyglot-c-py \
  -i multi-source-data-merger \
  -i configure-git-webserver \
  -i feal-differential-cryptanalysis \
  -i cancel-async-tasks \
  -i train-fasttext \
  -i sam-cell-seg \
  -i dna-assembly
```

**Stock Terminus 2 on the same 18 tasks:** run the same command as above, but remove `--agent-import-path agent.whale_agent:WhaleAgent` and add `--agent terminus-2`.

## Results

> Results will be populated after evaluation runs.

| Agent | Model | Subset Accuracy (18 tasks) | Full-Bench Reference |
|-------|-------|---------------------------|---------------------|
| Claude Code (baseline) | Claude Sonnet 4.5 | ~51.1% | ~40.1% |
| Terminus 2 (stock) | Claude Sonnet 4.5 | ~48.9% | ~42.8% |
| **Whale** | **Claude Sonnet 4.5** | **TBD** | — |

Raw logs and per-task results stored in `results/`.

## Assumptions

1. **Budget shapes scope.** $200 is insufficient for full 89-task evaluation with iteration. The 18-task subset trades coverage for iteration budget.
2. **Leaderboard data is ground truth for the baseline.** Per-task results extracted directly; no separate Claude Code run.
3. **k=5 matches leaderboard methodology.** Apples-to-apples comparison.
4. **Same model for harness and baseline.** Isolates harness quality, not model capability.
5. **Subset is representative.** Stratified across difficulty levels and categories.

## Limitations and Tradeoffs

- **Subset, not full benchmark.** 18/89 tasks. Not directly comparable to full-bench leaderboard entries.
- **Single model.** Only Claude Sonnet 4.5. May perform differently with other models.
- **Limited iteration.** Budget allows ~1 dev eval + 1 final eval at k=5.
- **No controlled Claude Code run.** Baseline from leaderboard, not a run we executed.
- **Builds on Terminus 2.** Incremental improvements on a tested base. Upstream vs. our work is clearly attributed.
- **No per-task prompt tuning.** Single system prompt and tool set across all tasks. Subset was frozen before development.

## Attribution

- [Harbor Framework](https://harborframework.com) — official Terminal-Bench harness
- [Terminus 2](https://harborframework.com/docs/terminus-2) — base agent by AfterQuery
- [Terminus-KIRA](https://github.com/krafton-ai/KIRA) — architectural inspiration by Krafton AI
- [Meta-Harness](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact) — environment bootstrapping ideas by Stanford IRIS Lab
