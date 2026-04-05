# Whale: A Small Harbor Agent for Terminal-Bench 2.0

Whale is a small [Harbor](https://harborframework.com/) agent built by subclassing [Terminus 2](https://harborframework.com/docs/terminus-2), not by replacing Harbor's environment stack. The goal was a narrow, reviewable diff against Terminus 2: switch to native tool calling, preserve exact terminal keystrokes, add optional vision for image files, and request Anthropic prompt caching.

Development was driven by a frozen 18-task subset in [`eval/subset.txt`](eval/subset.txt). I also used Terminus-KIRA as design inspiration, but Whale is not a KIRA port and does not reproduce KIRA's full orchestration stack.

## Benchmark Snapshot

| Agent | Subset Accuracy | Full-Benchmark Accuracy |
|-------|-----------------|-------------------------|
| Claude Code | 51.1% | 40.1% |
| Terminus 2 | 48.9% | 42.8% |
| Whale | 52.2% | - |

## Design Scope

- Whale keeps Harbor's Docker-backed task environments, tmux session model, trajectory format, and built-in summarization machinery.
- The model-facing tool surface is intentionally small: `execute_commands`, `task_complete`, and `image_read`.
- Text files are still inspected through shell commands like `cat`, `sed`, and `head`; there is no dedicated `read_file` tool.
- The implementation is intentionally compact and lives almost entirely in `agent/`.

## What Changed vs Stock Terminus 2

| Area | Stock Terminus 2 | Whale |
|------|------------------|-------|
| Tool interface | ICL-style structured responses | Native tool calling with `execute_commands`, `task_complete`, and `image_read` |
| Terminal execution | Fixed waits after keystrokes | Verbatim tmux keystrokes, inline completion polling only for simple one-line submitted commands, timed non-blocking sends for interactive or multiline input |
| Multimodal support | Text-only | `image_read` for screenshots and image files |
| Completion behavior | Model decides it is done | Two-step model-mediated completion with a confirmation turn |
| Tool messages in chat | Depends on base implementation | Minimal `tool` stubs for API sequencing; actual terminal observations are injected on the next user turn |
| Context management | Terminus 2 defaults | Terminus 2 summarization plus output limiting; no extra planner or memory layer |
| Prompt caching | None | Anthropic-style ephemeral `cache_control` hints on recent messages |

## Code Layout

- [`agent/whale_agent.py`](agent/whale_agent.py): composition layer over Terminus 2
- [`agent/tools.py`](agent/tools.py): model-facing tool schema
- [`agent/context.py`](agent/context.py): prompt-caching hint injection
- [`agent/whale/terminal.py`](agent/whale/terminal.py): terminal send/poll behavior
- [`agent/whale/llm.py`](agent/whale/llm.py): LiteLLM calls, image reading, and usage accounting
- [`agent/whale/loop.py`](agent/whale/loop.py): main agent loop and trajectory recording
- [`agent/whale/parsing.py`](agent/whale/parsing.py): native tool-call parsing
- [`agent/prompt_template.txt`](agent/prompt_template.txt): base system prompt

## Evaluation Design

- Development iteration used the frozen 18-task subset in [`eval/subset.txt`](eval/subset.txt).
- `k=5` was used for task-level comparisons, matching standard Terminal-Bench reporting practice.
- I compared Whale against stock Terminus 2 and a public Claude Code + Sonnet 4.5 leaderboard entry.

## Setup

**Prerequisites**

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker Engine / Docker Desktop with `docker compose`
- An OpenRouter API key

**Important Docker note**

Terminal-Bench 2.0 is heavily Docker-dependent. Before running any eval, make sure `docker info` and `docker compose version` work from the same shell that will run Harbor. On WSL, Docker Desktop WSL integration must be enabled for that distro. Several tasks pull large images, so disk space, registry throughput, and pull reliability matter.

```bash
git clone https://github.com/ascl1u/whale-code.git
cd whale-code
uv sync
export OPENROUTER_API_KEY="sk-or-v1-..."
```

## Running

Harbor selects tasks with `-i` (`--include-task-name`). Pass `-i` once per task name.

**Smoke test**

```bash
uv run harbor run \
  -d terminal-bench@2.0 \
  --agent-import-path agent.whale_agent:WhaleAgent \
  -m openrouter/anthropic/claude-sonnet-4.5 \
  -k 1 \
  -i fix-git
```

**18-task development subset, Whale (`k=5`)**

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

**Stock Terminus 2 on the same subset**

Run the same command as above, but remove `--agent-import-path agent.whale_agent:WhaleAgent` and add `--agent terminus-2`.

## Interpreting Artifacts

- Harbor's `n_errors` is not a pure "agent logic failed" counter. It can include agent timeouts, environment setup failures, verifier timeouts, and other infrastructure errors.
- A trial can earn reward `1.0` and still appear in `exception_stats` if the task was effectively solved but the run timed out during cleanup or post-completion handling.

## Limitations and Tradeoffs

- **This is an incremental Harbor/Terminus 2 artifact, not a new end-to-end framework.** The novelty is in a small set of agent-policy changes, not in replacing Harbor's environment or verifier stack.
- **Development was subset-driven.** The frozen 18-task subset was chosen for iteration cost and speed, not because it is a substitute for the full benchmark.
- **Docker infrastructure is a major source of variance.** In practice, some failures come from Docker image pulls, Docker Compose startup, registry/network instability, WSL integration, or disk pressure before the agent has a meaningful chance to act.
- **Large environment images matter.** Some Terminal-Bench tasks pull multi-GB images. Partial downloads, short reads, or slow pulls can dominate wall-clock time and can surface as Harbor `RuntimeError`, setup failure, or timeout artifacts.
- **Verifier and environment failures sit next to agent failures in the same Harbor summaries.** Review `exception_type` before attributing a bad trial to the policy itself.
- **Completion is intentionally conservative.** Whale still uses a two-step completion flow, which can spend an extra model turn on final confirmation.
- **Tool messages are minimal.** The model does not receive rich tool-role payloads containing the full terminal output; observations are fed back on the next user turn instead.
- **Prompt caching metrics are provider-dependent.** Whale requests Anthropic-style caching hints, but observed cached-token counts can still show up as zero depending on which LiteLLM/OpenRouter usage fields are populated.
- **Baseline comparison is not a perfectly controlled rerun.** Comparison numbers come from the public leaderboard rather than from a local rerun in this repository.

## Attribution

- [Harbor Framework](https://harborframework.com) for the official Terminal-Bench harness
- [Terminus 2](https://harborframework.com/docs/terminus-2) for the base Harbor agent
- [Terminus-KIRA](https://github.com/krafton-ai/KIRA) for design inspiration around tool calling and caching
- [Meta-Harness](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact) for environment bootstrapping ideas
