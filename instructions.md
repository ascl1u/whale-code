# Take-Home Challenge: Build an Agentic Harness that Outperforms Claude Code on Terminal Bench

## The Challenge

Build an **agentic harness** that outperforms **Claude Code** on **Terminal Bench**.

Your deliverable is a reproducible system that can be evaluated on the benchmark and demonstrates stronger performance than Claude Code under the same general conditions.

The benchmark to target is:

[Terminal Bench 2.0 Leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0)

## Objective

Design and implement a harness that can:

- run reliably in a terminal-driven environment
- solve Terminal Bench tasks with strong success rates
- produce measurable, reproducible results
- be straightforward for reviewers to run and verify

The emphasis is not only on benchmark performance, but also on engineering quality, reproducibility, and clarity of evaluation. Use of AI tools is allowed and encouraged, we are looking for how you think about the problem at hand and the care shown in your work.  

## Deliverable

Submit a **GitHub repository** containing the full implementation of your harness, along with everything needed for a reviewer to reproduce your results and confirm the benchmark outcome.

Your repository should be organized so that another engineer can clone it, configure it, run it, and verify the reported numbers without needing additional private context.

## Submission Requirements

Your GitHub repository should include the following:

### 1. Clear setup instructions

Document how to install dependencies, configure the environment, and prepare the system to run the harness.

### 2. Reproduction steps

Provide exact commands for:

- running the harness
- executing the benchmark workflow
- reproducing the reported evaluation results
- verifying that the results match what you claim

### 3. Evaluation notes

Document:

- which models were used
- how the harness is configured
- any assumptions or constraints
- any benchmark-specific adjustments or caveats

### 4. Results summary

Include a concise summary of benchmark performance and how the results compare against Claude Code.

### 5. Limitations and tradeoffs

Be explicit about what is incomplete, fragile, or still experimental. Honest documentation is preferred over polished but unclear claims.

## What We Will Evaluate

We will primarily evaluate:

- benchmark performance on Terminal Bench
- reproducibility of the reported results
- quality of the harness design
- clarity of the repository and instructions
- engineering judgment in how you structure, document, and validate the system

## Provided Resources

An **OpenRouter API key** with **$200 of credits** is available for this project.

I have this key, will personally set it up.

You may use this budget to explore model choices, prompting strategies, orchestration approaches, and evaluation runs needed to build the strongest harness you can within the scope of the take-home.

Do not commit credentials or secrets into the repository.

## Expectations

Please make the submission professional and reviewer-friendly:

- keep instructions concrete and easy to follow
- make result verification straightforward
- avoid ambiguous setup or hidden steps
- prefer reproducible scripts over manual explanation where possible

If there are important assumptions, call them out clearly in the repository.

## Final Note

The goal of this exercise is to assess both raw benchmark performance and your ability to build a rigorous, usable evaluation harness. A strong submission is one that performs well, is easy to reproduce, and is documented with clarity and honesty.

