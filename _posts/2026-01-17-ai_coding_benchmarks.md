# What LLM Coding Benchmarks Actually Measure (and What They Don’t)

AI coding assistants (Cursor, Claude Code, Antigravity, etc.) no longer need an introduction. Many of these tools let you choose the underlying LLM, and that choice definitely matters. However, with so many models on the market—and new ones coming out every few weeks—how do you make that choice?

Benchmarks are the obvious place to look. But how should you use them? And if you’ve followed this space even a little, you’ve probably heard plenty of criticism: benchmark gaming, training contamination, misaligned tasks, and so on. So are benchmarks useful at all?

A colleague recently asked me this, which triggered a longer train of thought that I wanted to write down.

## TL;DR

Coding benchmarks are often misread as predictors of developer productivity, but they actually measure how a model performs within a tightly constrained environment. For isolating *model-level coding capability*, [**SWE-bench Bash Only**](https://www.swebench.com) is a reasonable signal. However, models can feel very different in practice—so benchmarks should inform shortlisting, not the final choice.

## Are benchmarks useful at all

Benchmarking LLMs has always been hard, but it has become especially tricky now that there is a strong financial incentive to look good on well-known benchmarks. Once a benchmark becomes popular, model providers will aggressively optimize for it. Sometimes this means genuine improvements; sometimes it means overfitting to the task distribution; and sometimes benchmark data ends up in training pipelines simply because large-scale web scraping makes clean separation difficult. Some benchmarks (like [ARC-AGI](https://github.com/arcprize/ARC-AGI-2?tab=readme-ov-file#dataset-composition)) even explicitly provide training splits.

But the deeper issue is not cheating per se. The bigger problem is that benchmarks test models on a *specific set of tasks in a specific environment*. And for coding benchmarks, that environment matters a lot.

I’ll use my personal favorite [**SWE-bench**](https://www.swebench.com) as a concrete example, but the same arguments apply to most coding benchmarks.

## Why coding benchmarks are especially tricky

SWE-bench benchmark measures completion rate on a set of predefined coding tasks (most commonly [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) by OpenAI). On the **Bash Only** leaderboard, top models currently score around 74% (as of early 2026). It is tempting to read this literally and conclude:

> “This model will solve 74% of my coding tasks.”

That is the wrong mental model.

First, SWE-bench deliberately focuses on *hard* tasks. A benchmark where every frontier model scores 100% would not be very informative, so the task distribution is intentionally skewed toward complex, failure-prone cases. Your day-to-day work almost certainly contains a much larger fraction of simpler tasks.

Second—and more importantly—neither SWE-bench nor real-world coding uses a *raw LLM*. An LLM by itself is just a text-in/text-out model. Real coding work involves an existing codebase, partial context, multiple files, tests, and iteration. To bridge that gap, modern tools wrap models in an agent: file system access, editing tools, test execution, search, and a system prompt that explains how to use all of this.

This immediately introduces several confounding factors that benchmarks struggle to control:

* **Agentic setup:** tools, interfaces, and orchestration logic.
* **Instruction layer:** system prompts and tool descriptions.
* **Human–agent interaction:** latency, verbosity, edit granularity, production readiness, and the overall *conversation dynamic* the agent encourages—all the details that matter less for benchmark scores but matter greatly in daily use.

Standardizing the agent is very tricky and essentially becomes a design choice that strongly affects what the benchmark actually measures.

For example, earlier versions of SWE-bench allowed participants to submit results from their full custom agents. In theory, this sounds great—wouldn’t it be nice to have a leaderboard comparing Cursor, Claude Code, and others? In practice, however, this hurt usability. The leaderboard became polluted with agents of unknown provenance, some with no details, some hiding the underlying LLM, and so on. Conversely, many well-known solutions never appeared on the leaderboard at all.

I am not a developer of any of these tools, so I can only speculate, but my intuition is as follows. While full agent submission allows contenders to optimize performance, the benchmark setup inevitably favors certain design decisions. For example, imagine an agent encountering an underspecified task. In a tool like Cursor, one reasonable response would be to stop and ask the user for clarification. In an automated benchmark like SWE-bench, however, there is no human in the loop, so stopping would result in a failure. This specific case should arguably be treated as a data quality issue and removed from evaluation, but the broader point remains: maximizing SWE-bench performance is not always aligned with improving the user experience.

Couple this with the operational burden of constantly re-running evaluations to keep up with frequent agent updates, and it becomes easy to see why many off-the-shelf coding assistants chose not to participate. This ultimately undermined what was otherwise a promising idea.

You can still see results from the older setup under the **Verified** tab on the SWE-bench website. By default, however, you now see the leaderboard for the newer **Bash Only** setup, which takes the opposite approach. All models interact with the environment through a minimal, standardized interface centered around shell commands. Some operations (like editing files) are not especially ergonomic in Bash, but the interface is flexible, stable, widely known, and well represented in LLM training data.

This effectively defines a minimal agent that is *shared by all models*—each LLM gets a single row showing the percentage of tasks it successfully completes using this agent.

Bash Only likely does *not* represent the best possible performance you could get from a given model. Rich agents—with structured edits, indexing, patch application, and sub-agents—can often perform better by reducing the coordination burden placed on the model. But Bash Only removes a large class of agent-specific confounders, making it a cleaner proxy for *model capability* rather than *tool quality*.

That distinction matters—but it still does not paint a full picture. Two models with similar benchmark scores can feel very different in practice. How fast they are, how they structure solutions, how conservative they are, how well they follow instructions, and how much supervision they require are all critical to the developer experience—and largely invisible in a single benchmark score.

For example, my personal experience using Claude Code and Codex is that both tools are roughly on par when it comes to capabilities. This is also consistent with SWE-bench results, as GPT-5.2 has roughly the same score as Claude 4.5 Sonnet. However, in practice, Codex tends to take longer to perform the same task, and I find it a bit harder to configure (although OpenAI’s team has been making a strong effort to improve it lately).

One more thought: as we have already established, you do not always need the model to solve extremely complex tasks. Therefore, it is often worth sacrificing some percentage points for usability.

## Which benchmark should you use

I’ve just spent the previous section explaining how SWE-bench scores differ from real-world experience. So is it useful at all? I still think it is—especially if your tool is struggling with the tasks you give it. In that case, it may be worth checking whether newer or stronger models are available.

When it comes to measuring raw LLM coding capability, I believe nothing beats the straightforward approach of **SWE-bench Bash Only**. You take a minimal agent and a set of human-verified, complex coding tasks, and you measure success rates across models. To me, this is a very reasonable way to assess model capability while minimizing agent-specific advantages, and it aligns fairly well with real-world observations.

A practical way to use it is:

1. Use the leaderboard to shortlist a few strong models.
2. Try each one in your actual workflow. Some tools (essentially agents), like Cursor, let you switch models out of the box. Some models, like Anthropic’s Claude and OpenAI’s GPTs, also come with vendor-built tools (Claude Code and Codex, respectively). If possible, try both.
3. Decide based on your experience, not on model rank.

Both models and tools evolve quickly, so this is not a one-time decision.

## What to look out for

Different workflows value different things. A developer making many small, interactive changes will experience tools very differently from someone running larger, asynchronous tasks. I often prefer agents that can take a bigger task and run in the background while I work on something else. This reduces supervision but increases the risk of large, messy diffs—which can be disastrous in large legacy codebases.

Because of this, the exact decision criteria will depend on how you work. That said, if you still feel lost, here is a small set of criteria that I believe are fairly universal:

1. **Developer attention time**  
   How much active supervision, correction, and review does the tool require? This often matters more than raw wall-clock time. Be careful, though: there is [evidence](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/) that AI tools can slow developers down because waiting, verification, and correction overhead dominates, despite users *feeling* faster.

2. **Rework burden**  
   How much effort does it take to turn the model’s output into something you are comfortable shipping? This includes code smell, test coverage, diff size, comment quality, and adherence to constraints.

3. **Cost**  
   Cost should not be the first criterion, but it is still real. It is often worth paying more for a tool that meaningfully improves productivity than saving money on something that only creates friction. Still, budget caps and limits matter.

## Conclusion

LLM coding benchmarks are not useless, but they are routinely misinterpreted. They measure how a model performs inside a narrow, fixed agent on a curated set of hard tasks—not how productive you will be when using it day to day. **SWE-bench Bash Only** is valuable precisely because it strips away many agent-specific confounders, making it a relatively clean signal of *model-level coding capability*. Use it to filter weak models, then decide based on real workflow fit: interaction cost, rework burden, latency, and trust. Benchmarks should advise the decision, not make it for you.