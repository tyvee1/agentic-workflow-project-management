# Reflection

Ty

## Strengths

- **Reusable parts.** Each agent class does one job and takes everything it needs (key, persona, knowledge, criteria) as arguments, so the same classes serve the Phase 1 tests and the Phase 2 workflow without change.
- **The plan comes from the request.** The action planning agent decides the steps from the workflow prompt, so a narrower prompt such as "define only the key features" produces a shorter plan without code changes. The variant outputs show this.
- **Every answer is checked.** Each role's answer passes through an evaluation agent with explicit criteria, and the plan states whether each answer was accepted and how well it matches the required structure, counted in code.
- **Failures are visible.** A failed step, an unaccepted answer and a section filled by the coverage check are each labelled in the plan rather than hidden.

## Limitations

- **The evaluator is the same model as the worker.** It tends to share the worker's blind spots, and its Yes is a judgement, not proof. Structure that code can measure (the story pattern, the required field labels) is more reliable when code checks it, so each role's evaluation agent also runs a code check; the `--no-structure-check` comparison run shows what the model-only evaluation lets through.
- **Downstream roles depend on step order.** Each step is routed on its own short text, as the instructions require. The workflow passes the accepted user stories into the Program Manager's and Development Engineer's knowledge, but only once the stories exist: if the plan puts a task step before the story step, those tasks are still written without the stories. A first run without this sharing produced tasks citing a log-in story the Email Router spec never mentions.
- **Routing has no refusal.** Embedding similarity always selects some role. A step that fits none of them (for example a risk assessment) is still sent to the closest one, which answers in its own format.
- **Cost grows with retries.** Each evaluation round is three model calls. The limit bounds it, but a criterion the worker cannot meet spends every round.

## One improvement

**Check every task's story reference in code.** The tasks now receive the user stories, but nothing yet verifies that each task's "Related User Story" quotes one of them. Adding that to the Development Engineer's structure check would turn an invented reference into correction feedback, in the same way a missing field is today, and running the planned steps in dependency order (stories, then features, then tasks) would remove the step-order limitation above.
