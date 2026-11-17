# AI-Powered Agentic Workflow for Project Management

Submission by Ty for the Udacity Agentic AI nanodegree, Course 2 (Agentic Workflows).
Pilot product: the Email Router.

## What is here

| Path | What it is | Rubric / instruction it answers |
| --- | --- | --- |
| `phase_1/workflow_agents/base_agents.py` | The agent library: six implemented agents plus the provided `RAGKnowledgePromptAgent` | Agentic Toolkit: agent implementation |
| `phase_1/*.py` (7 scripts) | One test script per agent | Agentic Toolkit: agent testing |
| `outputs/phase_1/*.txt` (7 files) | Terminal output of each test script, captured as it ran | Evidence of successful test script execution |
| `phase_2/workflow_agents/base_agents.py` | Copy of the Phase 1 library, as the starter README instructs | Phase 2 setup |
| `phase_2/agentic_workflow.py` | The project-management workflow | Workflow setup, agents, support functions, orchestration |
| `outputs/phase_2/agentic_workflow_output.txt` | Terminal output of the workflow run with the default prompt | Final structured output for the Email Router |
| `outputs/phase_2/variant_*.txt` | Runs with other workflow prompts, and one with the code structure checks switched off for comparison | Stand-out suggestions |
| `reflection.md` | Strengths, limitations and one proposed improvement | Stand-out suggestion |
| `requirements.txt` | The course's dependency list | Needed by the commands below |

Not included, as the instructions require: virtual environments, the `.env` file, `__pycache__`, and data files.
`Product-Spec-Email-Router.txt` comes from the course starter (`project/starter/phase_2/`); copy it into `phase_2/` to run the workflow.

## Checking the outputs without running anything

Every output file begins with the command that produced it and ends with its exit status.

- **The seven test scripts:** `ls outputs/phase_1` lists one file per script, including `rag_knowledge_prompt_agent.txt`.
- **Knowledge source and persona prints:** `grep -n "Knowledge source" outputs/phase_1/direct_prompt_agent.txt outputs/phase_1/augmented_prompt_agent.txt` and `grep -n "Knowledge check" outputs/phase_1/knowledge_augmented_prompt_agent.txt`.
- **One consolidated plan, not only the last step:** the workflow output ends with `*** Consolidated Project Plan ***` followed by `--- User Stories ---`, `--- Product Features ---` and `--- Engineering Tasks ---`.
- **Structure of each section:** under every entry in the plan, a `[structure check: ...]` line counts, in code, how many stories match "As a ..., I want ... so that ..." and how many features and tasks carry every required field.

## Running it

Python 3.13 with the packages in `requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
printf 'OPENAI_API_KEY=%s\n' "your-course-key" > .env   # read by every script; never committed

cd phase_1
python direct_prompt_agent.py            # likewise for the other six scripts
cd ../phase_2
python agentic_workflow.py               # default prompt: full plan for the Email Router
python agentic_workflow.py --prompt "Define only the key features for the Email Router product"
python agentic_workflow.py --no-structure-check   # evaluation agents decide alone (comparison)
```

`agentic_workflow.py` exits 0 when every step completed, 1 when a step failed, 2 when there was nothing to run.

## Design notes

- **Evaluation agent.** Accepts an answer only when the evaluator's first word is Yes; it returns exactly `final_response`, `evaluation` and `iterations`. In the workflow each role's evaluation agent also runs a check in code: every story must read "As a ..., I want ... so that ...", and every feature and every task must carry each required field label. An answer is accepted only when both agree, and whatever the code check finds is sent back to the worker as feedback. `variant_evaluator_only.txt` shows a run with the code checks off. If the limit is reached without a Yes, the workflow keeps the last answer and marks it `[NOT accepted by the evaluation agent]` in the plan rather than presenting it as approved.
- **Routing.** Embedding similarity always picks the closest route, even when none fits, and an embedding does not understand "does not". The Program Manager and Development Engineer descriptions therefore name their own work only, and do not mention the other roles' work.
- **Stories flow downstream.** Each step is routed on its own short text, so on their own the Program Manager and Development Engineer would never see the user stories. Once the Product Manager's stories are accepted, they are added to those two agents' knowledge, so features group real stories and each task's "Related User Story" quotes one. The routing input and the support functions are unchanged.
- **No repeats in the plan.** Several planned steps can reach the same role (for example "define the tasks" and then "compile the development plan"), and each answer can restate work already done. The consolidated plan lists each story, feature and task once: a later item with the same story sentence, the same Feature Name, or the same Task ID and Task Title is left out, and the plan says how many were left out under each entry. The step-by-step log above the plan still shows every answer in full.
- **Coverage.** A complete plan needs all three sections. If no planned step reached a role, the workflow runs that role once directly, and the plan labels that step `[added by coverage check]`.
- **Failures.** Each step is isolated: a failed API call is logged, reported in the plan under "Steps that did not complete", and the remaining steps still run. Requests time out after 300 seconds (a long task list can take more than a minute to generate) and are retried twice by the client.
