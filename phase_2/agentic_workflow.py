# agentic_workflow.py
"""Agentic workflow for technical project management (pilot: Email Router).

How it works:
  1. An Action Planning Agent turns the workflow prompt into a list of steps.
  2. A Routing Agent sends each step to the closest role by meaning:
     Product Manager (user stories), Program Manager (product features) or
     Development Engineer (engineering tasks).
  3. Each role is a Knowledge Augmented Prompt Agent paired with an
     Evaluation Agent, which checks the answer against the role's required
     structure and asks for corrections until it passes or the limit is hit.
  4. The results are assembled into one project plan with a section per role.

Usage:
    python agentic_workflow.py                      # the default workflow prompt
    python agentic_workflow.py --no-structure-check # let the evaluation agents decide alone
    python agentic_workflow.py --prompt "Define only the key features for the Email Router product"

Exit status: 0 when every step completed, 1 when any step failed, 2 when the
plan had no steps or the product spec could not be read.
"""

import argparse
import logging
import os
import re
import sys

from dotenv import load_dotenv

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

# Progress and problems go to stderr with timestamps; the plan itself goes to stdout.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
log = logging.getLogger("workflow")

parser = argparse.ArgumentParser(description="Agentic project-management workflow")
parser.add_argument("--prompt", help="workflow prompt to use instead of the default")
parser.add_argument("--no-structure-check", action="store_true",
                    help="accept answers on the evaluation agent's verdict alone, without the code checks")
parser.add_argument("--max-interactions", type=int, default=5,
                    help="evaluation rounds per step (default 5)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Load the OpenAI key into a variable called openai_api_key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    log.error("OPENAI_API_KEY is not set. Put it in a .env file next to this script or above it.")
    sys.exit(2)

# Load the product spec document Product-Spec-Email-Router.txt into product_spec
SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Product-Spec-Email-Router.txt")
try:
    with open(SPEC_PATH, encoding="utf-8") as spec_file:
        product_spec = spec_file.read()
except OSError as error:
    log.error("Could not read the product spec at %s (%s). Copy Product-Spec-Email-Router.txt "
              "from the course starter (project/starter/phase_2/) next to this script.", SPEC_PATH, error)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Structure checks done in code. Each evaluation agent uses its role's check:
# an answer is accepted only when the evaluator says Yes AND the check finds
# nothing, and anything it finds is fed back to the worker. A model's Yes is a
# judgement; a missing "Description:" line is a fact code can see.
# ---------------------------------------------------------------------------

STORY_PATTERN = re.compile(r"\bAs an? [^,]+?, I want .+? so that .+", re.I)
FEATURE_FIELDS = ("Feature Name", "Description", "Key Functionality", "User Benefit")
TASK_FIELDS = ("Task ID", "Task Title", "Related User Story", "Description",
               "Acceptance Criteria", "Estimated Effort", "Dependencies")


def story_problems(text):
    """Problems with a list of user stories: none found, or a story line that
    starts "As a" but lacks "I want ... so that ..."."""
    lines = [ln.strip(" -*•0123456789.)\t") for ln in text.splitlines()]
    candidates = [ln for ln in lines if re.match(r"As an? ", ln, re.I)]
    if not candidates:
        return ["no line starts with 'As a'"]
    bad = [ln for ln in candidates if not STORY_PATTERN.search(ln)]
    return ["story without 'I want ... so that ...': %r" % ln[:70] for ln in bad[:3]]


LABEL = r"(?im)^[\s\-*#•\d.)]*\**%s\**\s*:"


def item_problems(text, fields):
    """Problems with a list of structured items (features or tasks).

    The text is split into items at each line that starts with the first
    field's label ("Feature Name:" / "Task ID:"); every item must carry every
    field label at the start of a line. Returns e.g.
    ["item 3 ('T-03') has no 'Description:' line"]."""
    starts = [m.start() for m in re.finditer(LABEL % re.escape(fields[0]), text)]
    if not starts:
        return [f"no line starts with '{fields[0]}:'"]
    problems = []
    for n, (a, b) in enumerate(zip(starts, starts[1:] + [len(text)]), 1):
        item = text[a:b]
        name = item.split(":", 1)[1].strip().splitlines()[0][:30] if ":" in item else ""
        for field in fields[1:]:
            if not re.search(LABEL % re.escape(field), item):
                problems.append(f"item {n} ({name!r}) has no '{field}:' line")
    return problems


def feature_problems(text):
    """Problems with a list of product features; [] when every feature is complete."""
    return item_problems(text, FEATURE_FIELDS)


def task_problems(text):
    """Problems with a list of engineering tasks; [] when every task is complete."""
    return item_problems(text, TASK_FIELDS)


def field_counts(text, fields):
    """How many times each required field label appears at the start of a line."""
    return {f: len(re.findall(LABEL % re.escape(f), text)) for f in fields}


def structure_score(role, text):
    """A one-line, code-computed summary of how well ``text`` fits its role's structure."""
    if role == "Product Manager":
        lines = [ln.strip(" -*•0123456789.)\t") for ln in text.splitlines()]
        stories = [ln for ln in lines if re.match(r"As an? ", ln, re.I)]
        good = sum(1 for ln in stories if STORY_PATTERN.search(ln))
        return f"{good} of {len(stories)} story lines follow 'As a ..., I want ... so that ...'"
    fields = FEATURE_FIELDS if role == "Program Manager" else TASK_FIELDS
    counts = field_counts(text, fields)
    items = counts[fields[0]]
    complete = min(counts.values())
    return f"{items} item(s); every field present in {complete} of them ({', '.join(f'{k}={v}' for k, v in counts.items())})"


# ---------------------------------------------------------------------------
# Instantiate all the agents
# ---------------------------------------------------------------------------

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    # The product spec loaded above is appended, so the stories come from it.
    + product_spec
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_product_manager, knowledge_product_manager)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_product_manager_eval,
    evaluation_criteria_product_manager,
    product_manager_knowledge_agent,
    args.max_interactions,
    structure_check=None if args.no_structure_check else story_problems,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_program_manager, knowledge_program_manager)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria_program_manager,
    program_manager_knowledge_agent,
    args.max_interactions,
    structure_check=None if args.no_structure_check else feature_problems,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_dev_engineer, knowledge_dev_engineer)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_dev_engineer,
    development_engineer_knowledge_agent,
    args.max_interactions,
    structure_check=None if args.no_structure_check else task_problems,
)

# ---------------------------------------------------------------------------
# Job function persona support functions
# ---------------------------------------------------------------------------

# Which role handled each routed step, and whether its evaluator accepted the
# answer. Filled by the support functions; read when the plan is assembled.
route_log = []


def record_verdict(role, result):
    """Note which role answered and whether its evaluation agent accepted the answer."""
    accepted = EvaluationAgent.is_accepted(result["evaluation"])
    route_log.append({"role": role, "accepted": accepted, "iterations": result["iterations"]})
    if not accepted:
        log.warning("%s: answer NOT accepted after %d interaction(s); keeping the last answer, marked.",
                    role, result["iterations"])


def product_manager_support_function(query):
    """Route target for user stories.

    Passes the step to the Product Manager evaluation agent, which calls its
    own worker (the Product Manager knowledge agent) and checks the answer;
    returns the evaluated stories.
    """
    result = product_manager_evaluation_agent.evaluate(query)
    record_verdict("Product Manager", result)
    return result["final_response"]


def program_manager_support_function(query):
    """Route target for product features: evaluated by the Program Manager evaluation agent."""
    result = program_manager_evaluation_agent.evaluate(query)
    record_verdict("Program Manager", result)
    return result["final_response"]


def development_engineer_support_function(query):
    """Route target for engineering tasks: evaluated by the Development Engineer evaluation agent."""
    result = development_engineer_evaluation_agent.evaluate(query)
    record_verdict("Development Engineer", result)
    return result["final_response"]


def share_user_stories(stories):
    """Give the Program Manager and Development Engineer the accepted user stories.

    Each step is routed on its own short text, so without this the features
    and tasks are written without ever seeing the stories, and a task's
    "Related User Story" can name a story that does not exist. The stories
    are added to the two agents' knowledge; the routing input and the support
    functions are unchanged.
    """
    shared = ("\nThe user stories already defined for this product are listed below. "
              "Base your answer on them, and when you refer to a user story, quote one of these exactly:\n" + stories)
    program_manager_knowledge_agent.knowledge = knowledge_program_manager + shared
    development_engineer_knowledge_agent.knowledge = knowledge_dev_engineer + shared
    log.info("Shared %d accepted user stories with the Program Manager and Development Engineer.",
             len(re.findall(r"(?im)\bAs an? ", stories)))


# Routing Agent
routing_agent = RoutingAgent(openai_api_key, {})
routes = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only. "
                       "Does not define features or tasks. Does not group stories",
        "func": lambda x: product_manager_support_function(x),
    },
    {
        "name": "Program Manager",
        # Descriptions avoid naming the other roles' work: an embedding has no
        # notion of "does not", so "does not write user stories" would pull
        # story steps toward this route.
        "description": "Responsible for defining product features: feature names, descriptions, "
                       "key functionality and user benefit, grouping related capabilities of the product",
        "func": lambda x: program_manager_support_function(x),
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for defining the development and engineering tasks needed to build "
                       "the product: task IDs, technical work, acceptance criteria, effort estimates and dependencies",
        "func": lambda x: development_engineer_support_function(x),
    },
]
routing_agent.agents = routes

# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "Produce a complete project plan for the Email Router product, including user stories, product features, and engineering tasks."
# ****
if args.prompt:
    workflow_prompt = args.prompt
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
if not workflow_steps:
    log.error("The action planning agent returned no steps; nothing to do.")
    sys.exit(2)
print("Workflow steps:")
for number, step in enumerate(workflow_steps, 1):
    print(f"  {number}. {step}")

completed_steps = []
step_records = []  # (step, role, accepted, result) for the consolidated plan
failures = 0

for number, step in enumerate(workflow_steps, 1):
    print(f"\n=== Step {number} of {len(workflow_steps)}: {step} ===")
    logged_before = len(route_log)
    try:
        # Route on the short step text only.
        result = routing_agent.route(step)
    except Exception as error:  # one failed step is reported, the others still run
        failures += 1
        log.error("Step %d failed: %s: %s", number, type(error).__name__, error)
        step_records.append((step, None, False, f"[step failed: {type(error).__name__}: {error}]"))
        continue
    completed_steps.append(result)
    entry = route_log[-1] if len(route_log) > logged_before else {"role": None, "accepted": False}
    step_records.append((step, entry["role"], entry["accepted"], result))
    if entry["role"] == "Product Manager" and entry["accepted"]:
        share_user_stories(result)
    print(f"\nResult of step {number} ({entry['role']}):\n{result}")

# Coverage check: a complete plan needs all three sections. Routing always
# picks the closest role, so a plan whose steps all land on one or two roles
# would leave a section empty. Each missing role is run once, directly, and
# labelled as such in the plan.
COVERAGE_STEPS = {
    "Product Manager": ("Define the user stories for the product described in the product spec",
                        product_manager_support_function),
    "Program Manager": ("Define the product features by grouping related user stories for the product",
                        program_manager_support_function),
    "Development Engineer": ("Define the engineering tasks needed to implement the product's user stories",
                             development_engineer_support_function),
}
if not args.prompt:  # only the default prompt asks for all three sections
    for role, (coverage_step, support_function) in COVERAGE_STEPS.items():
        if any(r[1] == role for r in step_records):
            continue
        log.warning("No planned step was routed to %s; running it once to complete the plan.", role)
        print(f"\n=== Coverage step for {role}: {coverage_step} ===")
        try:
            result = support_function(coverage_step)
        except Exception as error:
            failures += 1
            log.error("Coverage step for %s failed: %s: %s", role, type(error).__name__, error)
            continue
        completed_steps.append(result)
        step_records.append((f"{coverage_step} [added by coverage check]", role, route_log[-1]["accepted"], result))
        if role == "Product Manager" and route_log[-1]["accepted"]:
            share_user_stories(result)
        print(f"\nResult of coverage step ({role}):\n{result}")

# ---------------------------------------------------------------------------
# Consolidated project plan
# ---------------------------------------------------------------------------

def _norm(text):
    """Lower-case words only, so formatting differences do not hide a repeat."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def remove_repeats(role, text, seen):
    """Drop items an earlier plan entry already contains.

    Several planned steps can reach the same role ("define the tasks", then
    "compile the development plan"), and each answer repeats work the plan
    already holds. Stories repeat when the same sentence appears again;
    features when the same Feature Name does; tasks when the same Task ID
    has the same Task Title. A reused ID with a different title is kept: it
    is a different task. ``seen`` carries keys across the role's entries.
    Returns (kept text, items kept, items removed).
    """
    if role == "Product Manager":
        kept, removed, count = [], 0, 0
        for line in text.splitlines():
            m = STORY_PATTERN.search(line)
            if m:
                key = _norm(m.group(0))
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
                count += 1
            kept.append(line)
        return "\n".join(kept), count, removed
    fields = FEATURE_FIELDS if role == "Program Manager" else TASK_FIELDS
    starts = [m.start() for m in re.finditer(LABEL % re.escape(fields[0]), text)]
    if not starts:
        return text, 1, 0
    parts, removed = [text[:starts[0]]], 0
    for a, b in zip(starts, starts[1:] + [len(text)]):
        item = text[a:b]
        first = item.split(":", 1)[1].strip().splitlines()[0] if ":" in item else ""
        title = re.search(LABEL % re.escape(fields[1]) + r"(.*)", item)
        key = (_norm(first), _norm(title.group(1)) if title and role == "Development Engineer" else "")
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        parts.append(item)
    return "".join(parts), len(parts) - 1, removed


SECTIONS = (("User Stories", "Product Manager"),
            ("Product Features", "Program Manager"),
            ("Engineering Tasks", "Development Engineer"))

print("\n*** Consolidated Project Plan ***\n")
print(f"Product spec: {os.path.basename(SPEC_PATH)}\nWorkflow prompt: {workflow_prompt}\n"
      f"Steps planned: {len(workflow_steps)}; routed and completed: "
      f"{sum(1 for r in step_records if r[1] and not r[0].endswith('[added by coverage check]'))}; "
      f"added by coverage check: {sum(1 for r in step_records if r[0].endswith('[added by coverage check]'))}; "
      f"failed: {failures}\n")
for section, role in SECTIONS:
    print(f"--- {section} ---")
    entries = [r for r in step_records if r[1] == role]
    if not entries:
        print("(no step in this run was routed to this role)\n")
        continue
    seen = set()
    for step, _, accepted, result in entries:
        note = "" if accepted else "  [NOT accepted by the evaluation agent]"
        kept, kept_items, removed = remove_repeats(role, result, seen)
        print(f"[from step: {step}]{note}")
        if removed and not kept_items:
            print(f"[all {removed} item(s) in this answer repeat entries above; not shown again]\n")
            continue
        if removed:
            print(f"[{removed} item(s) already listed above were left out]")
        print(f"[structure check: {structure_score(role, kept)}]")
        print(kept.strip() + "\n")
unrouted = [r for r in step_records if r[1] is None]
if unrouted:
    print("--- Steps that did not complete ---")
    for step, _, _, result in unrouted:
        print(f"{step}: {result}")

print("\n*** Workflow execution finished ***")
sys.exit(1 if failures else 0)
