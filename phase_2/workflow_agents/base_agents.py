"""Reusable agent library for agentic workflows.

Seven agent classes, each a building block for larger workflows:

    DirectPromptAgent              prompt in, model answer out, nothing added
    AugmentedPromptAgent           the same, answered in a given persona
    KnowledgeAugmentedPromptAgent  answers from supplied knowledge only
    RAGKnowledgePromptAgent        retrieves the most relevant chunk, then answers (provided)
    EvaluationAgent                checks a worker agent's answer against criteria and
                                   asks it to correct itself, up to a limit
    RoutingAgent                   sends a prompt to the agent whose description is
                                   closest in meaning (embedding similarity)
    ActionPlanningAgent            turns a request into a clean list of steps

Every agent receives its API key when it is created; no key is stored in this
file. Every model call goes through the course's OpenAI-compatible endpoint.
"""

import re
import csv
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from openai import OpenAI

# The course endpoint. Every client in this file is created against it.
BASE_URL = "https://openai.vocareum.com/v1"

# A request that hangs should fail in a bounded time rather than stall a
# workflow; the client retries transient failures (rate limits, 5xx) itself.
# The bound must allow for long answers: a full list of engineering tasks can
# take well over a minute to generate, and a 60-second limit cut one off.
REQUEST_TIMEOUT_SECONDS = 300
MAX_RETRIES = 2


def _client(api_key):
    """Create an OpenAI client for the course endpoint with a bounded timeout."""
    return OpenAI(base_url=BASE_URL, api_key=api_key,
                  timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)


# DirectPromptAgent class definition
class DirectPromptAgent:
    """Relays a prompt to the model unchanged and returns the model's text.

    No system prompt, no persona, no extra knowledge: the answer comes from the
    model's general training alone. Useful as the baseline the other agents
    are compared against.
    """

    def __init__(self, openai_api_key):
        """Store the API key used for every request this agent makes."""
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        """Send ``prompt`` as the only (user) message and return the reply text."""
        client = _client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """Answers in a defined persona.

    The persona goes in the system message, which the model treats as the
    developer's instructions; the user's text goes in the user message.
    """

    def __init__(self, openai_api_key, persona):
        """Store the API key and the persona the agent must adopt."""
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Answer ``input_text`` as the persona and return only the reply text."""
        client = _client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system",
                 "content": f"{self.persona}. Forget all previous context. "
                            "Answer as this persona."},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """Answers in a persona, using only the knowledge it was given.

    The knowledge is placed in the system message with an instruction not to
    use the model's own knowledge. That is an instruction, not a guarantee:
    the model usually follows it, which is what the test script demonstrates
    with deliberately false knowledge.
    """

    def __init__(self, openai_api_key, persona, knowledge):
        """Store the API key, the persona and the knowledge to answer from."""
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Answer ``input_text`` from ``self.knowledge`` and return the reply text."""
        client = _client(self.openai_api_key)
        system_message = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
            f"Use only the following knowledge to answer, do not use your own knowledge: "
            f"{self.knowledge} "
            f"Answer the prompt based on this knowledge, not your own."
        )
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            # break the loop if we have reached the end of the text
            if end == len(text):
                break

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent:
    """Judges a worker agent's answer against criteria, and has it corrected.

    Each interaction: the worker answers; an evaluator call says Yes or No
    with a reason; on No, a second call turns the reason into correction
    instructions, and the worker answers again with them. The loop ends on
    the first Yes or after ``max_interactions``.

    ``evaluate`` returns exactly ``final_response``, ``evaluation`` and
    ``iterations``. When the limit is reached without a Yes, ``final_response``
    is the last answer and ``evaluation`` is the last No: a caller that needs
    to know whether the answer was accepted reads ``evaluation`` (see
    ``is_accepted``).

    ``structure_check`` (optional) is a function that inspects the answer
    with code and returns a list of problems. When given, an answer is only
    accepted if the evaluator says Yes AND the list is empty, and the
    problems are fed back to the worker. Without it, the class behaves
    exactly as the evaluator alone decides.
    """

    def __init__(self, openai_api_key, persona, evaluation_criteria, agent_to_evaluate,
                 max_interactions, structure_check=None):
        """Store the key, the evaluator's persona, the criteria, the worker and the limit."""
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate
        self.max_interactions = max_interactions
        self.structure_check = structure_check

    @staticmethod
    def is_accepted(evaluation):
        """True only if the evaluation's first word is Yes.

        Markdown emphasis and punctuation around the first word are ignored,
        so "**Yes**, it does" is accepted; "No, ... yes ..." and "Yesterday"
        are not.
        """
        first = re.match(r"[\W_]*([A-Za-z]+)", evaluation or "")
        return bool(first) and first.group(1).lower() == "yes"

    def evaluate(self, initial_prompt):
        """Run the answer-judge-correct loop for ``initial_prompt``.

        Returns {"final_response": str, "evaluation": str, "iterations": int}.
        """
        client = _client(self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        response_from_worker, evaluation, iterations = "", "", 0

        for i in range(self.max_interactions):
            iterations = i + 1
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.agent_to_evaluate.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": self.persona},
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0,
            )
            evaluation = response.choices[0].message.content.strip()

            # Optional code check: measurable structure is decided by code, not by the model.
            if self.structure_check is not None:
                problems = self.structure_check(response_from_worker)
                if problems:
                    evaluation = ("No. A structure check in code found: " + "; ".join(problems)
                                  + "\nEvaluator said: " + evaluation)
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if self.is_accepted(evaluation):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "system", "content": self.persona},
                        {"role": "user", "content": instruction_prompt},
                    ],
                    temperature=0,
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        else:
            print(f"⚠️ Not accepted after {self.max_interactions} interactions; returning the last response.")
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": iterations,
        }


class RoutingAgent:
    """Sends a prompt to the agent whose description is closest in meaning.

    ``agents`` is a list of {"name", "description", "func"}. The prompt and
    each description are turned into embeddings (text-embedding-3-large),
    compared by cosine similarity, and the best match's ``func`` is called
    with the prompt.

    Embedding routing always finds a closest agent, even when none fits; the
    printed scores show how close the choice was.
    """

    def __init__(self, openai_api_key, agents):
        """Store the key and the list of route dictionaries."""
        self.openai_api_key = openai_api_key
        self.agents = agents
        self._description_embeddings = {}  # description text -> embedding, reused across calls

    def get_embedding(self, text):
        """Return the text-embedding-3-large vector for ``text``."""
        client = _client(self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        embedding = response.data[0].embedding
        return embedding

    def _description_embedding(self, description):
        """Embedding of a route description, computed once and then reused."""
        if description not in self._description_embeddings:
            self._description_embeddings[description] = self.get_embedding(description)
        return self._description_embeddings[description]

    def route(self, user_input):
        """Call the best-matching agent's ``func`` with ``user_input`` and return its result."""
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self._description_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(f"[Router] {agent['name']}: {similarity:.3f}")

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    """Extracts the steps needed to carry out a request, using its knowledge.

    The model is asked for a list; the reply is then cleaned in code into one
    step per item: blank lines, headings and list markers are removed.
    """

    def __init__(self, openai_api_key, knowledge):
        """Store the key and the knowledge the steps must come from."""
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    @staticmethod
    def clean_steps(response_text):
        """Turn a model reply into a list of step strings.

        If the reply contains list items (numbered or bulleted), only those
        are kept and their markers removed; otherwise every non-empty line is
        a step. Lines that are only a heading ("Steps:", "# Scrambled Eggs")
        are dropped.
        """
        lines = [ln.strip() for ln in (response_text or "").splitlines() if ln.strip()]
        marker = re.compile(r"^(?:\d+[.)]|[-*•]|Step\s+\d+[:.)])\s+", re.I)
        items = [ln for ln in lines if marker.match(ln)]
        chosen = items if items else lines
        steps = []
        for ln in chosen:
            text = marker.sub("", ln).strip().strip("*").strip()
            if not text or text.startswith("#") or (text.endswith(":") and len(text.split()) <= 4):
                continue
            steps.append(text)
        return steps

    def extract_steps_from_prompt(self, prompt):
        """Return the list of steps for ``prompt``."""
        client = _client(self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system",
                 "content": "You are an action planning agent. Using your knowledge, you extract from the "
                            "user prompt the steps requested to complete the action the user is asking for. "
                            "You return the steps as a list. Only return the steps in your knowledge. "
                            f"Forget any previous context. This is your knowledge: {self.knowledge}"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content
        steps = self.clean_steps(response_text)
        return steps
