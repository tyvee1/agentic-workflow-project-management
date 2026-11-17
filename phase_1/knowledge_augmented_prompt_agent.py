"""Test script for KnowledgeAugmentedPromptAgent.

Gives the agent deliberately false knowledge ("the capital of France is
London") and shows that the answer follows the supplied knowledge, not the
model's own.
"""

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

# Instantiate the agent with the persona and the (deliberately false) knowledge
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)
knowledge_agent_response = knowledge_agent.respond(prompt)

print(f"Prompt: {prompt}")
print(knowledge_agent_response)

# Confirm the agent used the provided knowledge rather than its inherent knowledge
print(f"Knowledge check: the agent was given the knowledge \"{knowledge}\". The model's own "
      f"knowledge says Paris; an answer naming London shows it used the provided knowledge "
      f"instead of its inherent knowledge. "
      f"Answer names London: {'London' in knowledge_agent_response}.")
