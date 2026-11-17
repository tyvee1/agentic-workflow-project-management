"""Test script for AugmentedPromptAgent.

Asks the same question as the direct agent, but through a persona set in the
system prompt, and prints the answer.
"""

from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate the agent with the persona and send it the prompt
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(f"Prompt: {prompt}")
print(augmented_agent_response)

# Knowledge used: no knowledge was supplied, so the facts in the answer (Paris)
# come from the model's general training data, exactly as with the direct agent.
#
# Effect of the persona: the system prompt changes how the answer is delivered,
# not what is known. The reply opens with "Dear students," and is phrased as a
# professor would address a class; the underlying fact is unchanged.
print("Knowledge source: the model's general training data (no knowledge was supplied).")
print("Persona effect: the system prompt sets the voice and opening ('Dear students,'), "
      "not the facts.")
