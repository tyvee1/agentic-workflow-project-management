"""Test script for DirectPromptAgent.

Sends one question straight to the model, with no system prompt, and prints
the answer followed by where that answer's knowledge came from.
"""

from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

# Instantiate the DirectPromptAgent and send it the prompt
direct_agent = DirectPromptAgent(openai_api_key)
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(f"Prompt: {prompt}")
print(direct_agent_response)

# Explain the knowledge source used by the agent to generate the response
print("Knowledge source: the DirectPromptAgent sends the prompt with no system prompt, persona or "
      "supplied knowledge, so this answer comes only from the general knowledge the gpt-4.1-mini "
      "model learned during training.")
