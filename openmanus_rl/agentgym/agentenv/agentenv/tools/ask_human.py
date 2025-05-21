# openmanus_rl/agentgym/agentenv/agentenv/tools/ask_human.py
from .base import BaseTool
import sys

class AskHuman(BaseTool):
    name: str = "ask_human"
    description: str = "Use this tool to ask a human for help or input."
    parameters: dict = {
        "type": "object",
        "properties": {
            "inquire": { # Renamed from 'inquire' to 'prompt' for clarity, matching input()
                "type": "string",
                "description": "The question or prompt you want to ask the human.",
            }
        },
        "required": ["inquire"],
    }

    async def execute(self, inquire: str) -> str:
        """
        Asks the human for input via the command line.
        Note: This is a blocking call.
        """
        # In a more complex system, this might involve a different way to get human input,
        # especially if the environment is not run interactively.
        # For now, direct input() is ported.
        try:
            # Ensure stdout is flushed so the prompt appears before input is read.
            sys.stdout.flush()
            response = input(f"ASSISTANT ASKS: {inquire}\nYOUR RESPONSE: ")
            return response.strip()
        except EOFError:
            # Handle cases where input stream might be closed (e.g., non-interactive execution)
            return "Human input stream closed, could not get a response."
        except KeyboardInterrupt:
            # Allow graceful interruption if user hits Ctrl+C during input
            print("\nHuman input interrupted.")
            return "Human input was interrupted."
