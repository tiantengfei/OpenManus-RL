# openmanus_rl/agentgym/agentenv/agentenv/tools/terminate.py
from .base import BaseTool
from typing import Literal # For Python 3.8+

_TERMINATE_DESCRIPTION = """Terminate the interaction when the request is met OR 
if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work."""

class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    async def execute(self, status: Literal["success", "failure"]) -> str:
        """Signals task completion. The LocalToolExecutor will use this to set done=True."""
        # The actual termination (setting done=True and reward) will be handled
        # by the LocalToolExecutor based on this tool's call.
        # This tool itself just returns a confirmation message.
        return f"The interaction has been signaled to complete with status: {status}"
