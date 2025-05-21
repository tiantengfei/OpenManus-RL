# openmanus_rl/agentgym/agentenv/agentenv/tools/base.py
from typing import Dict, Any

class BaseTool:
    name: str = "base_tool"
    description: str = "Base class for tools"
    # JSON schema for describing parameters to an LLM or for validation
    parameters: Dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        '''
        Executes the tool with the given keyword arguments.
        Returns a string representing the observation or result of the tool execution.
        '''
        raise NotImplementedError("Each tool must implement its own execute method.")

    def __init__(self):
        # Ensure tools have a name, description, and parameters when instantiated
        if not self.name or self.name == "base_tool":
            raise ValueError("Tool name must be set.")
        if not self.description or self.description == "Base class for tools":
            raise ValueError("Tool description must be set.")
        # Parameters can be empty if a tool takes no arguments.
