import sys
import os
from openmanus_rl.agentgym.agentenv.agentenv.controller.env import BaseEnvClient, StepOutput

# Attempt to add OpenManus to sys.path
# This assumes 'OpenManus' is a directory located four levels above the current file's directory.
# (i.e., envs/ -> agentenv/ -> agentgym/ -> openmanus_rl/ -> ProjectRoot/OpenManus)
# Adjust the relative path ('../../../../OpenManus') if the submodule is located differently.
openmanus_submodule_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'OpenManus'))

if openmanus_submodule_path not in sys.path:
    sys.path.insert(0, openmanus_submodule_path)

try:
    from app.agent.manus import Manus
    from app.schema import AgentState
    # If Manus.create() is blocking or needs an event loop, this async context might be an issue
    # For now, assume it's awaitable as planned.
    print("Successfully imported Manus and AgentState from OpenManus submodule.")
except ImportError as e:
    print(f"Failed to import Manus or AgentState from OpenManus submodule: {e}. Using placeholders.")
    # Define placeholder classes if actual import fails
    class Manus:
        def __init__(self):
            self.state = "IDLE_PLACEHOLDER"
            self.current_step = 0
            self.max_steps = 10
            self.memory = type('Memory', (), {'messages': []})() # Simplified memory mock

        async def create(self, **kwargs):
            # This is a class method in the actual Manus class.
            # For a placeholder instance, we'll mimic it as an instance method returning self.
            print("Placeholder Manus.create called")
            return self

        async def cleanup(self):
            print("Placeholder Manus.cleanup called")

        def update_memory(self, role: str, content: str):
            print(f"Placeholder Manus.update_memory called with role='{role}', content='{content}'")
            # Simulate adding to a messages list if needed for observe() later
            self.memory.messages.append(type('Message', (), {'content': content, 'role': role})())


        async def step(self):
            print("Placeholder Manus.step called")
            self.current_step += 1
            if self.current_step >= self.max_steps:
                self.state = "FINISHED_PLACEHOLDER"
            # Simulate some action
            self.memory.messages.append(type('Message', (), {'content': 'Placeholder step observation', 'role': 'assistant'})())
            return "Placeholder step result content"

    class AgentState:
        FINISHED = "FINISHED_PLACEHOLDER"
        IDLE = "IDLE_PLACEHOLDER"
        # Add other states if they are directly accessed or set by Manus.create()
        # For now, these cover the planned interactions.

import asyncio # Ensure asyncio is imported at the top

# Predefined tasks list (should already be there)
PREDEFINED_TASKS = [
    "Write a Python script to print numbers from 1 to 10.",
    "What is the capital of France? Answer concisely.",
    "Use the browser to find the current weather in New York City.",
]

class OpenManusEnvClient(BaseEnvClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.predefined_tasks = PREDEFINED_TASKS
        self.manus_agent: Manus = None # Will be initialized in reset
        self.current_task_initial_prompt: str = ""
        self.last_observation: str = ""

    def __len__(self) -> int:
        return len(self.predefined_tasks)

    def observe(self) -> str:
        return self.last_observation

    async def step(self, action: str) -> StepOutput:
        if self.manus_agent is None:
            print("Error: Manus agent not initialized. Call reset() first.")
            # Return a StepOutput indicating an error state or terminal state
            return StepOutput(state="Error: Agent not initialized. Call reset() first.", reward=0.0, done=True)

        if self.manus_agent.state == AgentState.FINISHED:
             # If already finished, further steps shouldn't change state but signal done.
            print("Info: Agent was already in FINISHED state.")
            # Observation should be the last thing observed.
            # If messages exist, use the last one, otherwise use current last_observation.
            obs_content = self.manus_agent.memory.messages[-1].content if self.manus_agent.memory.messages else self.last_observation
            return StepOutput(state=obs_content, reward=0.0, done=True)

        try:
            # Add the RL agent's action as a user message to Manus agent's memory
            self.manus_agent.update_memory("user", action)

            # Execute one think-act cycle of the Manus agent
            # The ReActAgent.step() method (which ToolCallAgent inherits) returns a string,
            # but the core updates happen in memory.
            await self.manus_agent.step() # This calls think then act

            # Determine the new observation
            # The last message in memory is usually the most relevant observation
            # (either assistant's thought or tool's output message)
            new_observation = ""
            if self.manus_agent.memory.messages:
                new_observation = self.manus_agent.memory.messages[-1].content
            else:
                # Fallback if memory is somehow empty, though unlikely after a step
                new_observation = "No new observation from agent."
            
            self.last_observation = new_observation

            # Determine done status
            done = False
            if self.manus_agent.state == AgentState.FINISHED:
                done = True
            
            # Check if Manus agent reached its internal max_steps
            if self.manus_agent.current_step >= self.manus_agent.max_steps:
                done = True
                if self.manus_agent.state != AgentState.FINISHED: # Agent didn't naturally finish
                    print(f"Info: Manus agent reached its max_steps ({self.manus_agent.max_steps}).")
                    # Optionally, set Manus agent's state to FINISHED if max_steps implies termination
                    # self.manus_agent.state = AgentState.FINISHED 
                    # For now, just ensure 'done' is true for the environment.

            # Determine reward
            reward = 0.0
            if done and self.manus_agent.state == AgentState.FINISHED:
                # Task successfully completed by Manus agent's own logic (e.g., Terminate tool)
                reward = 1.0
            # If done due to max_steps but not AgentState.FINISHED, reward remains 0.0 (or could be negative)
            # If not done, reward is 0.0 (step cost)

            return StepOutput(state=new_observation, reward=reward, done=done)

        except Exception as e:
            print(f"Error during Manus agent step: {e}")
            # Consider what observation to return. Last known, or error message.
            error_message = f"Error during agent execution: {str(e)}"
            self.last_observation = error_message
            # If an error occurs, it's usually a terminal state for the episode
            return StepOutput(state=error_message, reward=0.0, done=True)

    async def reset(self, idx: int) -> None:
        if self.manus_agent is not None:
            # Ensure cleanup is called on the existing instance
            await self.manus_agent.cleanup()

        if not (0 <= idx < len(self.predefined_tasks)):
            print(f"Warning: Task index {idx} is out of bounds (0-{len(self.predefined_tasks)-1}). Resetting to task 0.")
            idx = 0
        
        initial_prompt = self.predefined_tasks[idx]
        
        # Create and initialize the Manus agent
        # Manus.create() is an async class method, so it should be awaited directly
        try:
            self.manus_agent = await Manus.create() # This uses the imported or placeholder Manus
        except Exception as e:
            print(f"Error creating Manus agent: {e}")
            # Fallback or re-raise if critical
            # For now, if Manus.create() fails with real Manus, it might leave self.manus_agent as None or an older instance.
            # If using placeholders, it should proceed.
            # Consider how to handle this failure robustly.
            # If Manus() placeholder is used, Manus.create() is an instance method.
            # This might need adjustment if the placeholder structure for Manus() vs Manus.create() is different.
            # The provided placeholder has create() as an instance method.
            # Let's assume Manus() creates an instance if Manus.create() is problematic for placeholders.
            if 'Manus.create() is not a coroutine' in str(e) or 'create() missing 1 required positional argument' in str(e): # Common placeholder issues
                 print("Attempting placeholder Manus instantiation due to create() issue.")
                 self.manus_agent = Manus() # Instantiating placeholder

        self.current_task_initial_prompt = initial_prompt
        if self.manus_agent: # Proceed only if agent creation was successful
            self.manus_agent.update_memory("user", initial_prompt)
            self.last_observation = initial_prompt
            # Ensure the agent's internal state is reset if not handled by Manus.create()
            self.manus_agent.current_step = 0 
            self.manus_agent.state = AgentState.IDLE # Assuming AgentState.IDLE exists
        else:
            print("Error: Manus agent not created in reset. Subsequent operations might fail.")
            self.last_observation = "Error: Agent not initialized."

if __name__ == '__main__':
    async def main_test():
        print("--- Starting OpenManusEnvClient Test ---")
        
        # Attempt to create environment instance
        # Add any required __init__ params if they exist, though current plan has none
        env = OpenManusEnvClient() 
        
        print("\n---> Resetting environment to task 0...")
        await env.reset(idx=0) # reset returns None
        initial_obs = env.observe()
        print(f"Initial Observation (Task 0): {initial_obs}")

        sample_action_1 = "What is the first step to solve this?"
        print(f"\n---> Taking Step 1 with action: '{sample_action_1}'")
        step_output_1 = await env.step(sample_action_1)
        print(f"Step 1 Output: State='{step_output_1.state}', Reward={step_output_1.reward}, Done={step_output_1.done}")

        # Only proceed if not done
        if not step_output_1.done:
            sample_action_2 = "Okay, proceed with the next logical action."
            print(f"\n---> Taking Step 2 with action: '{sample_action_2}'")
            step_output_2 = await env.step(sample_action_2)
            print(f"Step 2 Output: State='{step_output_2.state}', Reward={step_output_2.reward}, Done={step_output_2.done}")
        else:
            print("\n---> Skipping Step 2 because agent reported Done after Step 1.")

        print("\n--- OpenManusEnvClient Test Finished ---")

    # asyncio.run(main_test()) # This is already imported at the top.
    # If it wasn't, then:
    # import asyncio
    asyncio.run(main_test())
