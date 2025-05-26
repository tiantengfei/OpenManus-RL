from typing import Any, Mapping, Optional, Dict, Literal
import json
import re # For parsing tool calls
import asyncio

from ..controller import BaseEnvClient, BaseTask, StepOutput, ConversationMessage

# Import tools (assuming they are in ../tools/)
from ..tools.base import BaseTool
from ..tools.python_execute import PythonExecute
from ..tools.terminate import Terminate
from ..tools.ask_human import AskHuman
from ..tools.str_replace_editor import StrReplaceEditor, ToolError as EditorToolError
from ..tools.browser_use_tool_wrapper import BrowserUseToolWrapper, ToolError as BrowserToolError



SYSTEM_PROMPT = (
    "You are OpenManus, an all-capable AI assistant, aimed at solving any task presented by the user. You have various tools at your disposal that you can call upon to efficiently complete complex requests. Whether it's programming, information retrieval, file processing, web browsing, or human interaction (only for extreme cases), you can handle it all."
    "The initial directory is: {directory}"
)

NEXT_STEP_PROMPT = """
Based on user needs, proactively select the most appropriate tool or combination of tools. For complex tasks, you can break down the problem and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest the next steps.

If you want to stop the interaction at any point, use the `terminate` tool/function call.
"""

# Define a common ToolError if not already centralized, or use specific ones
# For now, specific tool errors can be caught, or a general one if defined in base.py
class ToolError(Exception): # General error for the executor
    pass

class LocalToolExecutor:
    def __init__(self, task_description: str, config: Optional[Dict[str, Any]] = None):
        self.task_description = task_description
        self.config = config if config else {}
        self.max_steps = self.config.get("max_steps", 20) # Max steps for the episode
        
        # Initialize tools
        # Config for tools can be passed from self.config if needed
        browser_tool_config = self.config.get("browser_tool_config", {})
        str_editor_workspace_root = self.config.get("str_editor_workspace_root", None) # Example

        self.python_tool = PythonExecute()
        self.terminate_tool = Terminate()
        self.ask_human_tool = AskHuman()
        self.str_replace_editor_tool = StrReplaceEditor(workspace_root=str_editor_workspace_root)
        self.browser_tool = BrowserUseToolWrapper(browser_config_args=browser_tool_config)

        self.tools: Dict[str, BaseTool] = {
            self.python_tool.name: self.python_tool,
            self.terminate_tool.name: self.terminate_tool,
            self.ask_human_tool.name: self.ask_human_tool,
            self.str_replace_editor_tool.name: self.str_replace_editor_tool,
            self.browser_tool.name: self.browser_tool,
        }
        
        self.current_step: int = 0
        self.task_completed: bool = False
        self.latest_status_info: Dict[str, Any] = {"message": "Session initialized."}
        self.next_prompt_value = self.config.get("next_prompt_value", NEXT_STEP_PROMPT)
        self.system_prompt_value = self.config.get("system_prompt_value", SYSTEM_PROMPT)
        self.current_observation_text: str = f"<|im_start|>system\n{self.system_prompt_value}<|im_end|>\n"

        print(f"[LocalToolExecutor] Initialized for task: '{self.task_description}'. Max steps: {self.max_steps}.")
        print(f"[LocalToolExecutor] Available tools: {list(self.tools.keys())}")

    def get_initial_observation(self) -> str:
        return self.current_observation_text

    def get_current_observation(self) -> str:
        return self.current_observation_text

    async def process_action(self, action_str: str) -> None:
        self.current_step += 1
        tool_executed_successfully = False
        
        if self.task_completed: # Prevent action if already completed
            self.current_observation_text = "Tried to act on a completed task. No state change."
            print("[LocalToolExecutor] Warning: Action processed on already completed task.")
            return

        try:
            parsed_action = json.loads(action_str)
            if not isinstance(parsed_action, dict):
                raise ToolError("Action must be a JSON object (dict).")

            tool_name = parsed_action.get("function_name")
            if not tool_name:
                raise ToolError("Missing 'function_name' in action JSON.")

            args = parsed_action.get("arguments")
            if args is None:
                args = {}
            elif not isinstance(args, dict):
                # Attempt to parse arguments if it's a string, otherwise default to empty dict
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                        if not isinstance(args, dict):
                            print(f"[LocalToolExecutor] Warning: 'arguments' field, when parsed from string, was not a dict. Type: {type(args)}. Using empty dict for tool args.")
                            args = {}
                    except json.JSONDecodeError:
                        print(f"[LocalToolExecutor] Warning: 'arguments' field was a string but not valid JSON: {args}. Using empty dict for tool args.")
                        args = {}
                else:
                    print(f"[LocalToolExecutor] Warning: 'arguments' field was not a dict or JSON string. Type: {type(args)}. Using empty dict for tool args.")
                    args = {}
            
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                try:
                    # Execute the tool
                    self.current_observation_text = await tool.execute(**args) # Tool output becomes new observation
                    tool_executed_successfully = True 

                    if tool_name == self.terminate_tool.name:
                        self.task_completed = True
                        status = args.get("status", "success") # Default to success if not specified
                        self.latest_status_info = {
                            "success": status == "success",
                            "message": f"Terminated by agent with status: {status}"
                        }
                # Specific tool errors are caught first
                except (ToolError, BrowserToolError, EditorToolError) as e: 
                    self.current_observation_text = f"Error executing tool {tool_name}: {e}"
                    self.latest_status_info = {"success": False, "message": f"Tool error: {e}"}
                except TypeError as e: # Catch argument mismatch errors (e.g. missing required arg)
                     self.current_observation_text = f"Error: Argument mismatch for tool {tool_name}. Details: {e}. Args provided: {args}"
                     self.latest_status_info = {"success": False, "message": f"Tool argument mismatch: {e}"}
                except Exception as e: # Catch any other unexpected error during tool execution
                    self.current_observation_text = f"Unexpected error executing tool {tool_name}: {str(e)}"
                    self.latest_status_info = {"success": False, "message": f"Unexpected tool error: {e}"}
            else:
                self.current_observation_text = f"Error: Unknown tool '{tool_name}'. Available tools: {', '.join(self.tools.keys())}"
                self.latest_status_info = {"success": False, "message": "Unknown tool."}
        
        except json.JSONDecodeError:
            self.current_observation_text = (
                f"Error: Action is not a valid JSON string: '{action_str}'. "
                f"Please provide action in JSON format with 'function_name' and 'arguments'."
            )
            self.latest_status_info = {"success": False, "message": "Action JSON parsing error."}
        except ToolError as e: # Catch custom ToolErrors raised from our parsing logic
            self.current_observation_text = f"Error: Invalid action format. {e}"
            self.latest_status_info = {"success": False, "message": f"Invalid action format: {e}"}
        except Exception as e: # Catch any other unexpected error during action parsing
            self.current_observation_text = f"Unexpected error processing action: {str(e)}"
            self.latest_status_info = {"success": False, "message": f"Unexpected action processing error: {e}"}

        if not self.task_completed and self.current_step >= self.max_steps:
            self.task_completed = True
            # Append to observation rather than replacing, so tool output isn't lost
            self.current_observation_text += "\nMax steps reached. Episode terminated."
            self.latest_status_info = {"success": False, "message": "Terminated due to max steps."}
        
        if tool_executed_successfully and not self.task_completed:
            self.latest_status_info = {"success": True, "message": "Tool executed."}
        
        print(f"[LocalToolExecutor] Step {self.current_step}: Action='{action_str}', Done={self.task_completed}")
        print(f"[LocalToolExecutor] Observation: {self.current_observation_text[:200]}...")


    def is_done(self) -> bool:
        return self.task_completed

    def get_reward(self) -> float:
        if not self.task_completed:
            return self.config.get("reward_step", 0.0) 

        if self.latest_status_info.get("success", False):
            return self.config.get("reward_success", 1.0)
        
        message = self.latest_status_info.get("message", "").lower()
        if "max steps" in message: # Timeout due to max_steps
             return self.config.get("reward_timeout", -1.0) # Penalize timeout
        return self.config.get("reward_failure", -0.5) # General failure, less penalty than timeout

    async def cleanup(self):
        print("[LocalToolExecutor] Cleaning up resources...")
        if hasattr(self.browser_tool, 'cleanup') and callable(self.browser_tool.cleanup):
            await self.browser_tool.cleanup()
        # Add other tool cleanups if needed, e.g. for StrReplaceEditor if it had temp files/dirs
        print("[LocalToolExecutor] Cleanup finished.")


class OpenManusLocalEnvClient(BaseEnvClient): # Renamed
    conversation_start = (ConversationMessage({"from": "human", "loss": None, "value": "Goal:"}),)

    def __init__(
        self,
        env_server_base: str, 
        data_len: int,
        *args,
        env_specific_config: Optional[Dict[str, Any]] = None, 
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_len = data_len
        self.tool_executor: Optional[LocalToolExecutor] = None # Renamed
        self.current_task_idx: Optional[int] = None
        
        self.executor_config = env_specific_config.copy() if env_specific_config else {}
        self.executor_config.setdefault("reward_step", 0.0)
        self.executor_config.setdefault("reward_success", 1.0)
        self.executor_config.setdefault("reward_failure", -0.5) 
        self.executor_config.setdefault("reward_timeout", -1.0) 
        self.executor_config.setdefault("max_steps", 20) # Default max steps for an episode

        print(f"[OpenManusLocalEnvClient] Initialized. Executor config: {self.executor_config}")

    def __len__(self):
        return self.data_len

    def observe(self) -> str:
        if self.tool_executor:
            return self.tool_executor.get_current_observation()
        print("[OpenManusLocalEnvClient] Observe called before reset or executor is None.")
        return "Environment not initialized. Please call reset."

    async def step(self, action: str) -> StepOutput: # Made async
        if not self.tool_executor:
            print("[OpenManusLocalEnvClient] Step called before reset or executor is None.")
            return StepOutput(state="Error: Tool executor not initialized.", reward=0.0, done=True)

        await self.tool_executor.process_action(action) # process_action is now async
        
        state = self.tool_executor.get_current_observation()
        reward = self.tool_executor.get_reward()
        done = self.tool_executor.is_done()

        return StepOutput(state=state, reward=reward, done=done)

    def reset(self, idx: int, task_description: Optional[str] = None) -> Dict[str, Any]:
        self.current_task_idx = idx
        actual_task_desc = task_description if task_description else f"Default Task ID: {idx}"

        # If there's an old executor, clean it up before creating a new one.
        if self.tool_executor and hasattr(self.tool_executor, 'cleanup'):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule and wait for cleanup if loop is running
                    # This might be an issue if reset is called from a context where await isn't possible
                    # For now, we assume reset is not called from a deeply synchronous part of an async flow
                    # or that ensure_future is sufficient for fire-and-forget if await is not possible.
                    # A better pattern might be to make reset async as well if cleanup must be awaited.
                    asyncio.ensure_future(self.tool_executor.cleanup())
                    print("[OpenManusLocalEnvClient] Scheduled cleanup of old tool executor.")
                else:
                    # Fallback if no loop is running - this is tricky for async cleanup
                    # loop.run_until_complete(self.tool_executor.cleanup()) # This would block and might error if loop is closed
                    print("[OpenManusLocalEnvClient] Warning: Event loop not running. Cleanup of old executor might be incomplete.")
            except RuntimeError: 
                 print("[OpenManusLocalEnvClient] Warning: No event loop for cleanup of old executor.")
            except Exception as e:
                 print(f"[OpenManusLocalEnvClient] Error during old executor cleanup in reset: {e}")


        print(f"[OpenManusLocalEnvClient] Resetting. Task: '{actual_task_desc}'")
        self.tool_executor = LocalToolExecutor(
            task_description=actual_task_desc,
            config=self.executor_config
        )
        initial_obs_string = self.tool_executor.get_initial_observation()
        initial_next_prompt = self.tool_executor.next_prompt_value
        return {"observation": initial_obs_string, "next_prompt": initial_next_prompt}

    async def close(self): 
        print("[OpenManusLocalEnvClient] Closing environment client...")
        if self.tool_executor:
            await self.tool_executor.cleanup()
            self.tool_executor = None # Release reference
        print("[OpenManusLocalEnvClient] Environment client closed.")


class OpenManusLocalTask(BaseTask): # Renamed
    env_client_cls = OpenManusLocalEnvClient
    env_name = "openmanus_local" # New name for registration

    def __init__(
        self,
        client_args: Mapping[str, Any], 
        n_clients: int,
        *args,
        **kwargs,
    ):
        # client_args should contain 'env_server_base', 'data_len', 
        # and 'env_specific_config' (formerly openmanus_config) for LocalToolExecutor
        print(f"[OpenManusLocalTask] Initializing with client_args: {client_args}, n_clients: {n_clients}")
        super().__init__(client_args, n_clients, *args, **kwargs)

    async def close(self): # Ensure this is async if BaseTask.close can be awaited or if clients need async close
        print(f"[{self.env_name}] Closing task and its clients...")
        # BaseTask.close is not async, so super().close() should be called normally.
        # The primary concern is ensuring our async clients are closed properly.
        
        # Close clients first
        for client in self.clients:
            if hasattr(client, 'close') and callable(client.close):
                try:
                    await client.close() # Call the async close method of OpenManusLocalEnvClient
                except Exception as e:
                    print(f"[{self.env_name}] Error closing client {type(client)}: {e}")

        # Then call superclass close if it exists and is not what we just did for clients
        # BaseTask.close() in agentenv.controller is synchronous and just iterates self.clients, calling client.close()
        # if it exists. Since we've already done an async close, we can skip super().close()
        # or ensure BaseTask.close() is robust to already closed clients or is also made async.
        # For now, let's assume our client.close() is sufficient.
        # if hasattr(super(), "close") and callable(super().close):
        #     try:
        #          super().close() 
        #     except Exception as e:
        #          print(f"[{self.env_name}] Error in super().close(): {e}")
        print(f"[{self.env_name}] Task and clients closed.")
