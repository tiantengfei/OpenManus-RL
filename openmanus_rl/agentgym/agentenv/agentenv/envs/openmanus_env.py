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

        # Dynamically construct the system prompt
        base_system_message = "You are OpenManus, an all-capable AI assistant, aimed at solving any task presented by the user. You have various tools at your disposal that you can call upon to efficiently complete complex requests. Whether it's programming, information retrieval, file processing, web browsing, or human interaction (only for extreme cases), you can handle it all."
        core_message = base_system_message # Default

        tools_section_str = ""
        if self.tools:
            tools_list_parts = []
            for tool in self.tools.values():
                schema_dict = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                tools_list_parts.append(json.dumps(schema_dict))
            
            tools_details = "\n".join(tools_list_parts)

            tools_section_str = (
                "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                f"{tools_details}\n"
                "</tools>\n\n"
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
            )

        # Assemble the final system prompt value
        # It includes the full structure with <|im_start|> and <|im_end|> tags and a trailing newline.
        self.system_prompt_value = (
            f"<|im_start|>system\n{core_message}{tools_section_str}<|im_end|>\n"
        )
        
        # The initial observation is simply the system prompt.
        self.current_observation_text = self.system_prompt_value
        
        self.current_step: int = 0
        self.task_completed: bool = False
        self.latest_status_info: Dict[str, Any] = {"message": "Session initialized."}
        self.next_prompt_value = self.config.get("next_prompt_value", NEXT_STEP_PROMPT)
        # self.system_prompt_value is now set dynamically above
        # self.current_observation_text is also set dynamically above

        print(f"[LocalToolExecutor] Initialized for task: '{self.task_description}'. Max steps: {self.max_steps}.")
        print(f"[LocalToolExecutor] Available tools: {list(self.tools.keys())}")
        print(f"[LocalToolExecutor] System Prompt: {self.system_prompt_value[:500]}...") # Log part of the new prompt

    def get_initial_observation(self) -> str:
        return self.current_observation_text

    def get_current_observation(self) -> str:
        return self.current_observation_text

    async def process_action(self, actions: List[str]) -> None:
        self.current_step += 1
        tool_executed_successfully_overall = False
        observation_parts = [] # New: Initialize list for observation parts

        if self.task_completed:
            # This part remains the same, sets current_observation_text directly
            self.current_observation_text = "Tried to act on a completed task. No state change."
            print("[LocalToolExecutor] Warning: Action processed on already completed task.")
            return

        if not actions:
            self.current_observation_text = "No actions were performed." # New: Specific message
            self.latest_status_info = {"success": True, "message": "Empty action list processed."}
            # Max steps check for this scenario
            if not self.task_completed and self.current_step >= self.max_steps:
                self.task_completed = True
                # Append to the "No actions" message if max steps reached here
                self.current_observation_text += "\nMax steps reached. Episode terminated."
                self.latest_status_info = {"success": False, "message": "Terminated due to max steps with empty action."}
            print(f"[LocalToolExecutor] Step {self.current_step}: Action(s)='{actions}', Done={self.task_completed}")
            print(f"[LocalToolExecutor] Observation: {self.current_observation_text[:200]}...")
            return
            
        self.latest_status_info = {"success": False, "message": "Processing actions."} # Default for non-empty list

        for i, action_json_string in enumerate(actions):
            if self.task_completed:
                break 

            action_item = None
            tool_name = "UnknownTool" # Default for logging if tool_name cannot be extracted
            args_for_log = {} # Default for logging
            tool_executed_successfully_item = False
            formatted_action_observation = ""

            try:
                action_item = json.loads(action_json_string)
                if not isinstance(action_item, dict):
                    raise ToolError(f"Parsed action is not a dictionary. Action string: '{action_json_string}'")

                tool_name = action_item.get("function_name")
                if not tool_name:
                    # Specific error for missing function_name, before calling_tool_line
                    error_message = f"Error: Missing 'function_name' in action: {action_json_string}"
                    observation_parts.append(f"{i+1}. {error_message}")
                    self.latest_status_info = {"success": False, "message": "Missing 'function_name'."}
                    continue # to next action_json_string

                args = action_item.get("arguments")
                if args is None: args = {}
                elif not isinstance(args, dict):
                    # Simplified args validation for logging, actual validation in tool
                    print(f"[LocalToolExecutor] Warning: 'arguments' in action is not a dict. Action: {action_json_string}")
                    # Depending on tool, this might be an error or handled by tool's arg parsing
                args_for_log = args # For use in calling_tool_line

                calling_tool_line = f"Calling tool {tool_name}: {json.dumps(args_for_log)}"

                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    tool_result = await tool.execute(**args)
                    tool_executed_successfully_item = True
                    self.latest_status_info = {"success": True, "message": f"Tool {tool_name} executed."}
                    formatted_action_observation = f"{i+1}. {calling_tool_line}\n\nResult: {tool_result}"
                    observation_parts.append(formatted_action_observation)

                    if tool_name == self.terminate_tool.name:
                        self.task_completed = True
                        status_arg = args.get("status", "success")
                        self.latest_status_info = {"success": status_arg == "success", "message": f"Terminated by agent with status: {status_arg}"}
                        # Observation from Terminate tool is its result, already captured.
                        break 
                else:
                    error_message = f"Unknown tool '{tool_name}'. Available tools: {', '.join(self.tools.keys())}"
                    formatted_action_observation = f"{i+1}. {calling_tool_line}\n\nError: {error_message}"
                    observation_parts.append(formatted_action_observation)
                    self.latest_status_info = {"success": False, "message": f"Unknown tool: {tool_name}."}

            except json.JSONDecodeError:
                error_message = f"Could not parse action JSON: {action_json_string}"
                observation_parts.append(f"{i+1}. Error: {error_message}")
                self.latest_status_info = {"success": False, "message": "Action JSON parsing error."}
                continue 
            except ToolError as e: # Covers missing function_name if not caught earlier, or other validation issues
                error_message = str(e)
                # calling_tool_line might not be fully formed if tool_name was missing
                if 'calling_tool_line' not in locals(): # tool_name might be missing
                    calling_tool_line = f"Attempted call with malformed action: {action_json_string}"
                formatted_action_observation = f"{i+1}. {calling_tool_line}\n\nError: {error_message}"
                observation_parts.append(formatted_action_observation)
                self.latest_status_info = {"success": False, "message": f"Invalid action item: {error_message}"}
            except (BrowserToolError, EditorToolError) as e: 
                error_message = str(e)
                formatted_action_observation = f"{i+1}. {calling_tool_line}\n\nError: {error_message}"
                observation_parts.append(formatted_action_observation)
                self.latest_status_info = {"success": False, "message": f"Tool error with {tool_name}: {error_message}"}
            except TypeError as e: 
                error_message = str(e)
                formatted_action_observation = f"{i+1}. {calling_tool_line}\n\nError: Argument mismatch for tool {tool_name}. Details: {error_message}"
                observation_parts.append(formatted_action_observation)
                self.latest_status_info = {"success": False, "message": f"Tool argument mismatch for {tool_name}: {error_message}"}
            except Exception as e: 
                error_message = str(e)
                formatted_action_observation = f"{i+1}. {calling_tool_line}\n\nError: Unexpected error: {error_message}"
                observation_parts.append(formatted_action_observation)
                self.latest_status_info = {"success": False, "message": f"Unexpected error with tool {tool_name}: {error_message}"}
            
            if tool_executed_successfully_item:
                tool_executed_successfully_overall = True
        
        # After loop, assemble current_observation_text from parts
        if observation_parts:
            self.current_observation_text = "\n\n".join(observation_parts)
        elif not actions: # Already handled, but as a safeguard for current_observation_text
             self.current_observation_text = "No actions were performed."
        else: # Actions list was not empty, but observation_parts is - e.g. all actions failed before appending.
              # self.latest_status_info should hold the last error.
              # self.current_observation_text might still hold text from a *previous* step if not cleared.
              # For safety, set a generic message if nothing specific was recorded.
              if self.latest_status_info.get("message") == "Processing actions.": # No specific error/success
                   self.current_observation_text = "Actions processed, but no specific observations were generated."
              # else, if there were errors, observation_parts should not be empty. This is a fallback.


        if not self.task_completed and self.current_step >= self.max_steps:
            self.task_completed = True
            # Append to the potentially multi-part observation
            self.current_observation_text += "\n\nMax steps reached. Episode terminated." 
            self.latest_status_info = {"success": False, "message": "Terminated due to max steps."}
        
        if not self.task_completed and self.latest_status_info.get("message") == "Processing actions.":
            if tool_executed_successfully_overall:
                self.latest_status_info = {"success": True, "message": "Tool(s) executed."}
            # If not overall success and message is still "Processing actions", means no tool succeeded and no specific error message
            # was set for the whole batch (e.g. all actions were valid but no-op, or all failed individually and their errors were logged in parts)
            # In this case, self.latest_status_info might still be the last individual error, which is fine.
            # Or, if all individual actions somehow led to no status update (unlikely), set a generic one.
            elif not tool_executed_successfully_overall :
                 self.latest_status_info = {"success": False, "message": "No tool executed successfully or actions resulted in no specific outcome."}


        print(f"[LocalToolExecutor] Step {self.current_step}: Action(s)='{actions}', Done={self.task_completed}")
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

    def step(self, actions: List[str]) -> StepOutput: # Signature changed
        if not self.tool_executor:
            print("[OpenManusLocalEnvClient] Step called before reset or executor is None.")
            return StepOutput(state="Error: Tool executor not initialized.", reward=0.0, done=True)

        try:
            # Run the async process_action method using asyncio.run()
            # This creates a new event loop, runs the coroutine, and closes the loop.
            asyncio.run(self.tool_executor.process_action(actions)) # Argument changed
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                print("[OpenManusLocalEnvClient][ERROR] asyncio.run() cannot be called from a running event loop. This indicates a conflict with the current threading/async setup, possibly within Ray. The application's async model needs review for proper integration.")
                # Re-raising allows the higher-level framework (Ray) to catch this critical error.
                raise RuntimeError(f"Asyncio conflict in OpenManusLocalEnvClient.step: {e}. Check async execution context.") from e
            else:
                # Re-raise other RuntimeErrors (e.g., loop closed, etc.)
                raise

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
