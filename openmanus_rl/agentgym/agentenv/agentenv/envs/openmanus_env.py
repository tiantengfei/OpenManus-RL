from typing import Any, Mapping, Optional, Dict

from agentenv.controller import BaseEnvClient, BaseTask, StepOutput, ConversationMessage

# --- BEGIN HYPOTHETICAL REAL OpenManus Library Integration ---
# IMPORTANT: The following imports and class names are HYPOTHETICAL.
# They need to be verified against the actual FoundationAgents/OpenManus library structure.
# Assumed structure:
# import openmanus.core.executor as openmanus_executor
# from openmanus.core.responses import StepResult as OpenManusStepResult

# For now, to make the code runnable if openmanus is not yet installed or structured this way,
# we'll try-except the import and fall back to a mock if it fails.

try:
    # Attempt to import the hypothetical real OpenManus components
    from openmanus.core.executor import OpenManusSessionExecutor
    # Assuming StepResult might be part of the executor module or a responses module
    from openmanus.core.executor import StepResult as OpenManusStepResult
    OPENMANUS_LIB_AVAILABLE = True
    print("[OpenManusSessionProxy] Successfully imported 'openmanus' library components.")
except ImportError:
    OPENMANUS_LIB_AVAILABLE = False
    print("[OpenManusSessionProxy] WARNING: 'openmanus' library not found or structured as expected. Falling back to MOCK implementation.")

    # --- Fallback Mock OpenManus Library Interface (if real one fails to import) ---
    class OpenManusStepResult: # Renamed MockOpenManusStepResult
        def __init__(self, observation: str, is_terminated: bool, status_info: Dict[str, Any]):
            self.observation = observation
            self.is_terminated = is_terminated
            self.status_info = status_info

    class OpenManusSessionExecutor: # Renamed MockOpenManusSessionExecutor
        def __init__(self, task_description: str, config: Optional[Dict[str, Any]] = None):
            self.task_description = task_description
            self.config = config or {}
            self.max_steps = self.config.get("max_steps", 10)
            self.current_step = 0
            self._current_observation = f"Task initialized (MOCK): '{self.task_description}'. Waiting for action."
            self._is_terminated = False
            print(f"[OpenManusSessionExecutor-MOCK] Session for '{task_description}' initialized. Max steps: {self.max_steps}.")

        def get_initial_observation(self) -> str:
            return self._current_observation

        def execute_step(self, action: str) -> OpenManusStepResult:
            self.current_step += 1
            status_info = {"success": False, "message": ""}

            if self._is_terminated:
                self._current_observation = "Session already terminated (MOCK)."
                status_info["message"] = "Action attempted on terminated session (MOCK)."
                return OpenManusStepResult(self._current_observation, self._is_terminated, status_info)

            if "success" in action.lower():
                self._current_observation = f"Action '{action}' led to task success (MOCK)."
                self._is_terminated = True
                status_info["success"] = True
                status_info["message"] = "Task completed successfully (MOCK)."
            elif "fail" in action.lower():
                self._current_observation = f"Action '{action}' led to task failure (MOCK)."
                self._is_terminated = True
                status_info["success"] = False
                status_info["message"] = "Task failed based on action (MOCK)."
            elif self.current_step >= self.max_steps:
                self._current_observation = f"Reached max steps ({self.max_steps}). Task terminating (MOCK)."
                self._is_terminated = True
                status_info["success"] = False
                status_info["message"] = f"Terminated due to max steps ({self.max_steps}) (MOCK)."
            else:
                self._current_observation = f"After action '{action}' (step {self.current_step}) (MOCK): Still ongoing."
                status_info["message"] = "Step processed, task ongoing (MOCK)."
            
            print(f"[OpenManusSessionExecutor-MOCK] execute_step: Obs='{self._current_observation[:50]}...', Done={self._is_terminated}")
            return OpenManusStepResult(self._current_observation, self._is_terminated, status_info)
    # --- End Fallback Mock OpenManus Library Interface ---

# --- END HYPOTHETICAL REAL OpenManus Library Integration ---


class OpenManusSessionProxy:
    def __init__(self, task_description: str, config: Optional[Dict[str, Any]] = None):
        print(f"[OpenManusSessionProxy] Initializing with task: '{task_description}' and config: {config}")
        self.task_description = task_description
        self.config = config if config else {}
        
        # Instantiate the actual (or mock if unavailable) OpenManus session executor
        self.executor = OpenManusSessionExecutor(task_description, self.config)
        
        self.current_observation_text: str = self.executor.get_initial_observation()
        self.task_completed: bool = False 
        self.latest_status_info: Dict[str, Any] = {"message": "Session just initialized"}

        print(f"[OpenManusSessionProxy] Initialized. Initial obs: {self.current_observation_text[:100]}...")

    def get_initial_observation(self) -> str:
        return self.current_observation_text

    def get_current_observation(self) -> str:
        return self.current_observation_text

    def process_action(self, action: str) -> None:
        print(f"[OpenManusSessionProxy] Processing action: '{action}'")
        
        if self.task_completed:
            print("[OpenManusSessionProxy] Warning: Action processed on already completed task.")
            self.current_observation_text = "Tried to act on a completed task."
            return

        # Interact with the OpenManus library (real or mock)
        step_result: OpenManusStepResult = self.executor.execute_step(action)
        
        self.current_observation_text = step_result.observation
        self.task_completed = step_result.is_terminated
        self.latest_status_info = step_result.status_info
        
        print(f"[OpenManusSessionProxy] New observation: {self.current_observation_text[:100]}...")
        if self.task_completed:
            print(f"[OpenManusSessionProxy] Task is now completed. Status: {self.latest_status_info.get('message')}")

    def is_done(self) -> bool:
        return self.task_completed

    def get_reward(self) -> float:
        if not self.task_completed:
            return self.config.get("reward_step", 0.0)

        if self.latest_status_info.get("success", False):
            return self.config.get("reward_success", 1.0)
        
        # Check for specific termination messages for differentiated rewards
        message = self.latest_status_info.get("message", "").lower()
        if "max steps" in message:
             return self.config.get("reward_timeout", -0.1)
        # Add other failure conditions if OpenManus provides more detailed status_info
        # For now, any other terminated state that isn't success is a general failure.
        return self.config.get("reward_failure", -1.0)


class OpenManusEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {"from": "human", "loss": None, "value": "Goal:"}
        ),
    )

    def __init__(
        self,
        env_server_base: str, # Not used by this client but part of BaseTask's expected args
        data_len: int,
        *args,
        openmanus_config: Optional[Dict[str, Any]] = None,
        timeout: int = 300,   # Not used by this client
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_len = data_len
        self.session_proxy: Optional[OpenManusSessionProxy] = None
        self.current_task_idx: Optional[int] = None
        
        self.openmanus_session_config = openmanus_config.copy() if openmanus_config else {}
        self.openmanus_session_config.setdefault("reward_step", 0.0)
        self.openmanus_session_config.setdefault("reward_success", 1.0)
        self.openmanus_session_config.setdefault("reward_failure", -1.0)
        self.openmanus_session_config.setdefault("reward_timeout", -0.1)
        self.openmanus_session_config.setdefault("max_steps", 10) # Default for mock

        print(f"[OpenManusEnvClient] Initialized. Data len: {data_len}. Session config: {self.openmanus_session_config}")
        if not OPENMANUS_LIB_AVAILABLE:
            print("[OpenManusEnvClient] Using MOCK OpenManus library.")


    def __len__(self):
        return self.data_len

    def observe(self) -> str:
        if self.session_proxy:
            return self.session_proxy.get_current_observation()
        print("[OpenManusEnvClient] Observe called before reset or session_proxy is None.")
        return "Environment not initialized. Please call reset."

    def step(self, action: str) -> StepOutput:
        if not self.session_proxy:
            print("[OpenManusEnvClient] Step called before reset or session_proxy is None.")
            return StepOutput(
                state="Error: OpenManus session not initialized. Call reset first.",
                reward=0.0,
                done=True,
            )

        self.session_proxy.process_action(action)
        
        state = self.session_proxy.get_current_observation()
        reward = self.session_proxy.get_reward()
        done = self.session_proxy.is_done()

        return StepOutput(state=state, reward=reward, done=done)

    def reset(self, idx: int, task_description: Optional[str] = None) -> str:
        self.current_task_idx = idx
        
        actual_task_description: str
        if task_description: # This might not be passed by OpenManusAgent; idx is primary
            actual_task_description = task_description
        else:
            # The 'idx' could be used to fetch a specific task from a dataset
            # or directly by the OpenManus library if it manages its own task set.
            actual_task_description = f"Task ID: {idx}" 

        print(f"[OpenManusEnvClient] Resetting to task_idx: {idx}, Task Description: '{actual_task_description}'")
        
        self.session_proxy = OpenManusSessionProxy(
            task_description=actual_task_description,
            config=self.openmanus_session_config
        )
        
        initial_observation = self.session_proxy.get_initial_observation()
        print(f"[OpenManusEnvClient] Reset complete. Initial obs: {initial_observation[:100]}...")
        return initial_observation


class OpenManusTask(BaseTask):
    env_client_cls = OpenManusEnvClient
    env_name = "OpenManus" # Name used in OpenManusAgent's ENV_TO_TASK_CLASS map

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        print(f"[OpenManusTask] Initializing with client_args: {client_args}, n_clients: {n_clients}")
        super().__init__(client_args, n_clients, *args, **kwargs)
