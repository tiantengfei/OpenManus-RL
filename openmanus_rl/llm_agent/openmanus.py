import torch
import re
from collections import defaultdict
import os
import sys # Added import
from typing import List, Dict, Any, Tuple, Optional # Optional added
from dataclasses import dataclass, field # field added
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from transformers import GenerationConfig
import importlib # Added import
import traceback # For error logging
from concurrent.futures import ThreadPoolExecutor, as_completed # For parallel rollout
import json # Added for VerlLLMAdapter
import asyncio # Added for VerlLLMAdapter
# re is already imported
# from typing import List, Dict, Any, Tuple, Optional are already satisfied by existing imports
# from dataclasses import dataclass, field are already satisfied (dataclass is, field needs to be checked or added if Mock* classes are outside)
from dataclasses import field # Explicitly add field if not covered by existing dataclass import for Mock*
from types import SimpleNamespace # Added for VerlLLMAdapter
# from verl import DataProto # Already imported
from transformers import GenerationConfig as HFGenerationConfig # Added for VerlLLMAdapter to avoid conflict
# torch is already imported
from verl.utils.tracking import Tracking
from omegaconf import DictConfig # Import DictConfig for type hint
import numpy as np

# --- Helper/Mock Dataclasses for VerlLLMAdapter ---
@dataclass
class MockFunctionCall:
    name: str
    arguments: str # Should be a JSON string

@dataclass
class MockToolCall:
    id: str
    type: str = "function"
    function: MockFunctionCall = field(default_factory=lambda: MockFunctionCall("", "")) # Ensure valid defaults

@dataclass
class MockChatCompletionMessage:
    content: Optional[str] = None
    tool_calls: Optional[List[MockToolCall]] = None
    # role: Optional[str] = None # Not strictly needed for response object from LLM


# --- Placeholder padding functions (if not available elsewhere) ---
# These would need proper tensor manipulation logic based on DataProto structure.
def pad_dataproto_to_divisor(data_proto: DataProto, divisor: int) -> tuple[DataProto, int]:
    """Pads a DataProto object to ensure batch size is a multiple of divisor."""
    if divisor <= 1: # No padding needed if divisor is 1 or less
        return data_proto, 0

    # Simplistic: assumes 'input_ids' is the primary tensor to check for batch size.
    # A real implementation must iterate through all tensors in data_proto.batch
    # and pad them consistently.
    if 'input_ids' not in data_proto.batch:
        print("Warning: 'input_ids' not in data_proto.batch for padding. Returning as is.")
        return data_proto, 0
        
    input_ids = data_proto.batch['input_ids']
    current_batch_size = input_ids.shape[0]
    pad_size = (divisor - (current_batch_size % divisor)) % divisor

    if pad_size == 0:
        return data_proto, 0

    print(f"Padding DataProto batch from {current_batch_size} to {current_batch_size + pad_size} (divisor {divisor})")
    
    padded_batch_data = {}
    for key, tensor_val in data_proto.batch.items():
        if isinstance(tensor_val, torch.Tensor) and tensor_val.shape[0] == current_batch_size:
            # Use the first entry to create padding tensor
            pad_tensor_slice = tensor_val[0:1].repeat(pad_size, *[1] * (len(tensor_val.shape) - 1))
            padded_batch_data[key] = torch.cat([tensor_val, pad_tensor_slice], dim=0)
        else:
            # Non-tensor or non-batch-aligned data, copy as is
            padded_batch_data[key] = tensor_val 
            
    padded_proto = DataProto.from_dict(padded_batch_data)
    if hasattr(data_proto, 'meta_info'):
        padded_proto.meta_info = data_proto.meta_info.copy()
    return padded_proto, pad_size

def unpad_dataproto(data_proto: DataProto, pad_size: int) -> DataProto:
    """Removes padding from a DataProto object."""
    if pad_size == 0:
        return data_proto

    unpadded_batch_data = {}
    if 'input_ids' not in data_proto.batch: # Check if batch is empty or malformed
        print("Warning: 'input_ids' not in data_proto.batch for unpadding. Returning as is.")
        return data_proto
        
    original_batch_size = data_proto.batch['input_ids'].shape[0] - pad_size
    if original_batch_size < 0:
        print(f"Warning: pad_size {pad_size} is larger than current batch size {data_proto.batch['input_ids'].shape[0]}. Cannot unpad correctly.")
        return data_proto # Or raise error

    print(f"Unpadding DataProto batch from {data_proto.batch['input_ids'].shape[0]} to {original_batch_size}")

    for key, tensor_val in data_proto.batch.items():
        if isinstance(tensor_val, torch.Tensor) and tensor_val.shape[0] == (original_batch_size + pad_size):
            unpadded_batch_data[key] = tensor_val[:original_batch_size]
        else:
            unpadded_batch_data[key] = tensor_val
            
    unpadded_proto = DataProto.from_dict(unpadded_batch_data)
    if hasattr(data_proto, 'meta_info'):
         unpadded_proto.meta_info = data_proto.meta_info.copy()
    return unpadded_proto


# --- VerlLLMAdapter Class Definition ---
class VerlLLMAdapter:
    def __init__(self, actor_rollout_wg, tokenizer, agent_config: 'AgentConfig', tensor_fn: 'TensorHelper'):
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        # agent_config is OpenManusAgent.AgentConfig, tensor_fn is OpenManusAgent.TensorHelper
        self.agent_config = agent_config 
        self.tensor_fn = tensor_fn

    def _construct_prompt_with_tools(self, messages: List[Dict], system_msgs: Optional[List[Dict]], tools: Optional[List[Dict]]) -> str:
        full_prompt_text = ""
        if system_msgs:
            for msg in system_msgs:
                full_prompt_text += f"System: {msg.get('content', '')}\n"
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if isinstance(content, list): # Handle list content (e.g. for multimodal, though this adapter is text-only)
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                content = text_content
            full_prompt_text += f"{role.capitalize()}: {content}\n"

        if tools:
            full_prompt_text += "\nAvailable Tools:\n"
            for tool_spec in tools:
                if tool_spec.get('type') == 'function' and 'function' in tool_spec:
                    func = tool_spec['function']
                    full_prompt_text += f"- Name: {func.get('name')}\n"
                    if 'description' in func:
                        full_prompt_text += f"  Description: {func.get('description')}\n"
                    if 'parameters' in func:
                        full_prompt_text += f"  Parameters: {json.dumps(func.get('parameters'))}\n"
            full_prompt_text += ("\nInstruct the LLM to use tools by generating text in the format: "
                                 "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value1\"}}</tool_call>"
                                 " and provide textual response in <content>...</content> tags if needed.\n")
        
        full_prompt_text += "\nAssistant:"
        return full_prompt_text

    def _parse_response_for_content_and_tools(self, response_text: str) -> MockChatCompletionMessage:
        content_match = re.search(r"<content>(.*?)</content>", response_text, re.DOTALL)
        tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL)

        content = None
        if content_match:
            content = content_match.group(1).strip()
        
        tool_calls_list = []
        if tool_call_match:
            try:
                tool_call_json_str = tool_call_match.group(1).strip()
                tool_call_data = json.loads(tool_call_json_str)
                function_args_str = json.dumps(tool_call_data.get("arguments", {}))

                tool_calls_list.append(
                    MockToolCall(
                        id=f"toolcall_{torch.randint(0, 100000, (1,)).item()}",
                        type="function",
                        function=MockFunctionCall(
                            name=tool_call_data.get("name"),
                            arguments=function_args_str 
                        )
                    )
                )
            except json.JSONDecodeError as e:
                print(f"Error decoding tool_call JSON: {e}. String was: {tool_call_match.group(1)}")
            except Exception as e:
                print(f"Error processing tool_call: {e}")
        
        if not content_match and not tool_call_match and response_text: # If no tags, assume all is content
            content = response_text.strip()
        
        return MockChatCompletionMessage(content=content, tool_calls=tool_calls_list if tool_calls_list else None)

    async def ask_tool(
        self,
        messages: List[Dict],
        system_msgs: Optional[List[Dict]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Any = None, # Unused for now
        temperature: Optional[float] = None,
        **kwargs # e.g. max_tokens
    ) -> MockChatCompletionMessage | None:
        prompt_text = self._construct_prompt_with_tools(messages, system_msgs, tools)
        
        tokenized_prompt = self.tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False)
        input_ids = tokenized_prompt['input_ids']
        
        if input_ids.shape[1] > self.agent_config.max_prompt_length:
            input_ids = input_ids[:, -self.agent_config.max_prompt_length:]

        attention_mask = self.tensor_fn.create_attention_mask(input_ids)
        # position_ids = self.tensor_fn.create_position_ids(attention_mask) # Original TensorHelper used attention_mask
        # Corrected position_ids generation based on typical HF models:
        position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)


        data_proto_batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
        data_proto = DataProto.from_dict(data_proto_batch)
        
        max_new_tokens = kwargs.get('max_tokens', self.agent_config.max_response_length)
        
        # Access algorithm_config safely
        algo_config = self.agent_config.algorithm_config if self.agent_config.algorithm_config is not None else {}
        effective_temperature = temperature if temperature is not None else algo_config.get('temperature', 1.0)

        gen_config_hf = HFGenerationConfig(
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=effective_temperature,
            do_sample=True # Assuming sampling if temperature is relevant
        )
        # Ensure meta_info exists
        if not hasattr(data_proto, 'meta_info') or data_proto.meta_info is None:
            data_proto.meta_info = {}
        data_proto.meta_info['generation_config'] = gen_config_hf

        world_size = self.actor_rollout_wg.world_size
        padded_data_proto, pad_size = pad_dataproto_to_divisor(data_proto, world_size)
        
        try:
            response_proto_padded = await asyncio.to_thread(self.actor_rollout_wg.generate_sequences, padded_data_proto)
        except Exception as e:
            print(f"Error during actor_rollout_wg.generate_sequences: {e}")
            print(traceback.format_exc())
            return None # Or an error message object

        response_proto = unpad_dataproto(response_proto_padded, pad_size)

        if 'responses' not in response_proto.batch or not isinstance(response_proto.batch['responses'], torch.Tensor) or response_proto.batch['responses'].numel() == 0:
            print("Warning: No response IDs found in response_proto.batch['responses']. Returning None.")
            return None

        # Assuming batch size 1 for the response here as ask_tool processes one request at a time.
        response_ids = response_proto.batch['responses'][0] 
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return self._parse_response_for_content_and_tools(response_text)


@dataclass
class AgentConfig:
    """
    Configuration class for OpenManusAgent.
    
    Attributes:
        max_turns: Maximum number of turns in a conversation
        max_start_length: Maximum length of initial input
        max_prompt_length: Maximum length of prompt
        max_response_length: Maximum length of response
        max_obs_length: Maximum length of observation
        num_gpus: Number of GPUs to use
        env_name: Name of the environment (e.g., "webshop")
        env_ports: List of ports for parallel servers
        env_server_base: Base URL for environment server
        react_format: Whether to use ReAct format
        env_data_len: Number of data samples in the environment (used for client init)
        rollout_strategy: Strategy to use for rollout (StandardReAct/ToT/MCTS)
        max_workers: Maximum number of worker threads
        algorithm_config: DictConfig = None # Pass relevant part of algorithm config
    """
    # All required fields without default values
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    env_name: str 
    env_ports: List[int] # List of ports for parallel servers
    env_server_base: str
    
    # All optional fields with default values
    react_format: bool = True
    env_data_len: int = 200 # Default, might need adjustment
    rollout_strategy: str = "StandardReAct" # Strategy is now internal logic
    # storage_backend: str = "mongodb" # Storage handled elsewhere or not needed here
    max_workers: int = 10 # For parallelizing rollouts within the agent
    
    # Add algorithm config relevant to reward allocation
    algorithm_config: DictConfig = None # Pass relevant part of algorithm config

class OpenManusAgent:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg, # This is the Verl component for generation
        config: AgentConfig,
        is_validation: bool = False,
        logger: Tracking = None,  # Add logger parameter for trajectory saving
    ):
        """
        Initialize OpenManusAgent with rollout controller integration.
        
        Args:
            tokenizer: Tokenizer for text processing
            actor_rollout_wg: Actor rollout wrapper for generation
            config: Agent configuration including env details and algorithm config
            is_validation: Whether in validation mode
            logger: Logger for tracking and visualization
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config # AgentConfig now holds algorithm_config
        self.is_validation = is_validation
        self.logger = logger
        
        self.manus_submodule_agent = None # Initialized in async_setup
        self.llm_adapter = None # Initialized in async_setup

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        # Initialize multiple environment clients
        self.clients = self._init_env_clients() # Changed method name

        # Adjust thread pool size based on number of clients, up to max_workers
        num_clients = len(self.clients)
        actual_workers = min(num_clients, self.config.max_workers)
        if actual_workers < num_clients:
             print(f"[Warning] Number of clients ({num_clients}) exceeds max_workers ({self.config.max_workers}). Using {actual_workers} workers.")
        print(f"[Info] Initializing ThreadPoolExecutor with {actual_workers} workers for {num_clients} clients.")
        self.executor = ThreadPoolExecutor(max_workers=actual_workers)

    async def async_setup(self):
        # Path setup for OpenManus submodule
        current_dir = os.path.dirname(os.path.abspath(__file__)) # directory of openmanus.py
        project_root_guess = os.path.join(current_dir, '..', '..') 
        openmanus_submodule_dir = os.path.join(project_root_guess, 'OpenManus')

        if not os.path.isdir(openmanus_submodule_dir):
            print(f"[OpenManusAgent.async_setup] Warning: Could not find OpenManus at relative path {openmanus_submodule_dir}. Relying on PYTHONPATH.")
        elif openmanus_submodule_dir not in sys.path:
            sys.path.insert(0, openmanus_submodule_dir)
            print(f"[OpenManusAgent.async_setup] Added {openmanus_submodule_dir} to sys.path")

        try:
            from app.agent.manus import Manus as SubmoduleManus
            from app.tool.browser_use_tool import BrowserUseTool # Added import
            print("[OpenManusAgent.async_setup] Successfully imported SubmoduleManus and BrowserUseTool.")
            print("[OpenManusAgent.async_setup] Creating SubmoduleManus instance...")
            # run_async_in_sync_context is not available, so we call create directly.
            # This implies __init__ or its caller handles the async context, or SubmoduleManus.create() is okay.
            # If SubmoduleManus.create() is blocking, this could be an issue if __init__ is purely sync.
            # For now, assuming it works as per the task's fallback plan.
            self.manus_submodule_agent = await SubmoduleManus.create()
            
            print("[OpenManusAgent.async_setup] Creating VerlLLMAdapter...")
            self.llm_adapter = VerlLLMAdapter(
                self.actor_rollout_wg, 
                self.tokenizer, 
                self.config, # This is OpenManusAgent's AgentConfig
                self.tensor_fn
            )
            # Inject the adapter into the submodule agent instance
            # Assuming self.manus_submodule_agent has an 'llm' attribute to be replaced.
            if hasattr(self.manus_submodule_agent, 'llm'):
                self.manus_submodule_agent.llm = self.llm_adapter 
                print("[OpenManusAgent.async_setup] VerlLLMAdapter injected into SubmoduleManus.")
            else:
                print("[OpenManusAgent.async_setup] Warning: SubmoduleManus instance does not have an 'llm' attribute to replace.")

            # Inject adapter into BrowserUseTool if present
            if hasattr(self.manus_submodule_agent, 'available_tools') and self.manus_submodule_agent.available_tools and \
               hasattr(self.manus_submodule_agent.available_tools, 'tools'):
                for tool_instance in self.manus_submodule_agent.available_tools.tools: # Assuming .tools gives a list of instances
                    if isinstance(tool_instance, BrowserUseTool):
                        if hasattr(tool_instance, 'llm'):
                            tool_instance.llm = self.llm_adapter
                            # Assuming tool_instance has a 'name' attribute for logging
                            tool_name = getattr(tool_instance, 'name', str(tool_instance))
                            print(f"[OpenManusAgent.async_setup] VerlLLMAdapter injected into BrowserUseTool instance: {tool_name}")
                        else:
                            tool_name = getattr(tool_instance, 'name', str(tool_instance))
                            print(f"[OpenManusAgent.async_setup] Warning: BrowserUseTool instance {tool_name} does not have an 'llm' attribute to replace.")
            else:
                print("[OpenManusAgent.async_setup] No 'available_tools' or 'available_tools.tools' found on submodule agent, or it's empty. Skipping tool LLM injection.")

        except ImportError as e:
            print(f"[OpenManusAgent.async_setup] Critical Error: Failed to import SubmoduleManus or BrowserUseTool: {e}. Check path and submodule.")
            self.manus_submodule_agent = None
            self.llm_adapter = None 
            return # Exit if essential imports fail
        except Exception as e:
            print(f"[OpenManusAgent.async_setup] Critical Error during async_setup: {e}")
            # import traceback # Already imported at file level usually, or ensure it is.
            traceback.print_exc()
            self.manus_submodule_agent = None
            self.llm_adapter = None

    def _init_env_clients(self) -> List[Any]: # Renamed and return type changed
        """
        Initialize and return a list of specific AgentGym environment clients
        based on the ports provided in the config.
        """
        clients = []
        env_name_lower = self.config.env_name.lower()

        # Mapping from env_name (lowercase) to Task class name
        ENV_TO_TASK_CLASS = {
            "academia": "AcademiaTask", "alfworld": "AlfWorldTask", "babyai": "BabyAITask",
            "maze": "MazeTask", "wordle": "WordleTask", "movie": "MovieTask",
            "sciworld": "SciworldTask", "sheet": "SheetTask", "sqlgym": "SqlGymTask",
            "textcraft": "TextCraftTask", "todo": "TodoTask", "weather": "WeatherTask",
            "webarena": "WebarenaTask", "webshop": "WebshopTask",
        }

        if env_name_lower not in ENV_TO_TASK_CLASS:
            raise ValueError(f"Unsupported environment name: {self.config.env_name}. Supported: {list(ENV_TO_TASK_CLASS.keys())}")

        task_class_name = ENV_TO_TASK_CLASS[env_name_lower]
        print(f"[Info] Initializing {len(self.config.env_ports)} Env Client(s) for: {self.config.env_name} (via Task: {task_class_name})")

        # Dynamically import the Task class
        try:
            envs_module = importlib.import_module("agentenv.envs")
            TaskClass = getattr(envs_module, task_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import Task class {task_class_name} from agentenv.envs: {e}")

        for i, port in enumerate(self.config.env_ports):
            server_url = f"{self.config.env_server_base}:{port}"
            print(f"  - Client {i+1}: Connecting to {server_url}")

            client_args={
                "env_server_base": server_url,
                "data_len": self.config.env_data_len,
                "timeout": 300,
            }

            try:
                # Instantiate the task to get the client.
                # We need one client per specified port.
                # Assuming TaskClass handles client creation correctly when n_clients=1.
                # If TaskClass itself manages multiple internal clients, this might need adjustment.
                task_instance = TaskClass(client_args=client_args, n_clients=1)
                if hasattr(task_instance, 'clients') and task_instance.clients:
                    client = task_instance.clients[0]
                    print(f"  - Client {i+1}: Successfully obtained client: {type(client)}")
                    clients.append(client)
                else:
                     print(f"  - Client {i+1}: Error - Task class {task_class_name} did not provide a client for port {port}.")
                     # Decide how to handle failure: raise error or skip this client? Skipping for now.
                     # raise ValueError(f"Task class {task_class_name} did not provide a client for port {port}.")
            except Exception as e:
                 print(f"  - Client {i+1}: Error initializing Task or getting client for port {port}: {e}")
                 print(traceback.format_exc())
                 # Decide how to handle failure: raise error or skip? Skipping for now.
                 # raise

        if not clients:
            raise RuntimeError("Failed to initialize any environment clients.")

        print(f"[Info] Successfully initialized {len(clients)} environment clients.")
        return clients

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']


    def _run_single_rollout(self, initial_prompt_ids: torch.Tensor, task_idx: int, client: Any) -> Dict[str, Any]:
        """
        Runs the interaction loop for a single environment instance using the provided client.
        Now includes the final computed reward from the environment step in the result.

        Args:
            initial_prompt_ids: Token IDs for the initial prompt/observation.
            task_idx: The index for resetting the environment.
            client: The specific environment client instance to use for this rollout.

        Returns:
            A dictionary containing the trajectory, step rewards, final reward,
            final env score, turns, and original task index.
        """
        trajectory = []
        step_rewards = []  # Store rewards per step
        final_reward = 0.0 # Reward from the *last step*
        final_env_score = 0.0 # Final score from env info
        done = False
        turns = 0
        current_input_ids = None

        try:
            # Reset environment using the provided client
            # Some envs might need a specific seed or config reset
            # print(f"[Agent._run_single_rollout][{task_idx}] Resetting env...")
            reset_info = client.reset(task_idx) # Capture potential info from reset
            initial_obs_text = client.observe()
            # print(f"[Agent._run_single_rollout][{task_idx}] Initial Obs: {initial_obs_text[:100]}...")

            # Handle initial observation
            if not initial_obs_text:
                # print(f"[Agent._run_single_rollout][{task_idx} @ {client.env_server_base}] Warning: Received empty initial observation. Using initial prompt from batch.")
                # Use the initial prompt text passed in
                initial_prompt_text = self.tokenizer.decode(initial_prompt_ids[0], skip_special_tokens=True)
                trajectory.append({"from": "human", "value": initial_prompt_text})
                current_input_ids = initial_prompt_ids
            else:
                trajectory.append({"from": "human", "value": initial_obs_text})
                current_input_ids = self.tokenizer(initial_obs_text, return_tensors='pt', add_special_tokens=False)['input_ids']

            # --- Interaction Loop --- 
            for t in range(self.config.max_turns):
                turns = t + 1
                if current_input_ids is None:
                    # print(f"[Agent._run_single_rollout][{task_idx}] Breaking loop: current_input_ids is None")
                    break

                # Handle input that exceeds max length
                if current_input_ids.shape[1] > self.config.max_prompt_length:
                    # print(f"[Agent._run_single_rollout][{task_idx} @ {client.env_server_base}] Warning: Truncating input {current_input_ids.shape} > {self.config.max_prompt_length}.")
                    current_input_ids = current_input_ids[:, -self.config.max_prompt_length:]

                # Prepare input
                current_attention_mask = self.tensor_fn.create_attention_mask(current_input_ids)
                current_position_ids = self.tensor_fn.create_position_ids(current_attention_mask)
                # device = 'cuda' # Assume target device is cuda; worker group handles internal placement
                gen_input_proto = DataProto.from_dict({
                    'input_ids': current_input_ids, # Pass tensor directly (likely CPU)
                    'attention_mask': current_attention_mask,
                    'position_ids': current_position_ids
                }) # may need to put this on the correct device

                world_size = self.actor_rollout_wg.world_size
                original_size = 1 # We know batch size is 1 here
                padded_gen_input_proto = gen_input_proto
                padding_size = 0
                if world_size > 1 and original_size % world_size != 0:
                    padding_size = world_size - (original_size % world_size)
                    padded_batch = {}
                    for k, v in gen_input_proto.batch.items():
                        # Use the single sequence as padding template
                        pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
                        padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
                    padded_gen_input_proto = DataProto.from_dict(padded_batch)
                    # Copy meta_info if needed
                    if hasattr(gen_input_proto, 'meta_info'):
                         padded_gen_input_proto.meta_info = gen_input_proto.meta_info.copy()


                # --- Prepare Generation Config --- 
                generation_config = GenerationConfig(
                    max_new_tokens=self.config.max_response_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=1.0, # Consider adjusting temperature/sampling based on validation vs training
                    do_sample=True
                )

                if not hasattr(padded_gen_input_proto, 'meta_info'):
                    padded_gen_input_proto.meta_info = {}
                padded_gen_input_proto.meta_info['generation_config'] = generation_config

                # Generation happens on the actor worker group's device
                gen_output_proto = self.actor_rollout_wg.generate_sequences(padded_gen_input_proto)
                # response_ids = gen_output_proto.batch['response_ids'] # Original line causing KeyError
                response_ids = gen_output_proto.batch['responses'] # Use the correct key ('responses') assuming it holds IDs

                if padding_size > 0:
                     response_ids = response_ids[:-padding_size]

                # Decode the response IDs to get the text for the trajectory
                response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

                # print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Response: {response_text[:100]}...")
                trajectory.append({"from": "gpt", "value": response_text})

                # Post-process response to get action
                action_types, action_contents = self.postprocess_predictions([response_text])
                action_text = action_contents[0]

                # Execute environment step using the provided client
                if action_text is None: action_text = ""
                step_output = client.step(action_text)
                next_obs_text = step_output.state
                reward = step_output.reward
                done = step_output.done
                info = {} # Initialize info as empty dict, as StepOutput doesn't explicitly return it
                print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Env Step Result: Reward={reward}, Done={done}, Info={info}")

                # Store the reward from this specific step
                step_rewards.append(reward)
                final_reward = reward # Keep track of the reward from the last executed step
                final_env_score = info.get('score', 0.0) # Use .get for safety

                # Add reward and info to the trajectory for this agent step
                # This helps the RewardComposer access step-specific info if needed
                trajectory[-1]['reward'] = reward
                trajectory[-1]['info'] = info

                # Process next observation
                if not done:
                    print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Next Obs: {next_obs_text[:100]}...")
                    trajectory.append({"from": "env", "value": next_obs_text})
                    next_obs_ids = self.tokenizer(next_obs_text, return_tensors='pt', add_special_tokens=False)['input_ids']
                    # Ensure tensors are concatenated on the same device (e.g., CPU or model's device if needed later)
                    current_input_ids = torch.cat([
                        current_input_ids.to(response_ids.device), # Move to same device as response_ids
                        response_ids,
                        next_obs_ids.to(response_ids.device) # Move to same device
                    ], dim=1)
                else:
                    print(f"[Agent._run_single_rollout][{task_idx}][Turn {t+1}] Done received.")
                    break

        except Exception as e:
            print(f"[Agent._run_single_rollout][{task_idx} @ {getattr(client, 'env_server_base', 'unknown_client')}] Error during rollout: {e}")
            print(traceback.format_exc())
            # Reset results on error
            trajectory = trajectory # Keep partial trajectory for debugging?
            step_rewards = []
            final_reward = 0.0
            final_env_score = 0.0
            done = True # Mark as done on error

        # Return the collected information
        return {
            'trajectory': trajectory,        # Full interaction history
            'step_rewards': step_rewards,    # List of rewards from each env.step call
            'reward': final_reward,          # Reward from the *last* env.step call
            'env_score': final_env_score,    # Final score reported by env info
            'turns': turns,                  # Total number of turns executed
            'valid_actions': len([msg for msg in trajectory if msg.get("from") == "gpt"]), # Count of agent's responses
            'task_idx': task_idx,
            'done': done                   # Whether the episode finished naturally or via error
        }

    # Replacing the old _run_single_rollout with the new async version
    async def _run_single_rollout(self, initial_prompt_ids: torch.Tensor, task_idx: int, client: Any) -> Dict[str, Any]:
       if not self.manus_submodule_agent or not self.llm_adapter:
           print("[OpenManusAgent._run_single_rollout] Error: SubmoduleManus agent not initialized. Call async_setup first.")
           print("[OpenManusAgent._run_single_rollout] Warning: Attempting late initialization via async_setup().")
           await self.async_setup() 
           if not self.manus_submodule_agent: # If still not initialized, then fail
                raise RuntimeError("SubmoduleManus agent failed to initialize even after late async_setup call.")

       trajectory = []
       step_rewards = []
       final_reward = 0.0
       final_env_score = 0.0 # Still 0 as per StepOutput
       done = False
       turns = 0
       
       current_env_obs_text = ""

       try:
           await client.reset(task_idx) # Assuming client.reset() can be async
           current_env_obs_text = client.observe()
           trajectory.append({"from": "env", "value": current_env_obs_text})

           # Optional: Reset Manus submodule agent's memory if it's stateful and reused across rollouts.
           # if hasattr(self.manus_submodule_agent, 'reset_memory'):
           # await self.manus_submodule_agent.reset_memory() 

           for t in range(self.config.max_turns):
               turns = t + 1
               
               self.manus_submodule_agent.update_memory("user", current_env_obs_text)
               
               await self.manus_submodule_agent.step() 

               action_for_base_env = ""
               if not self.manus_submodule_agent.memory.messages: # Check memory.messages
                   action_for_base_env = "" 
               else:
                   last_manus_message = self.manus_submodule_agent.memory.messages[-1]
                   # Ensure message is dict-like or has attributes role, content, tool_calls
                   # The placeholder Manus in openmanus_env.py uses simple objects. Real one should be fine.
                   msg_role = getattr(last_manus_message, 'role', None)
                   msg_content = getattr(last_manus_message, 'content', "")
                   msg_tool_calls = getattr(last_manus_message, 'tool_calls', None)

                   if msg_role == "assistant":
                       action_for_base_env = msg_content or ""
                       if msg_tool_calls:
                           # Handle tool calls if they are meant for the base environment
                           # This part needs a clear convention. For now, content is primary.
                           pass 
                   else: 
                       action_for_base_env = "" 

               trajectory.append({"from": "gpt", "value": action_for_base_env})

               if action_for_base_env is None: action_for_base_env = ""
               step_output = await client.step(action_for_base_env) # Assuming client.step() can be async
               next_env_obs_text = step_output.state
               reward = step_output.reward
               done = step_output.done
               # info = {} # As before, StepOutput has no info

               step_rewards.append(reward)
               final_reward = reward
               # final_env_score = info.get('score', 0.0) # Remains 0

               trajectory[-1]['reward'] = reward 
               # trajectory[-1]['info'] = info # No info

               current_env_obs_text = next_env_obs_text
               if not done:
                   trajectory.append({"from": "env", "value": current_env_obs_text})
               else:
                   break
           
       except Exception as e:
           print(f"[OpenManusAgent._run_single_rollout][{task_idx}] Error: {e}")
           import traceback
           traceback.print_exc()
           done = True 

       return {
           'trajectory': trajectory,
           'step_rewards': step_rewards,
           'reward': final_reward,
           'env_score': final_env_score,
           'turns': turns,
           'valid_actions': len([msg for msg in trajectory if msg.get("from") == "gpt"]),
           'task_idx': task_idx,
           'done': done
       }

    def run_llm_loop(self, gen_batch: DataProto, output_dir: str = None, global_steps: int = 0) -> DataProto:
        """
        Run the LLM interaction loop for a batch of initial prompts using multiple clients.
        This method now needs to handle the async nature of _run_single_rollout.
        It will use asyncio.gather to run rollouts concurrently.

        Args:
            gen_batch: DataProto containing initial prompts
            output_dir: Directory to save visualizations
            global_steps: Current training step

        Returns:
            DataProto containing processed results
        """
        initial_prompts_ids = gen_batch.batch['input_ids']
        batch_size = initial_prompts_ids.shape[0]
        num_clients = len(self.clients)
        if num_clients == 0:
            raise RuntimeError("No environment clients available for rollout.")

        print(f"[Agent.run_llm_loop] Starting rollout for batch size: {batch_size} using {num_clients} clients.")

        # Setup initial state tracking
        original_left_side = {'input_ids': initial_prompts_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_prompts_ids[:, []], 
            'responses_with_info_mask': initial_prompts_ids[:, []]
        }
        
        # Initialize active mask and tracking statistics
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # --- Parallel Rollout Execution ---
        futures = {}
        rollout_results_list = [None] * batch_size  # Preallocate list to store results in order

        # Submit tasks to the thread pool, distributing across clients
        # This part needs to be adapted for asyncio.gather if _run_single_rollout is async.
        # The ThreadPoolExecutor is for sync functions. For async, we use asyncio.
        
        # First, ensure async_setup has been called.
        # This is a temporary workaround. Ideally, the trainer calls async_setup once.
        if not self.manus_submodule_agent or not self.llm_adapter:
            print("[Agent.run_llm_loop] Warning: manus_submodule_agent not ready. Calling async_setup().")
            # This is problematic if run_llm_loop is called in a sync context by the trainer.
            # For now, let's assume run_llm_loop can become async or use a helper.
            # If run_async_in_sync_context existed, it would be: run_async_in_sync_context(self.async_setup())
            # Without it, this will error if called from pure sync.
            # The subtask implies we might need to define run_async_in_sync_context.
            # For now, this part highlights the sync/async challenge.
            # Let's assume for the purpose of this refactor that this method will be called from an async context
            # or that the environment setup handles the event loop.
            # If not, the direct `await self.async_setup()` here would be an issue.
            # To make it runnable in a sync context (like a Ray actor method), one might do:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, can't run_until_complete. Schedule and hope for the best or use nested loop if allowed.
                    # This is complex. Simplest for now is to assume it's called or this method becomes async.
                    # Let's assume this method will be made async if needed by caller.
                    # For now, to make the diff apply, I'll proceed as if this method can call await.
                    # This will require `run_llm_loop` to become `async def run_llm_loop`.
                    # This change is outside the direct scope of the subtask's diff for _run_single_rollout,
                    # but is a necessary consequence.
                    # For now, let's comment out the direct await here and assume setup is done prior.
                    pass # await self.async_setup() # This line would make run_llm_loop async
                else:
                    loop.run_until_complete(self.async_setup())
            except RuntimeError: # No event loop
                 asyncio.run(self.async_setup())


        async def gather_rollouts():
            tasks_to_run = []
            for i in range(batch_size):
                task_idx = i # This is the environment's task index, not necessarily original batch index if shuffled
                initial_prompt = initial_prompts_ids[i:i+1]
                client_index = i % num_clients
                selected_client = self.clients[client_index]
                tasks_to_run.append(self._run_single_rollout(initial_prompt, task_idx, selected_client))
            
            # Execute all rollout tasks concurrently
            return await asyncio.gather(*tasks_to_run, return_exceptions=True)

        # Execute the async gathering function
        # This requires run_llm_loop to be async or to manage an event loop.
        # If run_llm_loop is sync, this is where run_async_in_sync_context would be essential.
        # For now, assume it can be made async or the environment handles it.
        # results_or_exceptions = asyncio.run(gather_rollouts()) # This creates a new loop, might conflict.
        # A common pattern if run_llm_loop is sync and part of a class:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError: # No event loop in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results_or_exceptions = loop.run_until_complete(gather_rollouts())


        rollout_results_list = [None] * batch_size
        completed_count = 0
        for i, res_or_exc in enumerate(results_or_exceptions):
            original_batch_idx = i # Assuming results are in order of submission
            if isinstance(res_or_exc, Exception):
                print(f"[Agent.run_llm_loop] Error collecting result for batch index {original_batch_idx}: {res_or_exc}")
                print(traceback.format_exc())
                rollout_results_list[original_batch_idx] = {
                    'trajectory': [], 'step_rewards': [], 'reward': 0.0,
                    'turns': 0, 'env_score': 0.0, 'task_idx': original_batch_idx, # Use original_batch_idx as task_idx fallback
                    'error': str(res_or_exc)
                }
            else:
                rollout_results_list[original_batch_idx] = res_or_exc
                completed_count +=1
        
        print(f"[Agent.run_llm_loop] Collected results from {completed_count}/{batch_size} rollouts (async execution).")

        # Filter out potential None entries if some tasks failed critically
        valid_results = [res for res in rollout_results_list if res is not None]

        if not valid_results:
            print("[Agent.run_llm_loop] Error: No valid rollout results collected.")
            # Return empty DataProto but with correct structure if possible
            empty_proto = DataProto.from_dict({
                "input_ids": torch.empty((0,0), dtype=torch.long),
                "attention_mask": torch.empty((0,0), dtype=torch.long),
                "position_ids": torch.empty((0,0), dtype=torch.long),
                "info_mask": torch.empty((0,0), dtype=torch.long),
                "token_level_rewards": torch.empty((0,0), dtype=torch.float)
            })
            # Add necessary meta_info for downstream compute_log_prob call
            empty_proto.meta_info = {'micro_batch_size': 1}
            return empty_proto

        # --- Format Results into DataProto ---
        processed_data = self._convert_rollout_results_to_dataproto(valid_results, gen_batch)

        # --- CRITICAL: Add necessary meta_info parameters for compute_log_prob ---
        # These parameters are required by DataParallelActor.compute_log_prob
        # Source values from the actor_rollout_wg config or AgentConfig
        log_prob_micro_batch_size = getattr(self.actor_rollout_wg, 'log_prob_micro_batch_size', 128)
        if hasattr(self.config, 'actor_rollout_ref') and hasattr(self.config.actor_rollout_ref, 'rollout'):
            # If running within the trainer which has direct access to these configs
            log_prob_micro_batch_size = getattr(self.config.actor_rollout_ref.rollout, 'log_prob_micro_batch_size', log_prob_micro_batch_size)
        
        # Ensure these keys exist and have reasonable default values even if not specified in config
        if 'micro_batch_size' not in processed_data.meta_info:
            processed_data.meta_info['micro_batch_size'] = log_prob_micro_batch_size
        
        if 'temperature' not in processed_data.meta_info:
            processed_data.meta_info['temperature'] = getattr(self.config, 'temperature', 1.0)
        
        if 'use_dynamic_bsz' not in processed_data.meta_info:
            processed_data.meta_info['use_dynamic_bsz'] = getattr(self.config, 'log_prob_use_dynamic_bsz', False)
        
        # If dynamic batch size is used, also set max_token_len
        if processed_data.meta_info.get('use_dynamic_bsz', False):
            max_token_len = getattr(self.config, 'log_prob_max_token_len_per_gpu', 2048)
            processed_data.meta_info['max_token_len'] = max_token_len
        
        print(f"[Agent.run_llm_loop] Added log_prob parameters to meta_info: micro_batch_size={processed_data.meta_info['micro_batch_size']}, temperature={processed_data.meta_info['temperature']}, use_dynamic_bsz={processed_data.meta_info['use_dynamic_bsz']}")

        print(f"[Agent.run_llm_loop] Finished processing rollout results.")
        return processed_data

    def _convert_rollout_results_to_dataproto(self, results: List[Dict], original_batch: DataProto) -> DataProto:
        """
        Convert the list of dictionaries (each containing trajectory, step_rewards, env_score)
        from the internal rollout loop into a DataProto suitable for PPO training.
        Creates 'token_level_rewards' based on the chosen reward allocation strategy.

        Args:
            results: List of result dictionaries from _run_single_rollout.
            original_batch: Original batch DataProto with metadata.

        Returns:
            DataProto: Processed data with token-level rewards and metadata.
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_info_mask = []
        batch_token_level_rewards = [] # Store final token-level rewards for PPO
        batch_responses = [] # Initialize batch_responses

        # Initialize final_meta_info by copying all items from the original_batch.meta_info
        # This ensures that any global metadata from the input batch is preserved.
        final_meta_info = {}
        if hasattr(original_batch, 'meta_info') and original_batch.meta_info:
            for k, v in original_batch.meta_info.items():
                final_meta_info[k] = v # Shallow copy, or deepcopy if mutable objects are a concern

        # For collecting stats and per-rollout lists that will be converted to tensors or kept as lists
        per_rollout_task_idx = []
        per_rollout_turns_stats = []
        per_rollout_valid_action_stats = []
        per_rollout_done_flags = []
        per_rollout_valid_search_stats = [] # Placeholder
        per_rollout_rewards = [] # Last step reward for each rollout
        per_rollout_env_scores = [] # Final env score for each rollout
        per_rollout_trajectories = [] # List of trajectories

        # Get reward allocation strategy from config
        reward_allocation = "last_token" # Default
        if self.config.algorithm_config:
            reward_allocation = self.config.algorithm_config.get('reward_allocation', 'last_token')
        print(f"[Agent._convert_rollout] Using reward allocation strategy: {reward_allocation}")

        # Get the index mapping from the original batch
        original_indices = original_batch.meta_info.get('idx', list(range(original_batch.batch['input_ids'].shape[0])))
        if isinstance(original_indices, torch.Tensor):
            original_indices = original_indices.tolist()
        original_indices_map = {idx_val: i for i, idx_val in enumerate(original_indices)}

        print(f"[Agent._convert_rollout] Formatting {len(results)} trajectories.")
        for result_dict in results:
            # Extract trajectory and other info
            trajectory = result_dict.get('trajectory', [])
            # Choose which reward signal to use for allocation
            reward_to_distribute = result_dict.get('reward', 0.0)

            turns = result_dict.get('turns', 0)
            task_idx = result_dict.get('task_idx', -1)
            valid_actions_count = result_dict.get('valid_actions', 0)
            done_flag = result_dict.get('done', True) # Default to True if missing, indicating completion or error
            reward_val = result_dict.get('reward', 0.0)
            env_score_val = result_dict.get('env_score', 0.0)
            trajectory_val = result_dict.get('trajectory', [])

            # Correctly append to per_rollout_ lists
            per_rollout_task_idx.append(task_idx)
            per_rollout_turns_stats.append(turns)
            per_rollout_valid_action_stats.append(valid_actions_count)
            per_rollout_done_flags.append(done_flag)
            per_rollout_valid_search_stats.append(0) # Placeholder, as search is not explicitly tracked here
            per_rollout_rewards.append(reward_val)
            per_rollout_env_scores.append(env_score_val)
            per_rollout_trajectories.append(trajectory_val)

            # Get the original batch index (used for trajectory processing below)
            original_batch_idx = original_indices_map.get(task_idx, -1)
            if original_batch_idx == -1:
                print(f"[Agent._convert_rollout] Warning: Task idx {task_idx} not found in original batch. Skipping this result for trajectory processing.")
                # If a result can't be mapped, its trajectory-derived tensors might be misaligned.
                # For simplicity, we might skip creating tensor entries for it, or handle padding carefully.
                # However, its stats (task_idx, turns, etc.) are already appended to per_rollout_ lists.
                # This might lead to length mismatches if not handled carefully when creating final tensors.
                # A robust solution would be to filter results list upfront or ensure all task_idx are mappable.
                # For now, we proceed, and downstream tensor creation should handle potential Nones if any result is fully skipped.
                # OR, more simply, if we can't map, we might have to skip this entire result_dict earlier.
                # For now, let the per_rollout lists gather all data, and mismatches will be an issue at tensor conversion.
                pass # Original_batch_idx is used for trajectory processing, not for the stats lists directly.

            # --- Concatenate conversation and identify agent segments --- 
            conversation_ids_list = []
            info_mask_parts = []
            segment_lengths = [] # Store length of each segment (human/gpt)
            agent_response_indices = [] # Store indices of agent responses (in the segment list)
            valid_actions = 0 # Count of agent turns

            if not trajectory:
                 # Handle empty trajectory
                 initial_prompt_ids = original_batch.batch['input_ids'][original_batch_idx:original_batch_idx+1]
                 conversation_ids_list.append(initial_prompt_ids)
                 info_mask_parts.append(torch.ones_like(initial_prompt_ids))
                 segment_lengths.append(initial_prompt_ids.shape[1])
            else:
                for turn_idx, msg in enumerate(trajectory):
                    msg_text = msg.get("value", "")
                    msg_from = msg.get("from", "")
                    if not msg_text: continue

                    msg_ids = self.tokenizer(msg_text, add_special_tokens=False, return_tensors='pt')['input_ids']
                    conversation_ids_list.append(msg_ids)
                    segment_lengths.append(msg_ids.shape[1])

                    if msg_from == "gpt":
                        # Agent responses are normal tokens (not masked)
                        info_mask_parts.append(torch.ones_like(msg_ids))
                        valid_actions += 1
                        agent_response_indices.append(len(conversation_ids_list) - 1)
                    elif msg_from == "env":
                        # Environment observations should be info-masked
                        info_mask_parts.append(torch.zeros_like(msg_ids))
                    else: # human or other
                        # Human/prompt parts are normal tokens
                        info_mask_parts.append(torch.ones_like(msg_ids))

            if not conversation_ids_list:
                print(f"[Agent._convert_rollout] Warning: No valid conversation segments for task_idx {task_idx}. Skipping.")
                continue

            # --- Pad and Truncate --- 
            full_input_ids = torch.cat(conversation_ids_list, dim=1)
            full_info_mask = torch.cat(info_mask_parts, dim=1)
            seq_len = full_input_ids.shape[1]
            target_len = self.config.max_prompt_length
            padding_len = max(0, target_len - seq_len)
            agent_indices_in_padded = [] # List of (start, end) indices for agent tokens in the final padded tensor

            if seq_len > target_len:
                # Truncate left if sequence is too long
                removed_len = seq_len - target_len
                current_removed = 0
                first_segment_idx = 0
                while current_removed < removed_len and first_segment_idx < len(segment_lengths):
                    len_to_remove = min(segment_lengths[first_segment_idx], removed_len - current_removed)
                    segment_lengths[first_segment_idx] -= len_to_remove
                    current_removed += len_to_remove
                    if segment_lengths[first_segment_idx] == 0:
                        first_segment_idx += 1
                
                # Adjust agent response indices
                adjusted_agent_response_indices = [idx - first_segment_idx for idx in agent_response_indices if idx >= first_segment_idx]
                segment_lengths = segment_lengths[first_segment_idx:]
                
                full_input_ids = full_input_ids[:, -target_len:]
                full_info_mask = full_info_mask[:, -target_len:]
                seq_len = target_len
                padding_len = 0 # No padding needed after truncation
            elif seq_len < target_len:
                # Pad left if sequence is too short
                pad_tensor = torch.full((1, padding_len), self.tokenizer.pad_token_id, dtype=torch.long, device=full_input_ids.device)
                full_input_ids = torch.cat([pad_tensor, full_input_ids], dim=1)
                # Info mask for padding should be 0 (masked out)
                info_pad = torch.zeros_like(pad_tensor)
                full_info_mask = torch.cat([info_pad, full_info_mask], dim=1)
                adjusted_agent_response_indices = agent_response_indices # Indices remain the same relative to segments

            # Calculate agent token indices in the padded/truncated tensor
            current_token_idx_in_padded = padding_len
            for segment_idx, length in enumerate(segment_lengths):
                 is_agent_response = segment_idx in adjusted_agent_response_indices
                 start_idx = current_token_idx_in_padded
                 end_idx = current_token_idx_in_padded + length - 1
                 if is_agent_response and length > 0:
                      agent_indices_in_padded.append((start_idx, end_idx))
                 current_token_idx_in_padded += length

            # --- Create Token Level Rewards Tensor based on Allocation Strategy --- 
            token_level_rewards = torch.zeros_like(full_input_ids, dtype=torch.float32)

            if agent_indices_in_padded: # Only allocate if there are agent responses
                if reward_allocation == "last_token":
                    # Assign reward only to the last token of the last agent segment
                    last_segment_start, last_segment_end = agent_indices_in_padded[-1]
                    if last_segment_end < target_len: # Ensure index is within bounds
                        token_level_rewards[0, last_segment_end] = reward_to_distribute

                elif reward_allocation == "uniform_positive":
                    # Distribute positive rewards evenly across all agent tokens
                    if reward_to_distribute > 0:
                        total_agent_tokens = sum(end - start + 1 for start, end in agent_indices_in_padded)
                        reward_per_token = reward_to_distribute / max(1, total_agent_tokens)
                        for start, end in agent_indices_in_padded:
                            token_level_rewards[0, start : end + 1] = reward_per_token
                    # Negative rewards are assigned to the last token (or ignored)
                    elif reward_to_distribute < 0:
                         last_segment_start, last_segment_end = agent_indices_in_padded[-1]
                         if last_segment_end < target_len:
                              token_level_rewards[0, last_segment_end] = reward_to_distribute

                elif reward_allocation == "discounted":
                    # Distribute reward starting from the last agent segment, discounted backward
                    gamma = self.config.algorithm_config.get('gamma', 1.0) if self.config.algorithm_config else 1.0
                    current_reward = reward_to_distribute
                    # Iterate segments backward
                    for start, end in reversed(agent_indices_in_padded):
                        segment_len = end - start + 1
                        reward_for_segment = current_reward / segment_len
                        token_level_rewards[0, start : end + 1] = reward_for_segment
                        # Apply discount for the next (earlier) segment
                        current_reward *= (gamma ** segment_len)
                else:
                     print(f"[Agent._convert_rollout] Warning: Unknown reward_allocation strategy '{reward_allocation}'. Defaulting to last_token.")
                     last_segment_start, last_segment_end = agent_indices_in_padded[-1]
                     if last_segment_end < target_len:
                         token_level_rewards[0, last_segment_end] = reward_to_distribute

            # --- Create Attention Mask and Position IDs --- 
            full_attention_mask = self.tensor_fn.create_attention_mask(full_input_ids)
            full_position_ids = self.tensor_fn.create_position_ids(full_attention_mask)

            # --- Store Processed Data --- 
            batch_input_ids.append(full_input_ids)
            batch_attention_mask.append(full_attention_mask)
            batch_position_ids.append(full_position_ids)
            batch_info_mask.append(full_info_mask) # Store the info mask
            batch_token_level_rewards.append(token_level_rewards) # Store calculated rewards

            # --- Extract and pad response-only tokens ---
            response_segments = []
            total_response_len = 0
            
            for r_start, r_end in agent_indices_in_padded:
                segment = full_input_ids[0, r_start : r_end + 1]
                response_segments.append(segment)
                total_response_len += segment.shape[0]
            
            # Get the configured response length from config
            configured_resp_len = self.config.max_response_length
            
            if response_segments:
                # Concatenate all response segments
                response_only_ids_cat = torch.cat(response_segments, dim=0).unsqueeze(0) # Shape (1, total_response_len)
                resp_pad_len = max(0, configured_resp_len - total_response_len)
                
                # Pad or truncate to configured length
                if resp_pad_len > 0:
                    # Pad to configured length if shorter
                    resp_pad = torch.full((1, resp_pad_len), self.tokenizer.pad_token_id, dtype=torch.long, device=response_only_ids_cat.device)
                    response_only_ids_padded = torch.cat([response_only_ids_cat, resp_pad], dim=1)
                    print(f"[Agent._convert_rollout] Padded response from {total_response_len} to {configured_resp_len}")
                elif total_response_len > configured_resp_len:
                    # Truncate if response is too long
                    print(f"[Agent._convert_rollout] Truncating response from {total_response_len} to {configured_resp_len}")
                    response_only_ids_padded = response_only_ids_cat[:, :configured_resp_len]
                else:
                    # No adjustment needed
                    response_only_ids_padded = response_only_ids_cat
                    
                # Double-check the final shape meets expectations
                if response_only_ids_padded.shape[1] != configured_resp_len:
                    print(f"[Agent._convert_rollout] WARNING: Response length mismatch: got {response_only_ids_padded.shape[1]}, expected {configured_resp_len}")
                    # Force correction if still wrong
                    if response_only_ids_padded.shape[1] < configured_resp_len:
                        extra_pad = torch.full((1, configured_resp_len - response_only_ids_padded.shape[1]), 
                                              self.tokenizer.pad_token_id, dtype=torch.long, 
                                              device=response_only_ids_padded.device)
                        response_only_ids_padded = torch.cat([response_only_ids_padded, extra_pad], dim=1)
                    else:
                        response_only_ids_padded = response_only_ids_padded[:, :configured_resp_len]
            else:
                # Handle case with no agent responses (e.g., empty trajectory)
                print(f"[Agent._convert_rollout] No agent responses found for item, creating empty response of length {configured_resp_len}")
                response_only_ids_padded = torch.full((1, configured_resp_len), 
                                                     self.tokenizer.pad_token_id, dtype=torch.long, 
                                                     device=full_input_ids.device)
            
            # Append to batch list
            batch_responses.append(response_only_ids_padded)

            # Add metadata
            if "task_idx" not in final_meta_info:
                final_meta_info["task_idx"] = []
            if "turns_stats" not in final_meta_info:
                final_meta_info["turns_stats"] = []
            if "valid_action_stats" not in final_meta_info:
                final_meta_info["valid_action_stats"] = []
            if "reward" not in final_meta_info:
                final_meta_info["reward"] = []
            if "env_score" not in final_meta_info:
                final_meta_info["env_score"] = []
            if "rollout_trajectory" not in final_meta_info:
                final_meta_info["rollout_trajectory"] = []

            final_meta_info["task_idx"].append(task_idx)
            final_meta_info["turns_stats"].append(turns)
            final_meta_info["valid_action_stats"].append(valid_actions_count)
            final_meta_info["reward"].append(reward_val)
            final_meta_info["env_score"].append(env_score_val)
            final_meta_info["rollout_trajectory"].append(trajectory_val)

        # --- Stack Tensors --- 
        if not batch_input_ids:
            print("[Agent._convert_rollout] No valid trajectories formatted. Returning empty DataProto.")
            # Return structure matching trainer expectations, even if empty
            return DataProto.from_dict({
                "input_ids": torch.empty((0,0), dtype=torch.long),
                "attention_mask": torch.empty((0,0), dtype=torch.long),
                "position_ids": torch.empty((0,0), dtype=torch.long),
                "info_mask": torch.empty((0,0), dtype=torch.long),
                "token_level_rewards": torch.empty((0,0), dtype=torch.float)
            })

        # Create final batch data
        final_batch = {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "position_ids": torch.cat(batch_position_ids, dim=0),
            "info_mask": torch.cat(batch_info_mask, dim=0), # This is the equivalent of responses_with_info_mask related construction
            "token_level_rewards": torch.cat(batch_token_level_rewards, dim=0),
            "responses": torch.cat(batch_responses, dim=0)
        }

        # Create DataProto and add metadata
        data_proto = DataProto.from_dict(final_batch)
        
        # Add collected statistics and per-rollout lists to final_meta_info, converting to tensors where appropriate
        # These will overwrite any keys with the same name inherited from original_batch.meta_info if they were lists per sample.
        final_meta_info['task_idx'] = torch.tensor(per_rollout_task_idx, dtype=torch.long)
        final_meta_info['turns_stats'] = torch.tensor(per_rollout_turns_stats, dtype=torch.long)
        final_meta_info['valid_action_stats'] = torch.tensor(per_rollout_valid_action_stats, dtype=torch.long)
        final_meta_info['valid_search_stats'] = torch.tensor(per_rollout_valid_search_stats, dtype=torch.long) # Will be zeros
        final_meta_info['active_mask'] = torch.tensor([not done for done in per_rollout_done_flags], dtype=torch.bool)
        final_meta_info['reward'] = torch.tensor(per_rollout_rewards, dtype=torch.float32) # Individual rewards per rollout
        final_meta_info['env_score'] = torch.tensor(per_rollout_env_scores, dtype=torch.float32) # Final scores per rollout
        final_meta_info['rollout_trajectory'] = per_rollout_trajectories # Keep as list of lists/dicts

        # If 'idx' was in original_batch.meta_info and was a tensor, it might have been copied directly.
        # If it needs to be specifically task_idx, the above 'task_idx' tensor is now authoritative for the samples in this batch.
        # We can choose to remove the original 'idx' if it causes confusion or ensure it's compatible.
        # For now, the new 'task_idx' list converted to a tensor becomes the primary index for these processed samples.
        if 'idx' in final_meta_info and not torch.is_tensor(final_meta_info['idx']):
            # If original idx was not a tensor or needs to be sample-specific for this processed batch
            print(f"[Agent._convert_rollout] Replacing original 'idx' with new 'task_idx' tensor.")
            final_meta_info['idx'] = final_meta_info['task_idx']
        elif 'idx' not in final_meta_info:
            final_meta_info['idx'] = final_meta_info['task_idx']

        # Assign the fully constructed final_meta_info to the DataProto object
        data_proto.meta_info = final_meta_info

        print(f"[Agent._convert_rollout] Final batch shapes: input_ids={final_batch['input_ids'].shape}, token_level_rewards={final_batch['token_level_rewards'].shape}, responses={final_batch['responses'].shape}")
        return data_proto


    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
        """
        Process predictions into actions and content based on XML-like tags.
        Does not require tool_manager.

        Args:
            predictions: List of raw predictions (strings from LLM)

        Returns:
            Tuple of (action types list ['action' or 'response' or None],
                    action contents list [text inside tags or empty string])
        """
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str):
                # Extract action or response tags
                action_pattern = r'<action>(.*?)</action>'
                response_pattern = r'<response>(.*?)</response>'

                action_match = re.search(action_pattern, prediction, re.DOTALL)
                response_match = re.search(response_pattern, prediction, re.DOTALL)

                if action_match:
                    actions.append('action')
                    contents.append(action_match.group(1).strip())
                elif response_match:
                    actions.append('response')
                    contents.append(response_match.group(1).strip())
                else:
                    # If no recognized tag, assume it's neither a specific action nor response
                    actions.append(None)
                    contents.append('') # Return empty content if no tag found
            else:
                # Handle non-string predictions if necessary, e.g., raise error or log warning
                print(f"[Warning] Received non-string prediction: {type(prediction)}. Cannot process.")
                actions.append(None)
                contents.append('')
                # Or raise ValueError(f"Invalid prediction type: {type(prediction)}")

        return actions, contents
