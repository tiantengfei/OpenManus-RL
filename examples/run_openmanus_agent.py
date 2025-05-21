import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any
import sys
import os
import importlib # For patching

# Adjust path to import OpenManusAgent and other necessary components
# This assumes 'examples/' is at the root of the project, alongside 'openmanus_rl/'
# If openmanus_rl is not directly in PYTHONPATH, this might need adjustment
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from openmanus_rl.llm_agent.openmanus import OpenManusAgent, AgentConfig
# Attempt to import DataProto, provide a mock if verl is not installed
try:
    from verl import DataProto
except ImportError:
    print("Warning: 'verl' not found. Using a mock DataProto.")
    @dataclass
    class MockDataProto:
        batch: Dict[str, Any] = field(default_factory=dict)
        meta_info: Dict[str, Any] = field(default_factory=dict)

        @classmethod
        def from_dict(cls, data_dict: Dict[str, Any]):
            # Simplified: assumes data_dict might contain 'batch' and 'meta_info' or just batch data
            if 'batch' in data_dict or 'meta_info' in data_dict:
                 return cls(batch=data_dict.get('batch',{}), meta_info=data_dict.get('meta_info',{}))
            # If just batch data is passed
            return cls(batch=data_dict)

        def to_dict(self) -> Dict[str, Any]:
            return {'batch': self.batch, 'meta_info': self.meta_info}

    DataProto = MockDataProto


from omegaconf import DictConfig # For AgentConfig.algorithm_config

# --- Mock Components ---

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab = {"<pad>": 0, "<eos>": 1, "hello": 2, "world": 3, "get": 4, "info": 5, ".": 6, "webshop": 7, "search": 8, "item": 9, "<action>": 10, "</action>": 11}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.current_id = len(self.vocab)

    def __call__(self, text_list: List[str], return_tensors='pt', padding='longest', add_special_tokens=False):
        # Very basic tokenizer: splits by space, handles known tags, assigns new IDs for unknown words
        all_token_ids = []
        max_len = 0
        for text in text_list:
            # Simple handling for tags like <action>
            import re
            parts = re.split(r'(<action>|</action>)', text)
            tokens = []
            for part in parts:
                if not part: continue
                if part.startswith("<") and part.endswith(">"): # It's a known tag or needs to be
                    if part not in self.vocab: # Should not happen for <action>, </action>
                        self.vocab[part] = self.current_id
                        self.reverse_vocab[self.current_id] = part
                        self.current_id +=1
                    tokens.append(part)
                else:
                    tokens.extend(part.lower().split())
            
            token_ids = []
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = self.current_id
                    self.reverse_vocab[self.current_id] = token
                    self.current_id += 1
                token_ids.append(self.vocab[token])
            all_token_ids.append(token_ids)
            if len(token_ids) > max_len:
                max_len = len(token_ids)

        # Pad
        padded_ids = []
        attention_masks = []
        for token_ids in all_token_ids:
            pad_len = max_len - len(token_ids)
            padded_ids.append(token_ids + [self.pad_token_id] * pad_len)
            attention_masks.append([1] * len(token_ids) + [0] * pad_len)

        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor(padded_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
        return {'input_ids': padded_ids, 'attention_mask': attention_masks}

    def decode(self, token_ids_tensor: torch.Tensor, skip_special_tokens=True):
        if token_ids_tensor.ndim == 0: # Handle single ID
             token_ids_tensor = token_ids_tensor.unsqueeze(0)
        if token_ids_tensor.ndim == 1: # Handle 1D list of IDs
            token_ids_tensor = token_ids_tensor.unsqueeze(0)
        
        decoded_texts = []
        for seq_ids in token_ids_tensor:
            tokens = []
            for token_id_val in seq_ids.tolist(): # Use token_id_val to avoid conflict
                if skip_special_tokens and token_id_val in [self.pad_token_id, self.eos_token_id]:
                    continue
                tokens.append(self.reverse_vocab.get(token_id_val, "<unk>"))
            # Reconstruct phrases like "<action> search item </action>"
            decoded_texts.append(" ".join(tokens).replace(" <action> ", "<action>").replace(" </action> ", "</action>"))
        return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts


class MockActorRolloutWG:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.world_size = 1
        # Mock attributes that might be accessed from actor_rollout_wg.rollout config
        self.rollout = DictConfig({ # Using DictConfig for compatibility
            'log_prob_micro_batch_size': 1,
            # Add other attributes if agent accesses them, e.g.
            # 'temperature': 1.0,
            # 'log_prob_use_dynamic_bsz': False,
        })


    def generate_sequences(self, data_proto: DataProto) -> DataProto:
        input_ids = data_proto.batch['input_ids']
        # Simple mock response: "action: <action>search item</action>"
        mock_response_text = "<action>search item</action>" 
        # Tokenize the mock response. Ensure it's a list for the tokenizer.
        response_ids = self.tokenizer([mock_response_text], return_tensors='pt')['input_ids']
        
        # Ensure response_ids is 2D [batch_size, seq_len]
        if input_ids.shape[0] > 1 and response_ids.shape[0] == 1:
            response_ids = response_ids.repeat(input_ids.shape[0], 1)

        output_batch = {'responses': response_ids} # OpenManusAgent expects 'responses' key
        return DataProto.from_dict(output_batch)

class MockEnvClient: # Used by OpenManusAgent._init_env_clients
    def __init__(self, server_url, data_len, timeout):
        self.server_url = server_url
        self.task_idx_counter = 0 # Simulates current task state or sub-step
        self.current_task_main_idx = 0 # Simulates the overall task_idx from reset
        print(f"MockEnvClient initialized for {server_url}")

    async def reset(self, idx: int): # Made async to match BaseEnvClient
        self.current_task_main_idx = idx
        self.task_idx_counter = 0 # Reset sub-state
        print(f"MockEnvClient [{self.server_url}]: Reset to task {idx}")
        # Return None or some minimal info if expected by OpenManusAgent's use of reset_info
        return None 

    def observe(self) -> str: # Kept sync as per BaseEnvClient
        obs = f"This is the initial observation for task {self.current_task_main_idx} (step {self.task_idx_counter}) from {self.server_url}."
        print(f"MockEnvClient [{self.server_url}]: Observe: {obs}")
        return obs

    async def step(self, action: str): # Made async
        # Delayed import to avoid circular or early import issues if StepOutput itself has deps
        from openmanus_rl.agentgym.agentenv.agentenv.controller.env import StepOutput 
        print(f"MockEnvClient [{self.server_url}]: Step with action: '{action}' on task {self.current_task_main_idx}")
        self.task_idx_counter +=1 # Simulate state change within the task
        
        next_state = ""
        reward = 0.0
        done = False

        if "search item" in action:
            next_state = f"Found item: SuperWidget. Price $10. (Task {self.current_task_main_idx}, Step {self.task_idx_counter})"
            reward = 0.5
            done = False
        elif self.task_idx_counter > 2 : # End after a few steps for testing
            next_state = f"Task {self.current_task_main_idx} completed at step {self.task_idx_counter}."
            reward = 1.0
            done = True
        else:
            next_state = f"Action '{action}' received. New observation for task {self.current_task_main_idx}, step {self.task_idx_counter}."
            reward = 0.1
            done = False
        
        print(f"MockEnvClient [{self.server_url}]: Next State: {next_state}, Reward: {reward}, Done: {done}")
        return StepOutput(state=next_state, reward=reward, done=done)

# --- Main Example Script ---
def main(): # Changed from async def main() as run_llm_loop is blocking
    print("--- Starting OpenManusAgent Example Script ---")

    # 1. Initialize Mock Components
    mock_tokenizer = MockTokenizer()
    mock_actor_rollout_wg = MockActorRolloutWG(mock_tokenizer)

    # 2. Create AgentConfig & Patch importlib
    original_import_module = importlib.import_module
    
    def mocked_import_module(name, package=None):
        # Correctly check for the specific module OpenManusAgent tries to import
        if name == "openmanus_rl.agentgym.agentenv.agentenv.envs":
            print(f"Mocked importlib.import_module for '{name}'")
            mock_envs_module = type('module', (), {})()
            
            class MockTaskClass:
                def __init__(self, client_args_list, n_clients): # Adjusted signature based on OpenManusAgent
                    print(f"MockTaskClass.__init__ called with {len(client_args_list)} client_args, n_clients: {n_clients}")
                    self.clients = []
                    for client_args in client_args_list: # Iterate through the list of client_args
                        self.clients.append(MockEnvClient(
                            server_url=client_args['env_server_base'], 
                            data_len=client_args['data_len'],
                            timeout=client_args['timeout']
                        ))
                
                # Add a dummy close method if OpenManusAgent calls it
                def close(self):
                    print("MockTaskClass.close() called.")

            # The name 'WebshopTask' must match what OpenManusAgent._init_env_clients expects for env_name 'webshop'
            setattr(mock_envs_module, 'WebshopTask', MockTaskClass) 
            return mock_envs_module
        return original_import_module(name, package)

    importlib.import_module = mocked_import_module

    agent_config = AgentConfig(
        max_turns=5,
        max_start_length=512,
        max_prompt_length=1024,
        max_response_length=256,
        max_obs_length=512,
        num_gpus=0, 
        env_name="webshop", 
        env_ports=[12345, 12346], 
        env_server_base="http://mockserver",
        react_format=True,
        env_data_len=10, 
        max_workers=2, # Should be <= len(env_ports)
        algorithm_config=DictConfig({'reward_allocation': 'last_token', 'gamma': 0.99, 'log_prob_micro_batch_size': 1})
    )

    # 3. Instantiate OpenManusAgent
    print("\n--- Initializing OpenManusAgent ---")
    openmanus_agent = OpenManusAgent(
        tokenizer=mock_tokenizer,
        actor_rollout_wg=mock_actor_rollout_wg,
        config=agent_config,
    )
    print(f"OpenManusAgent initialized with {len(openmanus_agent.clients)} mock environment clients.")


    # 4. Prepare Sample DataProto Batch
    print("\n--- Preparing Sample DataProto Batch ---")
    initial_prompts_text = [
        "Find a durable coffee mug under $15.",
        "Search for a blue t-shirt, size medium." 
    ]
    batch_size = len(initial_prompts_text)
    
    tokenized_prompts = mock_tokenizer(initial_prompts_text, return_tensors='pt')
    
    sample_batch_dict = {'input_ids': tokenized_prompts['input_ids']}
    
    sample_meta_info = {'idx': torch.arange(batch_size)}

    sample_data_proto = DataProto.from_dict(sample_batch_dict)
    sample_data_proto.meta_info = sample_meta_info # type: ignore
    print(f"Sample DataProto created with batch_size: {sample_data_proto.batch['input_ids'].shape[0]}")

    # 5. Call run_llm_loop
    print("\n--- Calling OpenManusAgent.run_llm_loop ---")
    # OpenManusAgent.run_llm_loop uses ThreadPoolExecutor and is blocking.
    results_data_proto = openmanus_agent.run_llm_loop(sample_data_proto)

    # 6. Print Results
    print("\n--- Results from OpenManusAgent ---")
    if results_data_proto:
        print(f"Output DataProto batch contains keys: {list(results_data_proto.batch.keys())}")
        print(f"Output DataProto meta_info contains keys: {list(results_data_proto.meta_info.keys())}")

        print(f"  Input IDs shape: {results_data_proto.batch.get('input_ids', torch.empty(0)).shape}")
        print(f"  Token-level rewards shape: {results_data_proto.batch.get('token_level_rewards', torch.empty(0)).shape}")
        print(f"  Responses shape: {results_data_proto.batch.get('responses', torch.empty(0)).shape}")

        if 'rollout_trajectory' in results_data_proto.meta_info:
            print("\n  Rollout Trajectories:")
            for i, trajectory in enumerate(results_data_proto.meta_info['rollout_trajectory']):
                task_idx_tensor = results_data_proto.meta_info['task_idx']
                current_task_idx = task_idx_tensor[i].item() if torch.is_tensor(task_idx_tensor) else task_idx_tensor[i] # Handle tensor or list
                
                reward_tensor = results_data_proto.meta_info['reward']
                current_reward = reward_tensor[i].item() if torch.is_tensor(reward_tensor) else reward_tensor[i]

                env_score_tensor = results_data_proto.meta_info['env_score']
                current_env_score = env_score_tensor[i].item() if torch.is_tensor(env_score_tensor) else env_score_tensor[i]


                print(f"    Trajectory {i+1} (Task Index: {current_task_idx}):")
                for turn in trajectory:
                    print(f"      From: {turn.get('from', 'N/A')}, Value: {str(turn.get('value', 'N/A'))[:100]}...") 
                print(f"    Final Reward for Trajectory {i+1}: {current_reward}")
                print(f"    Final Env Score for Trajectory {i+1}: {current_env_score}")
        else:
            print("  No 'rollout_trajectory' in results meta_info.")
    else:
        print("  run_llm_loop returned None or empty results.")
        
    # Restore original importlib.import_module
    importlib.import_module = original_import_module

    print("\n--- OpenManusAgent Example Script Finished ---")

if __name__ == "__main__":
    # main() is synchronous, no asyncio needed here directly
    main()
