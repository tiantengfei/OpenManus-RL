import pytest
import torch # From openmanus_rl/llm_agent/openmanus.py
from omegaconf import DictConfig, OmegaConf # From openmanus_rl/llm_agent/openmanus.py

# Attempt to import necessary components from openmanus-rl
# These imports are based on 'openmanus_rl/llm_agent/openmanus.py' and 'verl/trainer/main_ppo.py'
try:
    from openmanus_rl.llm_agent.openmanus import OpenManusAgent, AgentConfig
    from verl.utils.tokenizer import Tokenizer # As used in Verl PPO trainer
    from verl import DataProto # For gen_batch
    VERL_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Could not import Verl/OpenManusRL components: {e}. Test will be skipped.")
    VERL_COMPONENTS_AVAILABLE = False

# A mock tokenizer and actor_rollout_wg for testing purposes
class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors=None, add_special_tokens=False, padding=None):
        # Simplified tokenizer, returns fixed-size tensor
        return {'input_ids': torch.randint(0, 100, (1, 10), dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        return f"decoded: {token_ids.tolist()}"

class MockActorRolloutWG:
    def __init__(self):
        self.world_size = 1 # Assuming single process for this test
        self.log_prob_micro_batch_size = 1 # Dummy value
        # Mock other attributes if AgentConfig or OpenManusAgent expects them

    def generate_sequences(self, gen_input_proto):
        # Mock sequence generation
        batch_size = gen_input_proto.batch['input_ids'].shape[0]
        # Return mock response IDs
        mock_response_ids = torch.randint(0, 100, (batch_size, 5), dtype=torch.long)
        # Wrap in DataProto as expected by OpenManusAgent
        return DataProto.from_dict({'responses': mock_response_ids})


@pytest.mark.skipif(not VERL_COMPONENTS_AVAILABLE, reason="Verl/OpenManusRL components not available")
def test_openmanus_env_initialization_and_step():
    print("Starting test_openmanus_env_initialization_and_step...")

    # 1. Setup mock components
    mock_tokenizer = MockTokenizer()
    mock_actor_wg = MockActorRolloutWG()

    # 2. Create AgentConfig for "openmanus" environment
    # These are minimal values, real training would use more complex configs.
    # The key is 'env_name' and a valid (even if unused by mock) 'env_ports' and 'env_server_base'.
    # 'openmanus_config' is passed via client_args in BaseTask, so it's nested.
    agent_cfg_dict = {
        "max_turns": 5,
        "max_start_length": 50,
        "max_prompt_length": 100,
        "max_response_length": 20,
        "max_obs_length": 200,
        "num_gpus": 0, # Assuming CPU for this test
        "env_name": "openmanus", # Critical: use the new environment name
        "env_ports": [12345],    # Dummy port, not used by local OpenManusEnvClient
        "env_server_base": "http://localhost", # Dummy server base
        "react_format": True,
        "env_data_len": 1, # Number of "tasks"
        "max_workers": 1,
        # Pass the openmanus_session_config here, which OpenManusTask will give to OpenManusEnvClient
        "openmanus_config": {
            "max_steps": 3, # For the mock OpenManusSessionExecutor
            "reward_success": 1.0,
            "reward_failure": -1.0,
            "reward_timeout": -0.5,
            "reward_step": 0.0,
        }
        # algorithm_config might be needed if AgentConfig strictly requires it.
        # For now, assuming it can be None or an empty DictConfig for this test.
        # "algorithm_config": {} # Example if it's needed.
    }
    # Convert dict to DictConfig
    # Note: OpenManusAgent expects algorithm_config as DictConfig. If we pass one, it must be DictConfig.
    # If AgentConfig has a default of None for algorithm_config, we might not need to pass it.
    # The AgentConfig in openmanus.py has: algorithm_config: DictConfig = None
    # So, we might not need to explicitly pass it if the default is fine.
    # However, the AgentConfig also takes 'openmanus_config' as a direct key if we modify AgentConfig,
    # or it's passed via client_args if we don't.
    # The current setup passes 'openmanus_config' via client_args in BaseTask.
    # Let's ensure AgentConfig is created correctly.
    
    # AgentConfig expects algorithm_config to be a DictConfig if provided.
    # For the rest, it takes them directly.
    # The 'openmanus_config' is not a direct field of AgentConfig but part of the **kwargs
    # that BaseTask passes down if we put it inside client_args.
    # However, OpenManusAgent constructor directly takes 'config: AgentConfig'.
    # The current BaseTask takes client_args: Mapping[str, Any].
    # And OpenManusTask.__init__ calls super().__init__(client_args, ...).
    # BaseTask.__init__ then creates clients using self.env_client_cls(**client_args_for_each_client).
    # So, 'openmanus_config' should be part of the dictionary that AgentConfig is made from.
    # Let's refine agent_cfg_dict to be exactly what AgentConfig takes.
    # The 'openmanus_config' should be passed within the `client_args` for the task,
    # which `OpenManusAgent` doesn't directly create.
    # `OpenManusAgent` creates the clients using `TaskClass(client_args=...)`
    # The `client_args` for `TaskClass` is formed internally in `OpenManusAgent` like:
    # client_args={"env_server_base": server_url, "data_len": self.config.env_data_len, ...}
    # It does NOT seem to pass arbitrary extra keys from AgentConfig into client_args.
    
    # This means we need 'openmanus_config' to be part of the config that OpenManusEnvClient receives.
    # The current OpenManusTask passes its `client_args` directly to OpenManusEnvClient.
    # So, if OpenManusAgent is modified to pass `openmanus_config` as part of its `client_args` to the Task, it would work.
    # Alternatively, `OpenManusTask` can retrieve it from a global config or a modified AgentConfig.

    # For this test, let's assume AgentConfig can hold openmanus_config and the Task can access it,
    # or that OpenManusAgent._init_env_clients can be minimally adjusted if needed (though plan is to avoid).
    # The current openmanus_env.py has OpenManusEnvClient taking 'openmanus_config' as a direct kwarg.
    # And OpenManusTask passes its whole 'client_args' mapping to OpenManusEnvClient.
    # So, if 'openmanus_config' is in the 'client_args' given to OpenManusTask, it works.
    # `OpenManusAgent._init_env_clients` makes `client_args` like:
    # client_args={"env_server_base": server_url, "data_len": self.config.env_data_len, "timeout": 300}
    # It does NOT add other things from self.config (AgentConfig instance) into these client_args.
    # This is a problem for passing `openmanus_config`.

    # Easiest solution for the test without modifying OpenManusAgent:
    # Make 'openmanus_config' a field in AgentConfig.
    # Then, OpenManusTask can receive AgentConfig, or OpenManusEnvClient can.
    # But the plan is to avoid modifying OpenManusAgent.
    # The current `OpenManusTask` init is `__init__(self, client_args: Mapping[str, Any], n_clients: int, ...)`
    # The `client_args` it receives *is* the dict from `OpenManusAgent`'s `_init_env_clients`.
    # This `client_args` does NOT contain `openmanus_config`.

    # Temporary solution for the test: assume 'openmanus_config' is hardcoded or globally accessible in OpenManusEnvClient,
    # or that the `agent_cfg_dict` used to create `AgentConfig` *also* serves as the source for `openmanus_config`
    # if `OpenManusEnvClient` could access the main `AgentConfig`.
    # The `OpenManusEnvClient` does not get the main `AgentConfig`. It gets a small `client_args` dict.

    # Let's simplify the test to focus on direct instantiation of OpenManusTask if testing OpenManusAgent is too complex due to config.
    # However, the request is to test integration with OpenManusAgent.
    # The `openmanus_config` in `agent_cfg_dict` will be used to make the `AgentConfig`.
    # If `OpenManusEnvClient` needs it, it must come from the `client_args` it receives.
    # Since `OpenManusAgent` doesn't pass it, the mock will use its own defaults for `max_steps` etc.
    # This is acceptable for a basic integration test. The mock has defaults.

    agent_config = AgentConfig(**OmegaConf.structured(AgentConfig(**agent_cfg_dict)))


    # 3. Instantiate OpenManusAgent
    # It's possible that OpenManusAgent's __init__ tries to connect to servers if env_ports is not empty.
    # The MockOpenManusSessionExecutor doesn't use ports, so this should be fine.
    try:
        agent = OpenManusAgent(
            tokenizer=mock_tokenizer,
            actor_rollout_wg=mock_actor_wg,
            config=agent_config,
            # logger=None # Default is None
        )
    except Exception as e:
        pytest.fail(f"Failed to initialize OpenManusAgent for 'openmanus' env: {e}")

    # Ensure clients were created
    assert agent.clients is not None, "Agent clients list should be initialized."
    assert len(agent.clients) > 0, "Agent should have at least one client for 'openmanus' env."
    # Check type of client (optional, depends on direct access)
    # from openmanus_rl.agentgym.agentenv.agentenv.envs.openmanus_env import OpenManusEnvClient
    # assert isinstance(agent.clients[0], OpenManusEnvClient), "Client should be an OpenManusEnvClient instance."


    # 4. Create a dummy batch for run_llm_loop
    # Based on PPOAgent.sample_batch() and how OpenManusAgent uses gen_batch
    dummy_initial_prompts_ids = torch.randint(0, 100, (1, agent_config.max_start_length), dtype=torch.long)
    gen_batch = DataProto.from_dict({
        'input_ids': dummy_initial_prompts_ids,
        'attention_mask': torch.ones_like(dummy_initial_prompts_ids),
        # 'idx': torch.tensor([0]), # If needed by any processing step
    })
    # OpenManusAgent's run_llm_loop also expects some meta_info in gen_batch if using original_indices_map
    # For this test, _convert_rollout_results_to_dataproto is the main consumer.
    # It gets 'idx' from meta_info or defaults to range(batch_size).
    if not hasattr(gen_batch, 'meta_info'):
        gen_batch.meta_info = {}
    gen_batch.meta_info['idx'] = torch.tensor([0], dtype=torch.long)


    # 5. Run the LLM loop (which includes reset, step, observe)
    print("Running agent.run_llm_loop...")
    try:
        processed_data = agent.run_llm_loop(gen_batch)
        print(f"run_llm_loop completed. Processed data: {processed_data.batch.keys()}")
    except Exception as e:
        pytest.fail(f"agent.run_llm_loop failed: {e}")

    # 6. Basic assertions on the output
    assert "input_ids" in processed_data.batch, "Processed data should contain 'input_ids'"
    assert "token_level_rewards" in processed_data.batch, "Processed data should contain 'token_level_rewards'"
    assert processed_data.batch['input_ids'].shape[0] == 1, "Should have one result for batch size 1"
    
    # Check for trajectory in meta_info (if mock logic produces one)
    assert "rollout_trajectory" in processed_data.meta_info
    assert len(processed_data.meta_info["rollout_trajectory"]) == 1
    trajectory = processed_data.meta_info["rollout_trajectory"][0]
    print(f"Sample trajectory: {trajectory}")
    assert len(trajectory) > 0, "Trajectory should not be empty"
    
    # Check if the task completed as per mock logic (e.g., max_steps reached)
    # MockOpenManusSessionExecutor terminates at max_steps (default 10, or 3 if config was passed).
    # OpenManusAgent's max_turns is 5.
    # The loop in _run_single_rollout runs for self.config.max_turns (5).
    # The mock environment's max_steps is 3 (if config is correctly passed).
    # So, the mock env should terminate.
    assert processed_data.meta_info["env_score"][0] != 0.0 or processed_data.meta_info["reward"][0] != 0.0, \
        "Expected some reward/score change, or task completion."
    
    # One of the 'done' flags in the trajectory's steps, or the final 'done' from _run_single_rollout, should be true.
    # The 'done' status from _run_single_rollout is not directly in processed_data but influences rewards/trajectory.
    # We expect the mock task to be "done" due to reaching max_steps.
    # If it's done, the reward should reflect that (e.g. reward_timeout).
    # This depends on how agent_config.openmanus_config flows to the mock.
    # If agent_config.openmanus_config["max_steps"] = 3 was used, it should timeout.
    # The default in MockOpenManusSessionExecutor is 10 steps.
    # Let's check the reward. If it's the timeout reward, config was passed.
    # Expected reward_timeout is -0.5 if config was passed.
    # If config wasn't passed, it would run for 5 turns (agent.max_turns)
    # and the mock env (10 max_steps) wouldn't be 'done' by its own max_steps.
    # It would be 'done' because the agent loop finishes.
    # The reward would be 0.0 (step_reward) unless the action was "success".

    print(f"Final reward from meta_info: {processed_data.meta_info['reward'][0]}")
    print(f"Final env_score from meta_info: {processed_data.meta_info['env_score'][0]}")
    # This test is primarily for integration (no crash), not specific reward values yet
    # unless config propagation is confirmed.

    print("test_openmanus_env_initialization_and_step completed successfully.")

# To run this test:
# Ensure pytest is installed.
# Navigate to the root of the openmanus-rl project.
# Run: pytest test/test_openmanus_env_integration.py
