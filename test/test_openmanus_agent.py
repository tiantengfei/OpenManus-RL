from unittest.mock import MagicMock, patch, call
import unittest
import torch

# Make sure the path to openmanus module is correct based on project structure
# Assuming 'openmanus_rl' is a top-level directory in the python path or PYTHONPATH is set
from openmanus_rl.llm_agent.openmanus import OpenManusAgent, AgentConfig

# Define a minimal StepOutput class or use MagicMock for it if it's simple
class MockStepOutput:
    def __init__(self, state, reward, done, info=None):
        self.state = state
        self.reward = reward
        self.done = done
        self.info = info if info is not None else {}

class TestOpenManusAgent(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock(spec=AgentConfig)
        # Common config values that might be accessed
        self.mock_config.max_turns = 5
        self.mock_config.max_prompt_length = 512
        self.mock_config.max_response_length = 128
        self.mock_config.max_obs_length = 256  # Example
        self.mock_config.max_start_length = 256 # Example
        
        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1
        # Let tokenizer call return some dummy tensor output
        self.mock_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
        self.mock_tokenizer.decode = lambda x, **kwargs: "decoded_text" # Simple decode mock

        # Mock actor_rollout_wg
        self.mock_actor_rollout_wg = MagicMock()
        # Mock generate_sequences to return a DataProto-like structure if needed,
        # or just a MagicMock if its internals are not deeply accessed by _run_single_rollout logic itself.
        # For _run_single_rollout, we need 'responses' in batch.
        mock_gen_output = MagicMock()
        mock_gen_output.batch = {'responses': torch.tensor([[4, 5, 6]])} # Dummy response IDs
        self.mock_actor_rollout_wg.generate_sequences.return_value = mock_gen_output
        self.mock_actor_rollout_wg.world_size = 1 # For padding logic in _run_single_rollout


        self.agent = OpenManusAgent(
            tokenizer=self.mock_tokenizer,
            actor_rollout_wg=self.mock_actor_rollout_wg,
            config=self.mock_config
        )
        
        # Mock tensor_fn if its methods are directly called and relevant
        self.agent.tensor_fn = MagicMock()
        self.agent.tensor_fn.create_attention_mask.return_value = torch.ones((1, 3), dtype=torch.long)
        self.agent.tensor_fn.create_position_ids.return_value = torch.arange(3, dtype=torch.long).unsqueeze(0)

    # --- Tests for postprocess_predictions ---

    def test_postprocess_tool_call(self):
        predictions = ["content_str<tool_call>{\"key\": \"value\"}</tool_call>"]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, ["tool_call"])
        self.assertEqual(contents, ["{\"key\": \"value\"}"])

    # test_postprocess_action_tag REMOVED
    # test_postprocess_response_tag REMOVED
    # test_postprocess_mixed_content_action REMOVED
        
    def test_postprocess_mixed_content_tool_call(self):
        # This test verifies that content_str before tool_call doesn't affect parsing of tool_call
        predictions = ["Thought: I need to call a tool. <tool_call>{\"tool_name\": \"calculator\", \"args\": \"1+1\"}</tool_call> Some trailing notes."]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, ["tool_call"])
        self.assertEqual(contents, ["{\"tool_name\": \"calculator\", \"args\": \"1+1\"}"])

    def test_postprocess_no_valid_tags(self):
        predictions = ["Just plain text without any recognized tags."]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, [None])
        self.assertEqual(contents, [""])

    def test_postprocess_multiple_predictions(self):
        predictions = [
            "Text with <action>action1</action>",
            "<tool_call>{\"tool\": \"tool_A\"}</tool_call>",
            "No tags here.",
            "<response>response1</response>" # This will now be None, ""
        ]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, [None, "tool_call", None, None])
        self.assertEqual(contents, ["", "{\"tool\": \"tool_A\"}", "", ""])

    def test_postprocess_empty_string_prediction(self):
        predictions = [""]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, [None])
        self.assertEqual(contents, [""])

    def test_postprocess_tag_with_no_content(self):
        predictions = [
            "<action></action>", # Now None, ""
            "<tool_call></tool_call>", 
            "<response></response>" # Now None, ""
        ]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, [None, "tool_call", None])
        self.assertEqual(contents, ["", "", ""])
        
    def test_postprocess_tag_with_whitespace_content(self):
        predictions = [
            "<action>  </action>", # Now None, ""
            "<tool_call> \n </tool_call>"
        ]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, [None, "tool_call"])
        self.assertEqual(contents, ["", ""]) # .strip() behavior for tool_call content

    def test_postprocess_non_string_prediction(self):
        predictions = [123, None]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, [None, None])
        self.assertEqual(contents, ["", ""])

    # --- Tests for _run_single_rollout ---
    # These tests will focus on the 'next_prompt' handling by checking tokenizer calls.

    @patch('openmanus_rl.llm_agent.openmanus.OpenManusAgent.postprocess_predictions') # Mock to simplify
    def test_run_single_rollout_next_prompt_from_reset_info(self, mock_postprocess):
        mock_postprocess.return_value = (["action"], ["mock_action"]) # Ensure it returns a tuple
        
        mock_client = MagicMock()
        mock_client.reset.return_value = {'next_prompt': 'Initial next prompt'}
        mock_client.observe.return_value = "Initial observation."
        # Make step return a StepOutput-like object to avoid breaking the loop immediately
        # and to allow checking for next_prompt from step info later if needed.
        mock_client.step.return_value = MockStepOutput(state="Obs after step", reward=0, done=True, info={})

        initial_prompt_ids = torch.tensor([[10, 20]]) # Dummy
        self.agent.tokenizer.decode.return_value = "Decoded initial prompt" # For the case where initial_obs_text is empty

        # Reset the mock tokenizer call history for this specific test
        self.agent.tokenizer.reset_mock()
        
        self.agent._run_single_rollout(initial_prompt_ids, task_idx=0, client=mock_client)

        # We expect tokenizer to be called with the combined initial observation and next_prompt
        # The first call to tokenizer in the try block is for initial_obs_text
        expected_initial_text = "Initial observation. Initial next prompt"
        
        # Check all calls to the tokenizer
        found_call = False
        for call_args in self.agent.tokenizer.call_args_list:
            args, kwargs = call_args
            if args and args[0] == expected_initial_text:
                found_call = True
                break
        self.assertTrue(found_call, f"Tokenizer not called with '{expected_initial_text}'")

    @patch('openmanus_rl.llm_agent.openmanus.OpenManusAgent.postprocess_predictions')
    def test_run_single_rollout_next_prompt_from_step_info(self, mock_postprocess):
        mock_postprocess.return_value = (["action"], ["mock_action"])

        mock_client = MagicMock()
        mock_client.reset.return_value = {} # No next_prompt initially
        mock_client.observe.return_value = "Initial observation."
        # Simulate one step, then done=True to exit loop.
        # The step_output contains the next_prompt in its info.
        step_info = {'next_prompt': 'Step next prompt'}
        mock_client.step.side_effect = [
            MockStepOutput(state="Next observation.", reward=0, done=False, info=step_info),
            MockStepOutput(state="Final observation", reward=0, done=True, info={}) # To terminate loop
        ]

        initial_prompt_ids = torch.tensor([[10, 20]])
        self.agent.tokenizer.decode.return_value = "Decoded initial prompt"
        
        # Reset mock before test
        self.agent.tokenizer.reset_mock()
        # Mock the return value for tokenizing the "Next observation. Step next prompt"
        # This is the key part we want to verify.
        # The structure is self.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids']
        # So, the mock_tokenizer itself should be configured.
        # Let's refine the mock_tokenizer setup in setUp for this.
        
        # For this test, we care about the text passed to the tokenizer for the observation *after* the step
        # The agent logic is:
        # 1. Get initial_obs_text -> tokenizer(initial_obs_text)
        # 2. Model generates response_ids
        # 3. client.step() -> next_obs_text, info (with next_prompt)
        # 4. Combined_next_obs_text = next_obs_text + " " + info['next_prompt']
        # 5. tokenizer(Combined_next_obs_text) -> this is what we want to check.

        # Re-mock tokenizer to capture calls more easily for this specific test scenario
        self.agent.tokenizer = MagicMock(
            return_value={'input_ids': torch.tensor([[1,2,3]])}, # Generic return
            pad_token_id=0, 
            eos_token_id=1,
            decode=lambda x, **kwargs: "decoded_text"
        )
        # Also re-mock generate_sequences on actor_rollout_wg as it's part of the agent
        self.agent.actor_rollout_wg.generate_sequences.return_value.batch = {'responses': torch.tensor([[4,5,6]])}


        self.agent._run_single_rollout(initial_prompt_ids, task_idx=0, client=mock_client)
        
        expected_step_obs_text = "Next observation. Step next prompt"
        
        found_call = False
        # Calls to tokenizer:
        # 1. Initial observation: "Initial observation."
        # 2. Observation after step: "Next observation. Step next prompt" (this is what we check)
        # 3. Observation after second step: "Final observation"
        for call_args in self.agent.tokenizer.call_args_list:
            args, kwargs = call_args
            if args and args[0] == expected_step_obs_text:
                found_call = True
                break
        self.assertTrue(found_call, f"Tokenizer not called with '{expected_step_obs_text}'")


    @patch('openmanus_rl.llm_agent.openmanus.OpenManusAgent.postprocess_predictions')
    def test_run_single_rollout_no_next_prompt(self, mock_postprocess):
        mock_postprocess.return_value = (["action"], ["mock_action"])

        mock_client = MagicMock()
        mock_client.reset.return_value = {} # No next_prompt
        mock_client.observe.return_value = "Initial observation."
        mock_client.step.return_value = MockStepOutput(state="Next observation.", reward=0, done=True, info={}) # No next_prompt

        initial_prompt_ids = torch.tensor([[10, 20]])
        self.agent.tokenizer.decode.return_value = "Decoded initial prompt"
        
        self.agent.tokenizer.reset_mock()
        # Re-mock tokenizer for this test
        self.agent.tokenizer = MagicMock(
            return_value={'input_ids': torch.tensor([[1,2,3]])}, 
            pad_token_id=0, 
            eos_token_id=1,
            decode=lambda x, **kwargs: "decoded_text"
        )
        self.agent.actor_rollout_wg.generate_sequences.return_value.batch = {'responses': torch.tensor([[4,5,6]])}


        self.agent._run_single_rollout(initial_prompt_ids, task_idx=0, client=mock_client)

        # Expected calls to tokenizer:
        # 1. "Initial observation."
        # 2. "Next observation." (no next_prompt appended)
        
        calls_made = [c.args[0] for c in self.agent.tokenizer.call_args_list if c.args]

        self.assertIn("Initial observation.", calls_made, "Tokenizer not called with 'Initial observation.'")
        self.assertIn("Next observation.", calls_made, "Tokenizer not called with 'Next observation.'")
        
        # Ensure no other text was snuck in
        self.assertNotIn("Initial observation. Initial next prompt", calls_made)
        self.assertNotIn("Next observation. Step next prompt", calls_made)


if __name__ == '__main__':
    unittest.main()
