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

    # Old tests are preserved below, new comprehensive tests are added here:
    def test_postprocess_predictions_updated(self):
        # Test Case 1: Natural Language and Single Tool Call (User's Example)
        pred1 = ['为了获取大理理想邦圣托里尼的最佳拍照时间段和取景角度，我将通过网络搜索相关信息。请稍候。\n<tool_call>\n{"name": "travel_web_search", "arguments": {"keyword": "大理理想邦圣托里尼 最佳拍照时间段 取景角度", "page_size": 10, "page_num": 0}}\n</tool_call>']
        exp_action1 = 'natural_language_and_tool_call'
        # Note: The tool_call_content itself is a string. json.dumps will escape internal quotes.
        exp_content1 = '{"natural_language": "为了获取大理理想邦圣托里尼的最佳拍照时间段和取景角度，我将通过网络搜索相关信息。请稍候。", "tool_call_content": "{\\"name\\": \\"travel_web_search\\", \\"arguments\\": {\\"keyword\\": \\"大理理想邦圣托里尼 最佳拍照时间段 取景角度\\", \\"page_size\\": 10, \\"page_num\\": 0}}"}'
        
        action_types, contents = self.agent.postprocess_predictions(pred1)
        self.assertEqual(action_types[0], exp_action1)
        self.assertEqual(contents[0], exp_content1)

        # Test Case 2: Only Single Tool Call
        pred2 = ['<tool_call>{"name": "calculator", "expression": "2+2"}</tool_call>']
        exp_action2 = 'tool_call'
        exp_content2 = '{"name": "calculator", "expression": "2+2"}'
        action_types, contents = self.agent.postprocess_predictions(pred2)
        self.assertEqual(action_types[0], exp_action2)
        self.assertEqual(contents[0], exp_content2)

        # Test Case 3: Only Natural Language
        pred3 = ['Hello, how are you today?']
        exp_action3 = 'natural_language_response'
        exp_content3 = 'Hello, how are you today?'
        action_types, contents = self.agent.postprocess_predictions(pred3)
        self.assertEqual(action_types[0], exp_action3)
        self.assertEqual(contents[0], exp_content3)

        # Test Case 4: Empty Input String
        pred4 = ['']
        exp_action4 = None
        exp_content4 = ''
        action_types, contents = self.agent.postprocess_predictions(pred4)
        self.assertEqual(action_types[0], exp_action4)
        self.assertEqual(contents[0], exp_content4)

        # Test Case 5: Non-String Input
        pred5_none = [None]
        action_types, contents = self.agent.postprocess_predictions(pred5_none)
        self.assertEqual(action_types[0], None)
        self.assertEqual(contents[0], '')
        
        pred5_int = [123]
        action_types, contents = self.agent.postprocess_predictions(pred5_int)
        self.assertEqual(action_types[0], None)
        self.assertEqual(contents[0], '')

        # Test Case 6: Natural Language and Multiple Tool Calls
        pred6 = ['First do this:\n<tool_call>{"name": "tool1", "arg": "A"}</tool_call>\nThen do that:\n<tool_call>{"name": "tool2", "arg": "B"}</tool_call>']
        exp_action6 = 'natural_language_and_tool_call'
        # The tool_call_content will be a string representing a JSON array of JSON objects (strings)
        exp_content6 = '{"natural_language": "First do this:", "tool_call_content": "[{\\"name\\": \\"tool1\\", \\"arg\\": \\"A\\"}, {\\"name\\": \\"tool2\\", \\"arg\\": \\"B\\"}]"}'
        action_types, contents = self.agent.postprocess_predictions(pred6)
        self.assertEqual(action_types[0], exp_action6)
        self.assertEqual(contents[0], exp_content6)
        
        # Test Case 7: Tool call without preceding natural language, but with text after
        pred7 = ['<tool_call>{"name": "intermediate_step"}</tool_call>\nThis is some trailing text.']
        exp_action7 = 'tool_call' # Natural language part is empty because it's before the first tool_call
        exp_content7 = '{"name": "intermediate_step"}'
        action_types, contents = self.agent.postprocess_predictions(pred7)
        self.assertEqual(action_types[0], exp_action7)
        self.assertEqual(contents[0], exp_content7)

        # Test Case 8: Prediction with only whitespace
        pred8 = ['   ']
        # Whitespace is stripped, if it becomes empty and no tool call, it's None type
        exp_action8 = 'natural_language_response' # Corrected: if natural_language_part is "   ", it strips to "", but it was not empty before stripping.
                                                # The logic is `bool(natural_language_part)` where `natural_language_part` is the stripped version.
                                                # If `prediction.strip()` is empty, and no tool_call, then `natural_language_part` is empty.
                                                # `has_natural_language` becomes `False`.
                                                # So, `current_action_type` becomes `None`.
        exp_content8 = '   '                 # Corrected: content is the stripped natural language part.
                                                # If prediction is "   ", natural_language_part (after strip) is "",
                                                # so has_natural_language is False. If no tool_call, type is None, content is "".
                                                # Re-evaluating based on current code:
                                                # prediction = "   " -> first_tool_call_idx raises ValueError
                                                # natural_language_part = prediction.strip() which is ""
                                                # tool_call_content_list is []
                                                # final_tool_call_output is None
                                                # has_natural_language = bool("") which is False
                                                # has_tool_call = False
                                                # current_action_type = None
                                                # current_action_content = ""
        action_types, contents = self.agent.postprocess_predictions(pred8)
        self.assertEqual(action_types[0], None) # Corrected expectation based on code logic
        self.assertEqual(contents[0], "")       # Corrected expectation


    def test_postprocess_tool_call(self):
        predictions = ["content_str<tool_call>{\"key\": \"value\"}</tool_call>"]
        # With the new logic, "content_str" becomes natural_language part.
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, ['natural_language_and_tool_call'])
        expected_content = '{"natural_language": "content_str", "tool_call_content": "{\\"key\\": \\"value\\"}"}'
        self.assertEqual(contents, [expected_content])

    # test_postprocess_action_tag REMOVED
    # test_postprocess_response_tag REMOVED
    # test_postprocess_mixed_content_action REMOVED
        
    def test_postprocess_mixed_content_tool_call(self):
        # This test verifies that content_str before tool_call doesn't affect parsing of tool_call
        predictions = ["Thought: I need to call a tool. <tool_call>{\"tool_name\": \"calculator\", \"args\": \"1+1\"}</tool_call> Some trailing notes."]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        # "Some trailing notes." is currently ignored by the logic if a tool_call is present and natural language is taken from before the first tool_call.
        self.assertEqual(action_types, ["natural_language_and_tool_call"])
        expected_content = '{"natural_language": "Thought: I need to call a tool.", "tool_call_content": "{\\"tool_name\\": \\"calculator\\", \\"args\\": \\"1+1\\"}"}'
        self.assertEqual(contents, [expected_content])

    def test_postprocess_no_valid_tags(self):
        predictions = ["Just plain text without any recognized tags."]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, ['natural_language_response']) # Corrected: no tags means natural_language_response
        self.assertEqual(contents, ["Just plain text without any recognized tags."]) # Corrected: content is the text itself

    def test_postprocess_multiple_predictions(self):
        predictions = [
            "Text with <action>action1</action>", # This would be natural language as <action> is not special
            "<tool_call>{\"tool\": \"tool_A\"}</tool_call>",
            "No tags here.",
            "<response>response1</response>" # This would be natural language as <response> is not special
        ]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, ['natural_language_response', "tool_call", 'natural_language_response', 'natural_language_response'])
        self.assertEqual(contents, ["Text with <action>action1</action>", "{\"tool\": \"tool_A\"}", "No tags here.", "<response>response1</response>"])

    def test_postprocess_empty_string_prediction(self):
        predictions = [""]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, [None]) # Correct: empty string leads to no action type
        self.assertEqual(contents, [""])

    def test_postprocess_tag_with_no_content(self):
        predictions = [
            "<action></action>", 
            "<tool_call></tool_call>", # tool_call with empty content is still a tool_call
            "<response></response>" 
        ]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, ['natural_language_response', "tool_call", 'natural_language_response'])
        self.assertEqual(contents, ["<action></action>", "", "<response></response>"]) # content of tool_call is ""
        
    def test_postprocess_tag_with_whitespace_content(self):
        predictions = [
            "<action>  </action>", 
            "<tool_call> \n </tool_call>" # Whitespace inside tool_call is stripped
        ]
        action_types, contents = self.agent.postprocess_predictions(predictions)
        self.assertEqual(action_types, ['natural_language_response', "tool_call"])
        self.assertEqual(contents, ["<action>  </action>", ""]) # .strip() behavior for tool_call content

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
