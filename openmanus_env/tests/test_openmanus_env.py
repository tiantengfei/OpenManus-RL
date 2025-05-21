import unittest
from unittest.mock import MagicMock, patch, Mock
import torch
from typing import List, Any, Dict

# Attempt to import from the openmanus_env package. Adjust path if necessary.
from ..config import OpenManusEnvConfig
from ..env_utils import initialize_env_clients
from ..core import OpenManusCoreEnv

from verl import DataProto # Assuming verl is installed and accessible
from transformers import GenerationConfig # Assuming transformers is installed

# A mock tokenizer class for testing
class MockTokenizer:
    def __init__(self, pad_token_id=0, eos_token_id=1):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4, "<action>":5, "</action>":6, "คิด": 7} # Simple vocab

    def __call__(self, text_batch, return_tensors=None, add_special_tokens=False, padding=False, truncation=False, max_length=None):
        if isinstance(text_batch, str):
            text_batch = [text_batch]
        
        all_ids = []
        for text in text_batch:
            # Simplified tokenization: split by space or known tokens
            tokens = []
            current_word = ""
            i = 0
            while i < len(text):
                found_known_token = False
                for known, id_val in self.vocab.items():
                    if text[i:].startswith(known):
                        if current_word:
                            tokens.append(current_word)
                            current_word = ""
                        tokens.append(known)
                        i += len(known)
                        found_known_token = True
                        break
                if not found_known_token:
                    if text[i] == ' ': # Treat space as separator
                        if current_word:
                            tokens.append(current_word)
                            current_word = ""
                    else:
                        current_word += text[i]
                    i += 1
            if current_word: # Add last word
                tokens.append(current_word)

            ids = [self.vocab.get(token, self.vocab["<pad>"]) for token in tokens] # Default to pad_token_id if OOV
            all_ids.append(ids)

        if padding == "longest" or padding is True:
            # Corrected max_len calculation for lists of lists
            max_len = 0
            if all_ids and any(all_ids): # ensure all_ids itself and its sublists are not empty
                max_len = max(len(ids) for ids in all_ids if ids) if any(len(ids)>0 for ids in all_ids) else 0
            
            for ids in all_ids:
                ids.extend([self.pad_token_id] * (max_len - len(ids)))
        
        if truncation and max_length:
            all_ids = [ids[:max_length] for ids in all_ids]

        if return_tensors == "pt":
            return {"input_ids": torch.tensor(all_ids, dtype=torch.long)}
        return {"input_ids": all_ids} # Return list of lists by default

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim > 1: # Handle batch
                token_ids = token_ids[0] # For simplicity, decode first in batch
            token_ids = token_ids.tolist()
        
        rev_vocab = {v: k for k, v in self.vocab.items()}
        tokens = []
        for tid in token_ids:
            token_str = rev_vocab.get(tid, "")
            if skip_special_tokens and (tid == self.pad_token_id or tid == self.eos_token_id or tid == 1): # 1 is <s>
                continue
            tokens.append(token_str)
        return "".join(tokens) # Simple join, might need space logic for real tokenizer

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]


class TestOpenManusEnv(unittest.TestCase):

    def test_openmanus_env_config_instantiation(self):
        config = OpenManusEnvConfig(
            max_turns=10,
            max_start_length=100,
            max_prompt_length=512,
            max_response_length=128,
            max_obs_length=1024,
            env_name="webshop",
            env_ports=[10000, 10001],
            env_server_base="http://localhost",
            react_format=True,
            env_data_len=50,
            max_workers=2
        )
        self.assertEqual(config.env_name, "webshop")
        self.assertEqual(len(config.env_ports), 2)

    @patch('openmanus_env.env_utils.importlib.import_module')
    def test_initialize_env_clients_basic(self, mock_import_module):
        # Mock the TaskClass and its client
        MockClient = MagicMock() # This is an instance of a client
        MockTaskInstance = MagicMock()
        MockTaskInstance.clients = [MockClient] # clients attribute should be a list of client instances
        MockTaskClass = MagicMock(return_value=MockTaskInstance) # TaskClass constructor returns an instance of the task
        
        mock_envs_module = MagicMock()
        # This assumes that in your env_utils.ENV_TO_TASK_CLASS, "webshop" maps to "WebshopTask"
        # and "WebshopTask" is the actual class name in the module.
        setattr(mock_envs_module, "WebshopTask", MockTaskClass) 
        mock_import_module.return_value = mock_envs_module

        # Create a config instance for the test
        config = OpenManusEnvConfig(
            env_name="webshop",
            env_ports=[10000],
            env_server_base="http://localhost",
            # Fill in other required fields for OpenManusEnvConfig
            max_turns=5,
            max_start_length=100,
            max_prompt_length=512,
            max_response_length=64,
            max_obs_length=1024,
            react_format=True, # Assuming a default or typical value
            env_data_len=200,  # Assuming a default or typical value
            max_workers=10   # Assuming a default or typical value
        )
        
        clients = initialize_env_clients(config)
        self.assertEqual(len(clients), 1)
        self.assertEqual(clients[0], MockClient) # Ensure the client instance is returned
        # The import path should match what's constructed in initialize_env_clients
        # e.g., f"agentenv.envs.{env_name_lower}.task"
        mock_import_module.assert_called_with("agentenv.envs.webshop.task") 
        MockTaskClass.assert_called_once()
        
        args, kwargs = MockTaskClass.call_args
        self.assertIn('client_args', kwargs)
        # server_url is constructed inside initialize_env_clients
        self.assertEqual(kwargs['client_args']['server_url'], "http://localhost:10000")


    def test_initialize_env_clients_unsupported_env(self):
        config = OpenManusEnvConfig(
            env_name="nonexistent_env", env_ports=[10000], env_server_base="http://localhost",
            # Fill in other required fields as in the previous test
            max_turns=5, max_start_length=100, max_prompt_length=512, max_response_length=64,
            max_obs_length=1024, react_format=True, env_data_len=200, max_workers=10
        )
        with self.assertRaises(ValueError):
            initialize_env_clients(config)

    def test_postprocess_predictions(self):
        # Dummy core_env, only postprocess_predictions is tested here
        # Provide necessary mock arguments for OpenManusCoreEnv constructor
        mock_config = MagicMock(spec=OpenManusEnvConfig)
        mock_tokenizer = MagicMock()
        mock_actor_rollout_wg = MagicMock()
        mock_clients_list = [] # Empty list if clients are not used in postprocess_predictions
        
        core_env = OpenManusCoreEnv(
            config=mock_config, 
            tokenizer=mock_tokenizer, 
            actor_rollout_wg=mock_actor_rollout_wg, 
            clients=mock_clients_list
        )
        
        pred1 = "<action>do_something</action>"
        pred2 = "<response>I did something.</response>"
        pred3 = "Just some text."
        pred4 = "<action>  leading/trailing spaces  </action>"

        actions, contents = core_env.postprocess_predictions([pred1, pred2, pred3, pred4])
        
        self.assertEqual(actions, ['action', 'response', None, 'action'])
        self.assertEqual(contents, ['do_something', 'I did something.', '', 'leading/trailing spaces'])

    def test_convert_rollout_results_to_dataproto_structure(self):
        mock_tokenizer = MockTokenizer(pad_token_id=0, eos_token_id=1)
        
        mock_config_dict = {
            'max_turns': 5, 'max_start_length': 50, 'max_prompt_length': 128, 
            'max_response_length': 64, 'max_obs_length': 100,
            'env_name': 'test_env', 'env_ports': [12345], 'env_server_base': 'http://dummy',
            'react_format': True, 'env_data_len': 10, 'max_workers': 1,
            # This will be used by config.algorithm_config.get()
            'algorithm_config': {'reward_allocation': 'last_token', 'gamma': 0.99}, 
            'temperature': 0.7, 'log_prob_use_dynamic_bsz': False 
        }
        
        # Use a real OpenManusEnvConfig and set algorithm_config as a dict for .get()
        config = OpenManusEnvConfig(**{k: v for k, v in mock_config_dict.items() if k != 'algorithm_config'})
        config.algorithm_config = mock_config_dict['algorithm_config'] # Set as dict
        # Also set other attributes directly if not part of constructor
        config.temperature = mock_config_dict['temperature']
        config.log_prob_use_dynamic_bsz = mock_config_dict['log_prob_use_dynamic_bsz']


        mock_actor_rollout_wg = MagicMock()
        mock_actor_rollout_wg.world_size = 1 
        mock_actor_rollout_wg.log_prob_micro_batch_size = 32


        core_env = OpenManusCoreEnv(config=config, tokenizer=mock_tokenizer, actor_rollout_wg=mock_actor_rollout_wg, clients=[MagicMock()])

        results = [
            {
                'trajectory': [
                    {"from": "human", "value": "hello world"},
                    {"from": "gpt", "value": "<action>คิด</action>"}, 
                    {"from": "env", "value": "OK"}
                ],
                'step_rewards': [0.5], 'reward': 0.5, 'env_score': 1.0, 
                'turns': 1, 'valid_actions': 1, 'task_idx': 0, 'done': False
            }
        ]

        original_input_ids = mock_tokenizer(["initial prompt"], return_tensors="pt")["input_ids"]
        original_batch = DataProto.from_dict({'input_ids': original_input_ids})
        original_batch.meta_info = {'idx': torch.tensor([0])}

        data_proto = core_env.convert_rollout_results_to_dataproto(results, original_batch)

        self.assertIsInstance(data_proto, DataProto)
        self.assertIn('input_ids', data_proto.batch)
        self.assertIn('attention_mask', data_proto.batch)
        self.assertIn('position_ids', data_proto.batch)
        self.assertIn('info_mask', data_proto.batch)
        self.assertIn('token_level_rewards', data_proto.batch)
        self.assertIn('responses', data_proto.batch)

        self.assertEqual(data_proto.batch['input_ids'].shape[0], 1) 
        self.assertEqual(data_proto.batch['input_ids'].shape[1], config.max_prompt_length)
        self.assertEqual(data_proto.batch['responses'].shape[1], config.max_response_length)
        
        
        self.assertTrue(torch.sum(data_proto.batch['token_level_rewards']) > 0) 
        if results[0]['env_score'] != 0:
             # For last_token, exactly one token should have the reward if score is non-zero
             if config.algorithm_config['reward_allocation'] == 'last_token':
                 self.assertEqual(torch.max(data_proto.batch['token_level_rewards']), results[0]['env_score'])
                 self.assertEqual(torch.sum(data_proto.batch['token_level_rewards'] > 0), 1)


        self.assertIn('task_idx', data_proto.meta_info)
        self.assertIn('reward', data_proto.meta_info)
        self.assertIn('env_score', data_proto.meta_info)
        self.assertIn('rollout_trajectory', data_proto.meta_info)
        
        self.assertIn('micro_batch_size', data_proto.meta_info)
        self.assertEqual(data_proto.meta_info['micro_batch_size'], 32)
        self.assertIn('temperature', data_proto.meta_info)
        self.assertEqual(data_proto.meta_info['temperature'], 0.7)
        self.assertIn('use_dynamic_bsz', data_proto.meta_info)
        self.assertEqual(data_proto.meta_info['use_dynamic_bsz'], False)


if __name__ == '__main__':
    unittest.main()
```
