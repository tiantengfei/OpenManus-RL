import torch
import re
import traceback
from typing import List, Dict, Any, Tuple
from verl import DataProto # Ensure DataProto is imported
from transformers import GenerationConfig
from .config import OpenManusEnvConfig
import concurrent.futures

# Temporary import, will be addressed later
from openmanus_rl.llm_agent.tensor_helper import TensorHelper, TensorConfig


class OpenManusCoreEnv:
    def __init__(self, config: OpenManusEnvConfig, tokenizer: Any, actor_rollout_wg: Any, clients: List[Any]):
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.clients = clients

        tensor_config = TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            max_prompt_length=self.config.max_prompt_length,
            max_obs_length=self.config.max_obs_length,
            max_start_length=self.config.max_start_length
        )
        self.tensor_fn = TensorHelper(tensor_config)

        num_clients = len(self.clients)
        actual_workers = 0
        if num_clients > 0: # Ensure clients list is not empty
            # Default max_workers to 1 if not defined or if it's 0 but clients exist
            configured_max_workers = getattr(self.config, 'max_workers', 1) 
            if configured_max_workers == 0 and num_clients > 0:
                configured_max_workers = 1 # Ensure at least 1 worker if clients are present
            actual_workers = min(num_clients, configured_max_workers)
        
        if actual_workers > 0:
            # Only warn if max_workers was explicitly set and constraining
            if hasattr(self.config, 'max_workers') and actual_workers < num_clients and self.config.max_workers > 0:
                print(f"[CoreEnv.__init__] Warning: Number of clients ({num_clients}) exceeds max_workers ({self.config.max_workers}). Using {actual_workers} workers.")
            print(f"[CoreEnv.__init__] Initializing ThreadPoolExecutor with {actual_workers} workers for {num_clients} clients.")
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers)
        else:
            print(f"[CoreEnv.__init__] No clients or configured workers to initialize ThreadPoolExecutor.")
            self.executor = None # Set executor to None if no workers


    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str):
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
                    actions.append(None)
                    contents.append('')
            else:
                print(f"[Warning] Received non-string prediction: {type(prediction)}. Cannot process.")
                actions.append(None)
                contents.append('')
        return actions, contents

    def _run_single_rollout(self, initial_prompt_ids: torch.Tensor, task_idx: int, client: Any) -> Dict[str, Any]:
        trajectory = []
        step_rewards = []
        final_reward = 0.0
        final_env_score = 0.0
        done = False
        turns = 0
        current_input_ids = None

        try:
            reset_info = client.reset(task_idx)
            initial_obs_text = client.observe()

            if not initial_obs_text:
                initial_prompt_text = self.tokenizer.decode(initial_prompt_ids[0], skip_special_tokens=True)
                trajectory.append({"from": "human", "value": initial_prompt_text})
                current_input_ids = initial_prompt_ids
            else:
                trajectory.append({"from": "human", "value": initial_obs_text})
                current_input_ids = self.tokenizer(initial_obs_text, return_tensors='pt', add_special_tokens=False)['input_ids']

            for t in range(self.config.max_turns):
                turns = t + 1
                if current_input_ids is None:
                    break

                if current_input_ids.shape[1] > self.config.max_prompt_length:
                    current_input_ids = current_input_ids[:, -self.config.max_prompt_length:]

                current_attention_mask = self.tensor_fn.create_attention_mask(current_input_ids)
                current_position_ids = self.tensor_fn.create_position_ids(current_attention_mask)
                
                gen_input_proto = DataProto.from_dict({
                    'input_ids': current_input_ids,
                    'attention_mask': current_attention_mask,
                    'position_ids': current_position_ids
                })

                world_size = self.actor_rollout_wg.world_size
                original_size = 1 
                padded_gen_input_proto = gen_input_proto
                padding_size = 0
                if world_size > 1 and original_size % world_size != 0:
                    padding_size = world_size - (original_size % world_size)
                    padded_batch = {}
                    for k, v_tensor in gen_input_proto.batch.items(): 
                        if v_tensor.shape[0] > 0:
                           pad_sequence = v_tensor[0:1].repeat(padding_size, *[1] * (len(v_tensor.shape) - 1))
                           padded_batch[k] = torch.cat([v_tensor, pad_sequence], dim=0)
                        else:
                           padded_batch[k] = v_tensor
                    padded_gen_input_proto = DataProto.from_dict(padded_batch)
                    if hasattr(gen_input_proto, 'meta_info'):
                         padded_gen_input_proto.meta_info = gen_input_proto.meta_info.copy()

                generation_config = GenerationConfig(
                    max_new_tokens=self.config.max_response_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=1.0,
                    do_sample=True
                )

                if not hasattr(padded_gen_input_proto, 'meta_info'):
                    padded_gen_input_proto.meta_info = {}
                padded_gen_input_proto.meta_info['generation_config'] = generation_config
                
                if 'input_ids' in padded_gen_input_proto.batch:
                     padded_gen_input_proto.batch['sequences'] = padded_gen_input_proto.batch['input_ids']

                gen_output_proto = self.actor_rollout_wg.generate_sequences(padded_gen_input_proto)
                response_ids = gen_output_proto.batch['responses']

                if padding_size > 0 and response_ids.shape[0] > original_size: 
                     response_ids = response_ids[:-padding_size]

                response_text = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
                trajectory.append({"from": "gpt", "value": response_text})

                _, action_contents = self.postprocess_predictions([response_text])
                action_text = action_contents[0] if action_contents else ""
                if action_text is None: action_text = ""

                step_output = client.step(action_text)
                next_obs_text = step_output.state
                reward = step_output.reward
                done = step_output.done
                info = getattr(step_output, 'info', {})

                step_rewards.append(reward)
                final_reward = reward
                final_env_score = info.get('score', 0.0)

                trajectory[-1]['reward'] = reward
                trajectory[-1]['info'] = info

                if not done:
                    trajectory.append({"from": "env", "value": next_obs_text})
                    if next_obs_text.strip():
                        next_obs_ids = self.tokenizer(next_obs_text, return_tensors='pt', add_special_tokens=False)['input_ids']
                        current_input_ids = torch.cat([
                            current_input_ids.to(response_ids.device),
                            response_ids,
                            next_obs_ids.to(response_ids.device)
                        ], dim=1)
                    else: 
                        current_input_ids = torch.cat([
                            current_input_ids.to(response_ids.device),
                            response_ids
                        ], dim=1)
                else:
                    break
        except Exception as e:
            print(f"[CoreEnv._run_single_rollout][{task_idx} @ {getattr(client, 'env_server_base', 'unknown_client')}] Error: {e}")
            print(traceback.format_exc())
            step_rewards = [] 
            final_reward = 0.0
            final_env_score = 0.0
            done = True

        return {
            'trajectory': trajectory,
            'step_rewards': step_rewards,
            'reward': final_reward,
            'env_score': final_env_score,
            'turns': turns,
            'valid_actions': len([msg for msg in trajectory if msg.get("from") == "gpt" and msg.get("value","").strip()]),
            'task_idx': task_idx,
            'done': done
        }

    def run_rollouts_parallel(self, gen_batch: DataProto) -> List[Dict[str, Any]]:
        initial_prompts_ids = gen_batch.batch['input_ids']
        batch_size = initial_prompts_ids.shape[0]
        num_clients = len(self.clients)

        if num_clients == 0:
            print("[CoreEnv.run_rollouts_parallel] Error: No environment clients available for rollout.")
            raise RuntimeError("No environment clients available for rollout.")
        if not self.executor:
            print("[CoreEnv.run_rollouts_parallel] Error: ThreadPoolExecutor is not initialized.")
            raise RuntimeError("ThreadPoolExecutor is not initialized. Check client and worker configuration.")

        print(f"[CoreEnv.run_rollouts_parallel] Starting rollout for batch size: {batch_size} using {num_clients} clients and {self.executor._max_workers} workers.")

        futures = {}
        rollout_results_list = [None] * batch_size

        for i in range(batch_size):
            task_idx = i
            initial_prompt = initial_prompts_ids[i:i+1]
            client_index = i % num_clients
            selected_client = self.clients[client_index]
            future = self.executor.submit(self._run_single_rollout, initial_prompt, task_idx, selected_client)
            futures[future] = i

        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            original_index = futures[future]
            try:
                result_dict = future.result()
                rollout_results_list[original_index] = result_dict
                completed_count += 1
            except Exception as e:
                print(f"[CoreEnv.run_rollouts_parallel] Error collecting result for batch index {original_index}: {e}")
                print(traceback.format_exc())
                rollout_results_list[original_index] = {
                    'trajectory': [], 'step_rewards': [], 'reward': 0.0,
                    'env_score': 0.0, 'turns': 0, 'valid_actions': 0,
                    'task_idx': original_index, 'done': True, 'error': str(e)
                }
        
        print(f"[CoreEnv.run_rollouts_parallel] Collected results from {completed_count}/{batch_size} rollouts.")
        valid_results = [res for res in rollout_results_list if res is not None] 
        if not valid_results and batch_size > 0:
             print("[CoreEnv.run_rollouts_parallel] Warning: No valid rollout results collected, though tasks were submitted.")
             return []
        return valid_results

    def convert_rollout_results_to_dataproto(self, results: List[Dict[str, Any]], original_batch: DataProto) -> DataProto:
        batch_input_ids = []
        batch_attention_mask = []
        batch_position_ids = []
        batch_info_mask = []
        batch_token_level_rewards = []
        batch_responses = []

        final_meta_info = {}
        if hasattr(original_batch, 'meta_info') and original_batch.meta_info:
            for k, v_meta in original_batch.meta_info.items(): 
                final_meta_info[k] = v_meta

        per_rollout_task_idx = []
        per_rollout_turns_stats = []
        per_rollout_valid_action_stats = []
        per_rollout_done_flags = []
        per_rollout_valid_search_stats = [] 
        per_rollout_rewards = []
        per_rollout_env_scores = []
        per_rollout_trajectories = []

        reward_allocation = "last_token"
        gamma_discount = 1.0 

        if self.config.algorithm_config and hasattr(self.config.algorithm_config, 'get'):
            reward_allocation = self.config.algorithm_config.get('reward_allocation', 'last_token')
            gamma_discount = self.config.algorithm_config.get('gamma', 1.0)
        print(f"[CoreEnv.convert_rollout] Using reward allocation: {reward_allocation}, gamma: {gamma_discount}")

        original_indices = final_meta_info.get('idx', list(range(len(results)))) 
        if isinstance(original_indices, torch.Tensor):
            original_indices = original_indices.tolist()
        
        if len(results) > len(original_indices):
            original_indices.extend(list(range(len(original_indices), len(results))))
        original_indices_map = {idx_val: i for i, idx_val in enumerate(original_indices)}

        print(f"[CoreEnv.convert_rollout] Formatting {len(results)} trajectories.")
        for result_idx, result_dict in enumerate(results):
            trajectory = result_dict.get('trajectory', [])
            reward_to_distribute = result_dict.get('env_score', 0.0) 

            turns = result_dict.get('turns', 0)
            task_idx = result_dict.get('task_idx', result_idx) 
            valid_actions_count = result_dict.get('valid_actions', 0)
            done_flag = result_dict.get('done', True)
            reward_val = result_dict.get('reward', 0.0)
            env_score_val = result_dict.get('env_score', 0.0)

            per_rollout_task_idx.append(task_idx)
            per_rollout_turns_stats.append(turns)
            per_rollout_valid_action_stats.append(valid_actions_count)
            per_rollout_done_flags.append(done_flag)
            per_rollout_valid_search_stats.append(0) 
            per_rollout_rewards.append(reward_val)
            per_rollout_env_scores.append(env_score_val)
            per_rollout_trajectories.append(trajectory) 

            original_batch_idx = original_indices_map.get(task_idx, result_idx)
            
            if not (0 <= original_batch_idx < original_batch.batch['input_ids'].shape[0]):
                if original_batch.batch['input_ids'].shape[0] > 0 :
                    original_batch_idx = result_idx % original_batch.batch['input_ids'].shape[0]
                else: 
                    print(f"[CoreEnv.convert_rollout] Warning: original_batch is empty. Cannot get initial_prompt_ids for task_idx {task_idx}. Skipping.")
                    continue

            conversation_ids_list = []
            info_mask_parts = []
            segment_lengths = []
            agent_response_segments_indices_in_conv = [] 

            if not trajectory:
                initial_prompt_ids_for_empty = original_batch.batch['input_ids'][original_batch_idx:original_batch_idx+1]
                conversation_ids_list.append(initial_prompt_ids_for_empty)
                info_mask_parts.append(torch.ones_like(initial_prompt_ids_for_empty, dtype=torch.long))
                segment_lengths.append(initial_prompt_ids_for_empty.shape[1])
            else:
                for turn_idx, msg in enumerate(trajectory):
                    msg_text = msg.get("value", "")
                    msg_from = msg.get("from", "")
                    if not msg_text.strip() and msg_from != "gpt": continue 

                    msg_ids = self.tokenizer(msg_text, add_special_tokens=False, return_tensors='pt')['input_ids']
                    if msg_ids.shape[1] == 0 and msg_from != "gpt": continue 

                    conversation_ids_list.append(msg_ids)
                    segment_lengths.append(msg_ids.shape[1])

                    if msg_from == "gpt":
                        info_mask_parts.append(torch.ones_like(msg_ids, dtype=torch.long))
                        agent_response_segments_indices_in_conv.append(len(conversation_ids_list) - 1)
                    elif msg_from == "env":
                        info_mask_parts.append(torch.zeros_like(msg_ids, dtype=torch.long))
                    else: 
                        info_mask_parts.append(torch.ones_like(msg_ids, dtype=torch.long))
            
            if not conversation_ids_list:
                print(f"[CoreEnv.convert_rollout] Warning: No valid conversation segments for task_idx {task_idx}. Skipping.")
                continue

            full_input_ids = torch.cat(conversation_ids_list, dim=1)
            full_info_mask = torch.cat(info_mask_parts, dim=1)
            seq_len = full_input_ids.shape[1]
            target_len = self.config.max_prompt_length
            
            current_segments = list(segment_lengths) 
            current_agent_indices_in_conv = list(agent_response_segments_indices_in_conv) 

            if seq_len > target_len: 
                truncated_len = 0
                num_segments_to_remove = 0
                for i in range(len(current_segments)):
                    if truncated_len + current_segments[i] >= seq_len - target_len:
                        to_remove_from_this_segment = (seq_len - target_len) - truncated_len
                        current_segments[i] -= to_remove_from_this_segment
                        if current_segments[i] == 0:
                            num_segments_to_remove = i + 1
                        truncated_len += to_remove_from_this_segment
                        break
                    truncated_len += current_segments[i]
                    num_segments_to_remove = i + 1
                
                full_input_ids = full_input_ids[:, -target_len:]
                full_info_mask = full_info_mask[:, -target_len:]
                current_segments = current_segments[num_segments_to_remove:]
                current_agent_indices_in_conv = [idx - num_segments_to_remove for idx in current_agent_indices_in_conv if idx >= num_segments_to_remove]
                seq_len = target_len
                padding_len = 0

            elif seq_len < target_len: 
                padding_len = target_len - seq_len
                pad_tensor = torch.full((1, padding_len), self.tokenizer.pad_token_id, dtype=torch.long, device=full_input_ids.device)
                full_input_ids = torch.cat([pad_tensor, full_input_ids], dim=1)
                info_pad = torch.zeros_like(pad_tensor, dtype=torch.long) 
                full_info_mask = torch.cat([info_pad, full_info_mask], dim=1)
            else:
                padding_len = 0


            agent_indices_in_padded = []
            current_token_idx = padding_len 
            for segment_idx, length in enumerate(current_segments):
                 is_agent_response = segment_idx in current_agent_indices_in_conv
                 start_idx = current_token_idx
                 end_idx = current_token_idx + length - 1
                 if is_agent_response and length > 0: 
                      agent_indices_in_padded.append((start_idx, end_idx))
                 current_token_idx += length

            token_level_rewards = torch.zeros_like(full_input_ids, dtype=torch.float32)
            if agent_indices_in_padded: 
                if reward_allocation == "last_token":
                    _ , last_segment_end = agent_indices_in_padded[-1]
                    if last_segment_end < target_len: 
                        token_level_rewards[0, last_segment_end] = reward_to_distribute
                elif reward_allocation == "uniform_positive":
                    if reward_to_distribute > 0:
                        total_agent_tokens = sum(end - start + 1 for start, end in agent_indices_in_padded)
                        reward_per_token = reward_to_distribute / max(1, total_agent_tokens)
                        for start, end in agent_indices_in_padded:
                            token_level_rewards[0, start : end + 1] = reward_per_token
                    elif reward_to_distribute < 0: 
                         _ , last_segment_end = agent_indices_in_padded[-1]
                         if last_segment_end < target_len:
                              token_level_rewards[0, last_segment_end] = reward_to_distribute
                elif reward_allocation == "discounted":
                    current_reward_val = reward_to_distribute
                    for start, end in reversed(agent_indices_in_padded): 
                        segment_len = end - start + 1
                        reward_for_segment_token = current_reward_val / segment_len 
                        token_level_rewards[0, start : end + 1] = reward_for_segment_token
                        current_reward_val *= (gamma_discount ** segment_len)


            full_attention_mask = self.tensor_fn.create_attention_mask(full_input_ids)
            full_position_ids = self.tensor_fn.create_position_ids(full_attention_mask)

            batch_input_ids.append(full_input_ids)
            batch_attention_mask.append(full_attention_mask)
            batch_position_ids.append(full_position_ids)
            batch_info_mask.append(full_info_mask)
            batch_token_level_rewards.append(token_level_rewards)

            response_segments_ids = [full_input_ids[0, r_start : r_end + 1] for r_start, r_end in agent_indices_in_padded if r_start <= r_end and r_end < full_input_ids.shape[1]]
            configured_resp_len = self.config.max_response_length
            if response_segments_ids:
                response_only_ids_cat = torch.cat(response_segments_ids, dim=0).unsqueeze(0)
                total_response_len = response_only_ids_cat.shape[1]
                if total_response_len > configured_resp_len:
                    response_only_ids_padded = response_only_ids_cat[:, :configured_resp_len]
                elif total_response_len < configured_resp_len:
                    resp_pad_len = configured_resp_len - total_response_len
                    resp_pad = torch.full((1, resp_pad_len), self.tokenizer.pad_token_id, dtype=torch.long, device=response_only_ids_cat.device)
                    response_only_ids_padded = torch.cat([response_only_ids_cat, resp_pad], dim=1)
                else:
                    response_only_ids_padded = response_only_ids_cat
            else:
                response_only_ids_padded = torch.full((1, configured_resp_len), self.tokenizer.pad_token_id, dtype=torch.long, device=full_input_ids.device)
            batch_responses.append(response_only_ids_padded)

        if not batch_input_ids:
            print("[CoreEnv.convert_rollout] No valid trajectories after processing. Returning empty DataProto.")
            empty_proto = DataProto.from_dict({
                "input_ids": torch.empty((0, self.config.max_prompt_length), dtype=torch.long), 
                "attention_mask": torch.empty((0, self.config.max_prompt_length), dtype=torch.long),
                "position_ids": torch.empty((0, self.config.max_prompt_length), dtype=torch.long), 
                "info_mask": torch.empty((0, self.config.max_prompt_length), dtype=torch.long),
                "token_level_rewards": torch.empty((0, self.config.max_prompt_length), dtype=torch.float), 
                "responses": torch.empty((0, self.config.max_response_length), dtype=torch.long)
            })
            empty_proto.meta_info = {'micro_batch_size': getattr(self.actor_rollout_wg, 'log_prob_micro_batch_size', 1)} 
            return empty_proto

        final_batch = {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "position_ids": torch.cat(batch_position_ids, dim=0),
            "info_mask": torch.cat(batch_info_mask, dim=0),
            "token_level_rewards": torch.cat(batch_token_level_rewards, dim=0),
            "responses": torch.cat(batch_responses, dim=0)
        }
        data_proto = DataProto.from_dict(final_batch)

        num_processed_trajectories = len(batch_input_ids)
        final_meta_info['task_idx'] = torch.tensor(per_rollout_task_idx[:num_processed_trajectories], dtype=torch.long)
        final_meta_info['turns_stats'] = torch.tensor(per_rollout_turns_stats[:num_processed_trajectories], dtype=torch.long)
        final_meta_info['valid_action_stats'] = torch.tensor(per_rollout_valid_action_stats[:num_processed_trajectories], dtype=torch.long)
        final_meta_info['valid_search_stats'] = torch.tensor(per_rollout_valid_search_stats[:num_processed_trajectories], dtype=torch.long)
        final_meta_info['active_mask'] = torch.tensor([not done for done in per_rollout_done_flags[:num_processed_trajectories]], dtype=torch.bool)
        final_meta_info['reward'] = torch.tensor(per_rollout_rewards[:num_processed_trajectories], dtype=torch.float32)
        final_meta_info['env_score'] = torch.tensor(per_rollout_env_scores[:num_processed_trajectories], dtype=torch.float32)
        final_meta_info['rollout_trajectory'] = per_rollout_trajectories[:num_processed_trajectories]
        
        final_meta_info['idx'] = final_meta_info['task_idx'] 

        log_prob_mbs = 1 
        if hasattr(self.actor_rollout_wg, 'log_prob_micro_batch_size'):
            log_prob_mbs = self.actor_rollout_wg.log_prob_micro_batch_size
        elif hasattr(self.config, 'actor_rollout_ref') and hasattr(self.config.actor_rollout_ref, 'rollout') and hasattr(self.config.actor_rollout_ref.rollout, 'log_prob_micro_batch_size'):
             log_prob_mbs = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size
        
        final_meta_info['micro_batch_size'] = log_prob_mbs
        final_meta_info['temperature'] = getattr(self.config, 'temperature', 1.0) 
        final_meta_info['use_dynamic_bsz'] = getattr(self.config, 'log_prob_use_dynamic_bsz', False)
        if final_meta_info.get('use_dynamic_bsz', False):
            final_meta_info['max_token_len'] = getattr(self.config, 'log_prob_max_token_len_per_gpu', 2048)

        data_proto.meta_info = final_meta_info
        print(f"[CoreEnv.convert_rollout] Final batch shapes: input_ids={final_batch['input_ids'].shape}, responses={final_batch['responses'].shape}, tlr={final_batch['token_level_rewards'].shape}")
        return data_proto

```
