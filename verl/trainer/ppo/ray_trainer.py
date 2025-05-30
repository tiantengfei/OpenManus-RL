# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from openmanus_rl.llm_agent.openmanus import OpenManusAgent, AgentConfig

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss # Ensure agg_loss is imported
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
# Removed import of reduce_metrics from verl.utils.metric
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.reward_score import SUPPORTED_REWARD_SCORE_FNS
from verl.workers.rollout.async_server import AsyncLLMServerManager

WorkerType = Type[Worker]

KNOWN_AGENTGYM_ENVS = [
    "webshop", "webarena", "maze", "wordle", "alfworld",
    "sciworld", "babyai", "textcraft", "weather", "movie",
    "academia", "todo", "sheet", "sqlgym", "openmanus_local"
]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}
    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last

# Added local definition of reduce_metrics
def reduce_metrics(metrics_dict: dict):
    '''Reduces a dict of metrics by averaging lists of values.'''
    reduced = {}
    for k, v in metrics_dict.items():
        if isinstance(v, list) and v:
            try:
                reduced[k] = np.mean([float(x) for x in v]) # Attempt to convert to float before mean
            except (ValueError, TypeError):
                print(f"[reduce_metrics] Warning: Could not compute mean for key '{k}' with value {v}. Skipping.")
                reduced[k] = v # Keep original if reduction fails
        elif isinstance(v, (int, float)):
            reduced[k] = v # Already a scalar
        else:
            # If it's a tensor or other type, try to convert to a common format or keep as is
            try:
                if hasattr(v, 'item'): # For torch tensors
                    reduced[k] = v.item()
                elif isinstance(v, np.ndarray) and v.size == 1:
                    reduced[k] = v.item()
                else:
                    reduced[k] = v # Keep as is if no clear reduction path
            except Exception as e:
                print(f"[reduce_metrics] Warning: Error processing key '{k}' value {v}: {e}. Skipping.")
                reduced[k] = v
    return reduced

class RayPPOTrainer:
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup, 
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO, AdvantageEstimator.GRPO_PASSK, AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX, AdvantageEstimator.RLOO, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_loss_mask(self, batch, metrics):
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask
        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        return batch, metrics

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size", "critic": "micro_batch_size",
                "reward_model": "micro_batch_size", "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }
            if name in settings:
                param = settings[name]; param_per_gpu = f"{param}_per_gpu"
                if mbs is None and mbs_per_gpu is None: raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")
                if mbs is not None and mbs_per_gpu is not None: raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported.")
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size, config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, "actor_rollout_ref.actor")
            if self.use_reference_policy: check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size, config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu, "actor_rollout_ref.ref")
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size, config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu, "actor_rollout_ref.rollout")
        if self.use_critic and not config.critic.use_dynamic_bsz: check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz: check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus
        assert config.actor_rollout_ref.actor.loss_agg_mode in ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"
        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss: print("NOTICE: You have both enabled in-reward kl and kl loss.")
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
        if self.use_critic and config.critic.strategy == "fsdp" and config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
            assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."
        if config.data.get("val_batch_size", None) is not None: print("WARNING: val_batch_size is deprecated. Validation datasets are sent as a whole batch.")
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample: assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature > 0 for do_sample"
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set for multi_turn with tool"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"
        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        if train_dataset is None: train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None: val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        if train_sampler is None: train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None: from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn; collate_fn = default_collate_fn
        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset, batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size), num_workers=self.config.data.get("dataloader_num_workers", 8), drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
        val_batch_size = self.config.data.val_batch_size if self.config.data.val_batch_size is not None else len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(dataset=self.val_dataset, batch_size=val_batch_size, num_workers=self.config.data.get("dataloader_num_workers", 8), shuffle=False, drop_last=False, collate_fn=collate_fn)
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None: total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"): self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"): self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e: print(f"Warning: Could not set total_training_steps in config. Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
        n = len(inputs)
        base_data = {"input": inputs, "output": outputs, "score": scores, "step": [self.global_steps] * n}
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n: base_data[k] = v
        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0: return
        import numpy as np
        samples = list(zip(inputs, outputs, scores)); samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42); rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        if self.config.data.env_name in KNOWN_AGENTGYM_ENVS:
            print(f"[Trainer] Detected AgentGym environment ({self.config.data.env_name}), using OpenManusAgent for validation.")
            agent_config = AgentConfig(
                env_name=self.config.data.env_name, max_turns=self.config.data.get("max_turns", 10),
                max_dialog_size=self.config.data.get("max_dialog_size", 20480), max_start_length=self.config.data.max_prompt_length,
                max_token_length=self.config.actor_rollout_ref.rollout.max_new_tokens, num_val_episodes=self.config.data.get("num_val_episodes", 10),
                agent_type=self.config.data.get("agent_type", "default_agent_type"), seed=self.config.data.seed,
                output_dir=self.config.data.get("output_dir", "outputs"), debug=self.config.data.get("debug", False),
                temperature=self.config.actor_rollout_ref.rollout.temperature, top_p=self.config.actor_rollout_ref.rollout.top_p,
                n=self.config.actor_rollout_ref.rollout.n, presence_penalty=self.config.actor_rollout_ref.rollout.get("presence_penalty", 0.0),
                frequency_penalty=self.config.actor_rollout_ref.rollout.get("frequency_penalty", 0.0), stop_token_ids=self.config.actor_rollout_ref.rollout.stop_token_ids,
                multi_turn_tool_config_path=self.config.actor_rollout_ref.rollout.multi_turn.tool_config_path,
                multi_turn_max_tool_steps=self.config.actor_rollout_ref.rollout.multi_turn.max_tool_steps
            )
            if not hasattr(self, 'log_dir'): self.log_dir = self.config.trainer.get("default_local_dir", "./verl_checkpoints/default_log_dir"); print(f"Set self.log_dir in _validate (AgentGym path): {self.log_dir}")
            if not hasattr(self, 'logger') or self.logger is None:
                from verl.utils.tracking import Tracking; print("Warning: self.logger not found in _validate (AgentGym). Initializing."); self.logger = Tracking(project_name=self.config.trainer.project_name, experiment_name=self.config.trainer.experiment_name, default_backend=self.config.trainer.logger, config=OmegaConf.to_container(self.config, resolve=True))
            self.validation_agent = OpenManusAgent(tokenizer=self.tokenizer, actor_rollout_wg=self.actor_rollout_wg, config=agent_config, is_validation=True, logger=self.logger)
            all_metrics = defaultdict(list); num_episodes_processed = 0; trajectories = []
            print(f"Starting AgentGym validation loop. Num val episodes to run: {agent_config.num_val_episodes}")
            for _ in range(agent_config.num_val_episodes):
                if hasattr(self.validation_agent, 'reset_env'): self.validation_agent.reset_env()
                processed_info = self.validation_agent.run_llm_loop(log_dir=self.log_dir, global_steps=self.global_steps)
                if 'env_score' in processed_info: all_metrics['env_score'].append(processed_info['env_score'])
                if 'turns' in processed_info: all_metrics['turns'].append(processed_info['turns'])
                if 'valid_action_ratio' in processed_info: all_metrics['valid_action_ratio'].append(processed_info['valid_action_ratio'])
                if 'tool_calls' in processed_info: all_metrics['tool_calls'].append(processed_info['tool_calls'])
                if 'trajectory' in processed_info: trajectories.append(processed_info['trajectory'])
                num_episodes_processed +=1
            print(f"AgentGym validation loop finished. Processed {num_episodes_processed} episodes.")
            if self.config.data.get("save_trajectory_video", False) and trajectories:
                from openmanus_rl.utils.visualization import save_trajectory_to_output
                output_video_dir = os.path.join(self.log_dir, "videos", str(self.global_steps)); os.makedirs(output_video_dir, exist_ok=True)
                for i, trajectory in enumerate(trajectories): save_trajectory_to_output(trajectory=trajectory, output_path=os.path.join(output_video_dir, f"trajectory_{i}.mp4"), env_name=self.config.data.env_name)
                print(f"Saved {len(trajectories)} trajectory videos to {output_video_dir}")
            final_metrics = {}
            if all_metrics['env_score']: final_metrics["val/env_score_mean"] = np.mean(all_metrics['env_score']); final_metrics["val/env_score_std"] = np.std(all_metrics['env_score'])
            if all_metrics['turns']: final_metrics["val/turns_mean"] = np.mean(all_metrics['turns']); final_metrics["val/turns_std"] = np.std(all_metrics['turns'])
            if all_metrics['valid_action_ratio']: final_metrics["val/valid_action_ratio_mean"] = np.mean(all_metrics['valid_action_ratio'])
            if all_metrics['tool_calls']: final_metrics["val/tool_calls_mean"] = np.mean(all_metrics['tool_calls'])
            if final_metrics: print(f"AgentGym validation metrics (step {self.global_steps}): {final_metrics}"); self.logger.log(data=final_metrics, step=self.global_steps)
            else: print("No AgentGym validation metrics collected.")
            return final_metrics if final_metrics else {}
        else:
            data_source_lst = []; reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []; sample_outputs = []; sample_scores = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model": return {}
            input_ids = test_batch.batch["input_ids"]; input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]; sample_inputs.extend(input_texts)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]; non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch: non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch: non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch: non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(batch_keys=batch_keys_to_pop, non_tensor_batch_keys=non_tensor_batch_keys_to_pop)
            test_gen_batch.meta_info = {"eos_token_id": self.tokenizer.eos_token_id, "pad_token_id": self.tokenizer.pad_token_id, "recompute_log_prob": False, "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample, "validate": True}
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode: test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else: self.async_rollout_manager.wake_up(); test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded); self.async_rollout_manager.sleep()
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size); print("validation generation end")
            output_ids = test_output_gen_batch.batch["responses"]; output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]; sample_outputs.extend(output_texts)
            test_batch = test_batch.union(test_output_gen_batch)
            result = self.val_reward_fn(test_batch, return_dict=True); reward_tensor = result["reward_tensor"]; scores = reward_tensor.sum(-1).cpu().tolist(); sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items(): reward_extra_infos_dict[key].extend(lst)
            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir: self._dump_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, reward_extra_infos_dict=reward_extra_infos_dict, dump_path=val_data_dir)
        for key_info, lst in reward_extra_infos_dict.items(): assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
        data_sources = np.concatenate(data_source_lst, axis=0)
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    metric_sec = "val-core" if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name) else "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"; metric_dict[pfx] = metric_val
        return metric_dict

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.actor_rollout_ref, role="actor_rollout")
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else: raise NotImplementedError
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
        all_wg = {}; wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None: wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys()); all_wg.update(spawn_wg)
        if self.use_critic: self.critic_wg = all_wg["critic"]; self.critic_wg.init_model()
        if self.use_reference_policy and not self.ref_in_actor: self.ref_policy_wg = all_wg["ref"]; self.ref_policy_wg.init_model()
        if self.use_rm: self.rm_wg = all_wg["rm"]; self.rm_wg.init_model()
        self.actor_rollout_wg = all_wg["actor_rollout"]; self.actor_rollout_wg.init_model()
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(config=self.config.actor_rollout_ref, worker_group=self.actor_rollout_wg)

    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save: print("Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local_path)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f: f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable": return 0
        if self.config.trainer.default_hdfs_dir is not None: raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder): checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None: print("Training from scratch"); return 0
        else: # resume_mode == "resume_path"
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder): global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}"); print(f"Resuming from {global_step_folder}")
        actor_path = os.path.join(global_step_folder, "actor"); critic_path = os.path.join(global_step_folder, "critic")
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        if self.use_critic: self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path): self.train_dataloader.load_state_dict(torch.load(dataloader_local_path, weights_only=False))
        else: print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        attention_mask = batch.batch["attention_mask"]; batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        metrics.update(log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix))

    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        # import os, uuid, torch, numpy as np (already at module level)
        # from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto (already at module level)
        # from verl.trainer.ppo.core_algos import agg_loss (already at module level)

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.log_dir = self.config.trainer.get("default_local_dir", "./verl_checkpoints/default_log_dir")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[Trainer.fit] Log directory set to: {self.log_dir}")
        print(f"[Trainer.fit] Logger initialized: {self.logger}")

        self.global_steps = 0
        self._load_checkpoint()

        self.is_agentgym_run = self.config.data.env_name in KNOWN_AGENTGYM_ENVS
        generation_manager = None
        if self.is_agentgym_run:
            print(f"[Trainer.fit] Initializing OpenManusAgent for AgentGym: {self.config.data.env_name}")
            agent_config_params = {
                'max_turns': self.config.data.get("max_turns", 10),
                'max_start_length': self.config.data.max_prompt_length, # Should be data.max_prompt_length based on old code
                'max_prompt_length': self.config.data.max_prompt_length,
                'max_response_length': self.config.data.max_response_length,
                'max_obs_length': self.config.data.max_obs_length,
                'num_gpus': self.config.trainer.n_gpus_per_node,
                'env_name': self.config.data.env_name,
                'env_ports': self.config.data.env_ports,
                'env_server_base': self.config.data.env_server_base,
                'env_data_len': self.config.data.get('env_data_len', 200),
                'max_workers': self.config.actor_rollout_ref.rollout.get('max_workers', 10),
                'algorithm_config': self.config.algorithm,
                'react_format': self.config.data.get('react_format', True),
                'rollout_strategy': self.config.data.get('rollout_strategy', "StandardReAct"),
                'max_dialog_size': self.config.data.get("max_dialog_size", 20480), 
                'max_token_length': self.config.actor_rollout_ref.rollout.max_new_tokens, 
                'agent_type': self.config.data.get("agent_type", "default_agent_type"), 
                'seed': self.config.data.seed, 
                'output_dir': self.config.data.get("output_dir", "outputs"), 
                'debug': self.config.data.get("debug", False), 
                'temperature': self.config.actor_rollout_ref.rollout.temperature, 
                'top_p': self.config.actor_rollout_ref.rollout.top_p, 
                'n': self.config.actor_rollout_ref.rollout.n, 
                'presence_penalty': self.config.actor_rollout_ref.rollout.get("presence_penalty", 0.0), 
                'frequency_penalty': self.config.actor_rollout_ref.rollout.get("frequency_penalty", 0.0), 
                'stop_token_ids': self.config.actor_rollout_ref.rollout.stop_token_ids, 
                'multi_turn_tool_config_path': self.config.actor_rollout_ref.rollout.multi_turn.tool_config_path, 
                'multi_turn_max_tool_steps': self.config.actor_rollout_ref.rollout.multi_turn.max_tool_steps 
            }
            gen_config = AgentConfig(**agent_config_params)
            generation_manager = OpenManusAgent(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=gen_config,
                is_validation=False,
                logger=self.logger
            )
            print(f"[Trainer.fit] OpenManusAgent for training created.")

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics: 
                pprint(f"Initial validation metrics: {val_metrics}")
                self.logger.log(data=val_metrics, step=self.global_steps)
            else:
                pprint(f"Initial validation returned no metrics (step {self.global_steps}).")
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1 
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                metrics = {}
                timing_raw = {}
                batch_prompts_initial: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch_prompts_initial.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch_prompts_initial.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch_prompts_initial.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch_prompts_initial.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                if 'idx' not in gen_batch.meta_info and 'input_ids' in gen_batch.batch:
                     gen_batch.meta_info['idx'] = torch.arange(gen_batch.batch['input_ids'].shape[0])
                if 'reward_model' not in gen_batch.meta_info and 'input_ids' in gen_batch.batch:
                     gen_batch.meta_info['reward_model'] = [{} for _ in range(gen_batch.batch['input_ids'].shape[0])]

                is_last_step = self.global_steps >= self.total_training_steps
                batch_after_generation = None 
                gen_data_output = None 

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        if self.is_agentgym_run:
                            output_dir_agentgym = os.path.join(self.log_dir, f"train_step_{self.global_steps}")
                            os.makedirs(output_dir_agentgym, exist_ok=True)
                            batch_from_agent = generation_manager.run_llm_loop(
                                gen_batch=gen_batch, output_dir=output_dir_agentgym, global_steps=self.global_steps
                            )
                            actor_rollout_world_size = self.actor_rollout_wg.world_size
                            padded_batch_for_logp, pad_size_logp = pad_dataproto_to_divisor(
                                batch_from_agent, actor_rollout_world_size
                            )
                            logp_mbs_config = self.config.actor_rollout_ref.rollout
                            logp_mbs = logp_mbs_config.get('log_prob_micro_batch_size_per_gpu', logp_mbs_config.log_prob_micro_batch_size)
                            use_dyn_bsz = logp_mbs_config.get('log_prob_use_dynamic_bsz', False)
                            temperature = logp_mbs_config.temperature
                            padded_batch_for_logp.meta_info['micro_batch_size'] = logp_mbs
                            padded_batch_for_logp.meta_info['use_dynamic_bsz'] = use_dyn_bsz
                            padded_batch_for_logp.meta_info['temperature'] = temperature
                            if use_dyn_bsz:
                                max_token_len_logp = logp_mbs_config.get('log_prob_max_token_len_per_gpu', self.config.data.max_prompt_length)
                                padded_batch_for_logp.meta_info['max_token_len'] = max_token_len_logp
                            else:
                                padded_batch_for_logp.meta_info.pop('max_token_len', None)
                            output_logp_padded = self.actor_rollout_wg.compute_log_prob(padded_batch_for_logp)
                            output_logp = unpad_dataproto(output_logp_padded, pad_size=pad_size_logp)
                            batch_after_generation = batch_from_agent.union(output_logp)
                            if 'token_level_rewards' in batch_after_generation.batch: batch_after_generation.batch['token_level_scores'] = batch_after_generation.batch['token_level_rewards'].clone()
                            elif 'input_ids' in batch_after_generation.batch: batch_after_generation.batch['token_level_scores'] = torch.zeros_like(batch_after_generation.batch['input_ids'], dtype=torch.float)
                            
                            current_batch_size = 0
                            if 'input_ids' in batch_after_generation.batch: current_batch_size = batch_after_generation.batch['input_ids'].shape[0]
                            elif 'idx' in batch_after_generation.meta_info and isinstance(batch_after_generation.meta_info['idx'], torch.Tensor): current_batch_size = batch_after_generation.meta_info['idx'].shape[0]
                            
                            uid_list = []
                            if 'idx' in batch_after_generation.meta_info and isinstance(batch_after_generation.meta_info['idx'], torch.Tensor): uid_list = batch_after_generation.meta_info['idx'].cpu().tolist()
                            elif current_batch_size > 0 : uid_list = [str(uuid.uuid4()) for _ in range(current_batch_size)]
                            batch_after_generation.non_tensor_batch['uid'] = np.array(uid_list, dtype=object)
                        else: # Standard Verl generation
                            if not self.async_rollout_mode: gen_data_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            else: self.async_rollout_manager.wake_up(); gen_data_output = self.async_rollout_manager.generate_sequences(gen_batch); self.async_rollout_manager.sleep()

                    if not self.is_agentgym_run:
                        repeated_batch_prompts = batch_prompts_initial.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch_after_generation = repeated_batch_prompts.union(gen_data_output) # gen_data_output is from the non-AgentGym path
                        current_batch_size_nga = 0
                        if 'input_ids' in batch_after_generation.batch: # Should be from repeated_batch_prompts
                             current_batch_size_nga = batch_after_generation.batch['input_ids'].shape[0]
                        uid_list_nga = [str(uuid.uuid4()) for _ in range(current_batch_size_nga)]
                        batch_after_generation.non_tensor_batch['uid'] = np.array(uid_list_nga, dtype=object)
                    elif self.is_agentgym_run and self.config.actor_rollout_ref.rollout.n > 1: # AgentGym path, but repetition is configured
                        batch_after_generation = batch_after_generation.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    
                    if batch_after_generation and "responses" in batch_after_generation.batch and "attention_mask" in batch_after_generation.batch:
                         batch_after_generation.batch["response_mask"] = compute_response_mask(batch_after_generation)
                    else:
                         print(f"[Trainer.fit][WARN] Step {self.global_steps}: Cannot compute 'response_mask'.")

                    if self.config.trainer.balance_batch and batch_after_generation and 'attention_mask' in batch_after_generation.batch:
                        self._balance_batch(batch_after_generation, metrics=metrics)
                    if batch_after_generation and 'attention_mask' in batch_after_generation.batch:
                        batch_after_generation.meta_info["global_token_num"] = torch.sum(batch_after_generation.batch["attention_mask"], dim=-1).tolist()

                    if not self.is_agentgym_run:
                        with _timer("old_log_prob", timing_raw):
                            if 'responses' not in batch_after_generation.batch and gen_data_output and 'input_ids' in gen_data_output.batch:
                                batch_after_generation.batch['responses'] = gen_data_output.batch['input_ids']
                            if 'responses' in batch_after_generation.batch :
                                old_log_prob_output = self.actor_rollout_wg.compute_log_prob(batch_after_generation)
                                if "entropys" in old_log_prob_output.batch and "response_mask" in batch_after_generation.batch:
                                    entropys = old_log_prob_output.batch["entropys"]
                                    response_masks_for_entropy = batch_after_generation.batch["response_mask"]
                                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                    entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks_for_entropy, loss_agg_mode=loss_agg_mode)
                                    metrics.update({"actor/entropy_loss": entropy_loss.detach().item()})
                                    old_log_prob_output.batch.pop("entropys")
                                batch_after_generation = batch_after_generation.union(old_log_prob_output)
                    
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch_after_generation)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch_after_generation)
                            batch_after_generation = batch_after_generation.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch_after_generation)
                            batch_after_generation = batch_after_generation.union(values)
                    
                    reward_extra_infos_dict_local = {} 
                    with _timer("adv", timing_raw):
                        if self.use_rm and not self.is_agentgym_run:
                            rm_score_tensor = self.rm_wg.compute_rm_score(batch_after_generation)
                            batch_after_generation = batch_after_generation.union(rm_score_tensor)
                        if not self.is_agentgym_run:
                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch_after_generation, self.config, self.tokenizer)
                                reward_tensor_computed, reward_extra_infos_dict_local = ray.get(future_reward)
                            else:
                                 reward_tensor_computed, reward_extra_infos_dict_local = compute_reward(batch_after_generation, self.reward_fn)
                            batch_after_generation.batch["token_level_scores"] = reward_tensor_computed
                        if reward_extra_infos_dict_local:
                            batch_after_generation.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict_local.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            if 'ref_log_prob' in batch_after_generation.batch and \
                               'old_log_probs' in batch_after_generation.batch and \
                               'token_level_scores' in batch_after_generation.batch and \
                               'responses' in batch_after_generation.batch and \
                               ('loss_mask' in batch_after_generation.batch or 'attention_mask' in batch_after_generation.batch): 
                                batch_after_generation, kl_metrics = apply_kl_penalty(
                                    batch_after_generation, kl_ctrl=self.kl_ctrl_in_reward, 
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable
                                )
                                metrics.update(kl_metrics)
                            elif 'token_level_scores' in batch_after_generation.batch: # Fallback if KL cannot be computed
                                batch_after_generation.batch["token_level_rewards"] = batch_after_generation.batch["token_level_scores"].clone()
                        elif 'token_level_scores' in batch_after_generation.batch :
                            batch_after_generation.batch["token_level_rewards"] = batch_after_generation.batch["token_level_scores"].clone()
                        
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch_after_generation = compute_advantage(
                            batch_after_generation,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch_after_generation)
                        if hasattr(critic_output, 'meta_info') and 'metrics' in critic_output.meta_info:
                             metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            if self.is_agentgym_run and self.config.actor_rollout_ref.actor.get('state_masking', False):
                                if 'info_mask' in batch_after_generation.batch and \
                                   'responses' in batch_after_generation.batch and \
                                   'attention_mask' in batch_after_generation.batch:
                                    batch_after_generation, actor_metrics_loss_mask = self._create_loss_mask(batch_after_generation, {})
                                    metrics.update(actor_metrics_loss_mask)
                            batch_after_generation.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch_after_generation)
                        if hasattr(actor_output, 'meta_info') and 'metrics' in actor_output.meta_info:
                            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                    
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            prompts_for_dump = batch_after_generation.batch.get("prompts", batch_prompts_initial.batch.get("prompts")) 
                            if prompts_for_dump is not None and \
                               all(key in batch_after_generation.batch for key in ["responses", "token_level_scores"]):
                                inputs_decoded = self.tokenizer.batch_decode(prompts_for_dump, skip_special_tokens=True)
                                outputs_decoded = self.tokenizer.batch_decode(batch_after_generation.batch["responses"], skip_special_tokens=True)
                                scores_summed = batch_after_generation.batch["token_level_scores"].sum(-1).cpu().tolist()
                                self._dump_generations(
                                    inputs=inputs_decoded, outputs=outputs_decoded, scores=scores_summed,
                                    reward_extra_infos_dict=reward_extra_infos_dict_local, 
                                    dump_path=rollout_data_dir,
                                )

                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                       (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics_step: dict = self._validate()
                            if is_last_step: last_val_metrics = val_metrics_step
                        if val_metrics_step: metrics.update(val_metrics_step)

                    if self.config.trainer.save_freq > 0 and \
                       (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                if batch_after_generation and "responses" in batch_after_generation.batch: 
                    metrics.update(compute_data_metrics(batch=batch_after_generation, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch_after_generation, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch_after_generation, timing_raw=timing_raw, n_gpus=n_gpus))
                
                self.logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    if last_val_metrics: pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
