from dataclasses import dataclass
from typing import List, Any

@dataclass
class OpenManusEnvConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    env_name: str
    env_ports: List[int]
    env_server_base: str
    react_format: bool = True
    env_data_len: int = 200
    max_workers: int = 10
    algorithm_config: Any = None
