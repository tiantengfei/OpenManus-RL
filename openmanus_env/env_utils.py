import importlib
import traceback
from typing import List, Any
from .config import OpenManusEnvConfig

ENV_TO_TASK_CLASS = {
    "academia": "AcademiaTask", "alfworld": "AlfWorldTask", "babyai": "BabyAITask",
    "maze": "MazeTask", "wordle": "WordleTask", "movie": "MovieTask",
    "sciworld": "SciworldTask", "sheet": "SheetTask", "sqlgym": "SqlGymTask",
    "textcraft": "TextCraftTask", "todo": "TodoTask", "weather": "WeatherTask",
    "webarena": "WebarenaTask", "webshop": "WebshopTask",
}

def initialize_env_clients(config: OpenManusEnvConfig) -> List[Any]:
    """Initializes and returns a list of environment clients."""
    clients = []
    env_name_lower = config.env_name.lower()

    if env_name_lower not in ENV_TO_TASK_CLASS:
        error_msg = f"Environment '{config.env_name}' not supported. Available environments: {list(ENV_TO_TASK_CLASS.keys())}"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    task_class_name = ENV_TO_TASK_CLASS[env_name_lower]
    print(f"[INFO] Attempting to import TaskClass: {task_class_name} for environment: {config.env_name}")

    try:
        module = importlib.import_module(f"agentenv.envs.{env_name_lower}.task")
        TaskClass = getattr(module, task_class_name)
        print(f"[INFO] Successfully imported {task_class_name} from agentenv.envs.{env_name_lower}.task")
    except ImportError:
        error_msg = f"ImportError: Could not import module agentenv.envs.{env_name_lower}.task"
        print(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
        raise ImportError(error_msg)
    except AttributeError:
        error_msg = f"AttributeError: Could not find class {task_class_name} in agentenv.envs.{env_name_lower}.task"
        print(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
        raise AttributeError(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred during TaskClass import: {e}"
        print(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
        raise

    for port in config.env_ports:
        server_url = f"{config.env_server_base}:{port}"
        client_args = {
            "server_url": server_url,
            "env_data_len": config.env_data_len,
            "max_obs_length": config.max_obs_length,
            "max_prompt_length": config.max_prompt_length,
            "max_response_length": config.max_response_length,
            "max_start_length": config.max_start_length,
            "max_turns": config.max_turns,
            "react_format": config.react_format,
        }
        try:
            print(f"[INFO] Initializing {task_class_name} with server_url: {server_url}")
            task_instance = TaskClass(client_args=client_args, n_clients=1)
            if task_instance.clients and len(task_instance.clients) > 0:
                client = task_instance.clients[0]
                clients.append(client)
                print(f"[INFO] Successfully initialized client for {server_url}")
            else:
                print(f"[WARNING] No clients found for {task_class_name} instance with server_url: {server_url}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize {task_class_name} for port {port}: {e}\n{traceback.format_exc()}")

    if not clients:
        error_msg = f"No clients were initialized for environment '{config.env_name}'. Please check server URLs and task initialization."
        print(f"[ERROR] {error_msg}")
        raise RuntimeError(error_msg)

    print(f"[INFO] Successfully initialized {len(clients)} clients for environment '{config.env_name}'.")
    return clients
