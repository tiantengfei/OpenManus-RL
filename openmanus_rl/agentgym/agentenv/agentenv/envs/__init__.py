from .academia import AcademiaEnvClient, AcademiaTask
from .alfworld import AlfWorldEnvClient, AlfWorldTask
from .babyai import BabyAIEnvClient, BabyAITask
from .lmrlgym import MazeEnvClient, MazeTask, WordleEnvClient, WordleTask
from .movie import MovieEnvClient, MovieTask
from .openmanus_env import OpenManusTask # Added
from .sciworld import SciworldEnvClient, SciworldTask
from .sheet import SheetEnvClient, SheetTask
from .sqlgym import SqlGymEnvClient, SqlGymTask
from .textcraft import TextCraftEnvClient, TextCraftTask
from .todo import TodoEnvClient, TodoTask
from .weather import WeatherEnvClient, WeatherTask
from .webarena import WebarenaEnvClient, WebarenaTask
from .webshop import WebshopEnvClient, WebshopTask

__all__ = [
    "AcademiaTask",
    "AlfWorldTask",
    "BabyAITask",
    "MazeTask",
    "WordleTask",
    "MovieTask",
    "OpenManusTask", # Added
    "SciworldTask",
    "SheetTask",
    "SqlGymTask",
    "TextCraftTask",
    "TodoTask",
    "WeatherTask",
    "WebarenaTask",
    "WebshopTask",
]
