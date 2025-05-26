import os

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, 'version/version')) as f:
    version = f.read().strip()

# Move the DataProto import to the end or make it conditional
def __getattr__(name):
    if name == 'DataProto':
        from .protocol import DataProto
        return DataProto
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from .utils.logging_utils import set_basic_config
import logging

set_basic_config(level=logging.WARNING)
