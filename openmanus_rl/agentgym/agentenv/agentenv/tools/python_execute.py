# openmanus_rl/agentgym/agentenv/agentenv/tools/python_execute.py
import asyncio
import multiprocessing
import sys
from io import StringIO
from typing import Dict, Any

from .base import BaseTool

# Helper function to run code in a separate process, adapted from original
# This function itself cannot be async if it's the target of multiprocessing.Process
def _run_code_in_process(code: str, result_dict: Dict[str, Any], safe_globals: Dict[str, Any]) -> None:
    original_stdout = sys.stdout
    output_buffer = StringIO()
    sys.stdout = output_buffer
    try:
        exec(code, safe_globals, safe_globals) # Using safe_globals for both globals and locals
        result_dict["output"] = output_buffer.getvalue()
        result_dict["success"] = True
    except Exception as e:
        result_dict["output"] = str(e)
        result_dict["success"] = False
    finally:
        sys.stdout = original_stdout
        output_buffer.close()

class PythonExecute(BaseTool):
    name: str = "python_execute"
    description: str = (
        "Executes Python code string in a separate process with a timeout. "
        "Note: Only print outputs are visible; function return values are not directly captured. "
        "Use print statements to see results. The code runs in a restricted environment."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds. Default is 5 seconds.",
                "default": 5,
            },
        },
        "required": ["code"],
    }

    async def execute(self, code: str, timeout: int = 5) -> str:
        """
        Executes the provided Python code with a timeout in a separate process.
        Returns a string representing the captured stdout or an error message.
        """
        
        # Prepare safe globals for exec.
        # This is a basic attempt at sandboxing. For more robust sandboxing,
        # a more sophisticated approach would be needed (e.g., restricted __builtins__).
        # For now, provide a clean dict for __builtins__.
        safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "True": True,
            "False": False,
            "None": None,
            # Common and relatively safe builtins. Add more as needed.
            # Avoid adding things like 'open', 'eval', 'exec', 'importlib', 'os', 'sys' directly
            # unless the execution environment is fully trusted or further sandboxed.
        }
        safe_globals = {"__builtins__": safe_builtins}

        # Multiprocessing manager is context-managed
        with multiprocessing.Manager() as manager:
            result_dict = manager.dict({"output": "", "success": False})
            
            process = multiprocessing.Process(
                target=_run_code_in_process, args=(code, result_dict, safe_globals)
            )
            
            process.start()
            # Wait for the process to finish or timeout
            # Since process.join() is blocking, run it in an executor to keep execute() async
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, process.join, timeout)
            except Exception as e: # Should not happen with process.join
                 process.terminate() # Ensure termination on unexpected error
                 process.join(1) 
                 return f"Error during process execution management: {str(e)}"


            if process.is_alive():
                process.terminate() # Terminate if still alive (timed out)
                process.join(1)   # Wait for termination to complete
                # Check if result_dict was populated by a racing condition before timeout
                # This is unlikely with how join works, but as a safeguard:
                if not result_dict["output"]: # If output is empty, it truly timed out
                    return f"Execution timed out after {timeout} seconds."
                # If there's output, it might be a partial result before timeout kill, or error during setup
                # For simplicity, if it timed out and we killed it, we assume timeout message is primary.
                # However, if an error occurred *during* the timeout and was caught by _run_code_in_process,
                # that error might be in result_dict['output'].
                # This logic prioritizes the timeout message if the process had to be killed.
                return f"Execution timed out after {timeout} seconds. Partial output/error (if any): {result_dict['output']}"


            # Process finished normally or was terminated and joined
            if result_dict.get("success", False):
                return f"Execution successful:\n{result_dict['output']}"
            else:
                return f"Execution failed:\n{result_dict['output']}"
