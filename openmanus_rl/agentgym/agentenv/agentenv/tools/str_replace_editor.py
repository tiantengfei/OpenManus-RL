# openmanus_rl/agentgym/agentenv/agentenv/tools/str_replace_editor.py
import os
import shutil
from pathlib import Path
from typing import Any, DefaultDict, List, Literal, Optional, get_args, Union
from collections import defaultdict

from .base import BaseTool

# Type alias for Path-like objects
PathLike = Union[str, Path]

# Simplified FileOperator for local operations.
# In a full implementation, this could be expanded or use a proper sandbox.
class LocalFileOperator:
    async def exists(self, path: PathLike) -> bool:
        return Path(path).exists()

    async def is_directory(self, path: PathLike) -> bool:
        return Path(path).is_dir()

    async def read_file(self, path: PathLike) -> str:
        try:
            return Path(path).read_text(encoding='utf-8')
        except Exception as e:
            raise ToolError(f"Error reading file {path}: {e}")

    async def write_file(self, path: PathLike, content: str) -> None:
        try:
            Path(path).write_text(content, encoding='utf-8')
        except Exception as e:
            raise ToolError(f"Error writing to file {path}: {e}")

    async def run_command(self, command: str) -> tuple[int, str, str]:
        # This is a simplified and potentially unsafe way to run commands.
        # For 'find' as used in original, it might be okay, but needs care.
        # Consider replacing with more Pythonic directory traversal if possible.
        # For now, replicating the idea for 'view directory'.
        import subprocess
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            returncode = process.returncode
            return returncode, stdout.decode('utf-8'), stderr.decode('utf-8')
        except Exception as e:
            return 1, "", str(e)

# Custom Exception for Tool Errors (can be defined in a common place later)
class ToolError(Exception):
    pass

Command = Literal["view", "create", "str_replace", "insert", "undo_edit"]

SNIPPET_LINES: int = 4
MAX_RESPONSE_LEN: int = 16000 # Max length of file content to return
TRUNCATED_MESSAGE: str = (
    "<response clipped>\n<NOTE>To save on context only part of this file has been shown to you. "
    "Consider using 'view' with a line range for targeted viewing.</NOTE>"
)

def maybe_truncate(content: str, truncate_after: Optional[int] = MAX_RESPONSE_LEN) -> str:
    if not truncate_after or len(content) <= truncate_after:
        return content
    return content[:truncate_after] + "\n" + TRUNCATED_MESSAGE

class StrReplaceEditor(BaseTool):
    name: str = "str_replace_editor"
    description: str = (
        "Tool for viewing, creating, and editing files. "
        "Supports string replacement, line insertion, and undoing edits. "
        "Operates on absolute paths."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "Allowed commands: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                "type": "string",
            },
            "path": {"description": "Absolute path to file or directory.", "type": "string"},
            "file_text": {"description": "Content for `create` command.", "type": "string"},
            "old_str": {"description": "String to replace for `str_replace`.", "type": "string"},
            "new_str": {"description": "New string for `str_replace` or `insert`.", "type": "string"},
            "insert_line": {"description": "Line number (1-indexed) AFTER which to insert for `insert` command.", "type": "integer"},
            "view_range": {
                "description": "Optional [start_line, end_line] for `view` file command. Use -1 for end_line to view till end.",
                "type": "array", "items": {"type": "integer"},
            },
        },
        "required": ["command", "path"],
    }

    def __init__(self, workspace_root: Optional[PathLike] = None):
        super().__init__()
        # TODO: Workspace root needs to be configurable for the agent/environment
        # For now, allow all absolute paths, or restrict to a default if needed.
        # self.workspace_root = Path(workspace_root if workspace_root else Path.cwd()).resolve()
        # print(f"[StrReplaceEditor] Initialized. Workspace (not strictly enforced yet): {self.workspace_root}")
        
        self._file_history: DefaultDict[Path, List[str]] = defaultdict(list)
        self._operator: LocalFileOperator = LocalFileOperator() # Using local operator for now

    # Helper to ensure path is within a workspace (currently not strictly enforced)
    async def _validate_path_within_workspace(self, path: Path) -> Path:
        # resolved_path = path.resolve()
        # if self.workspace_root not in resolved_path.parents and resolved_path != self.workspace_root:
        #     raise ToolError(f"Path {path} is outside the allowed workspace {self.workspace_root}")
        # return resolved_path
        if not path.is_absolute(): # From original validation
             raise ToolError(f"The path {path} is not an absolute path. Current working directory is {Path.cwd()}")
        return path # For now, just ensure it's absolute

    async def _validate_path_for_command(self, command: str, path: Path) -> None:
        if command != "create":
            if not await self._operator.exists(path):
                raise ToolError(f"Path {path} does not exist.")
            is_dir = await self._operator.is_directory(path)
            if is_dir and command not in ["view"]: # Only view can operate on existing dirs
                raise ToolError(f"Path {path} is a directory. Command '{command}' cannot be used on directories.")
        elif command == "create":
            if await self._operator.exists(path):
                raise ToolError(f"File or directory already exists at: {path}. Cannot use `create`.")

    async def execute(
        self,
        command: Command,
        path: str,
        file_text: Optional[str] = None,
        view_range: Optional[List[int]] = None,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        insert_line: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        try:
            path_obj = Path(path)
            validated_path = await self._validate_path_within_workspace(path_obj)
            await self._validate_path_for_command(command, validated_path)

            if command == "view":
                return await self._view(validated_path, view_range)
            elif command == "create":
                if file_text is None:
                    raise ToolError("`file_text` is required for `create` command.")
                await self._operator.write_file(validated_path, file_text)
                self._file_history[validated_path].append(file_text) # Store initial content for undo
                return f"File created successfully at: {validated_path}"
            elif command == "str_replace":
                if old_str is None:
                    raise ToolError("`old_str` is required for `str_replace` command.")
                return await self._str_replace(validated_path, old_str, new_str if new_str is not None else "")
            elif command == "insert":
                if insert_line is None or new_str is None:
                    raise ToolError("`insert_line` and `new_str` are required for `insert` command.")
                return await self._insert(validated_path, insert_line, new_str)
            elif command == "undo_edit":
                return await self._undo_edit(validated_path)
            else:
                # Should be caught by Literal[Command] type hint and JSON schema, but as a safeguard:
                raise ToolError(f"Unknown command: {command}")
        except ToolError as e:
            return f"Error: {e}"
        except Exception as e:
            # Catch any other unexpected errors during tool execution
            return f"An unexpected error occurred: {str(e)}"

    def _format_output_with_lines(self, content: str, file_descriptor: str, start_line: int = 1) -> str:
        content = content.expandtabs() # Expand tabs for consistent numbering
        numbered_lines = [f"{i + start_line:6}\t{line}" for i, line in enumerate(content.splitlines())]
        formatted_content = "\n".join(numbered_lines)
        return f"Content of {file_descriptor}:\n{maybe_truncate(formatted_content)}\n"


    async def _view(self, path: Path, view_range: Optional[List[int]] = None) -> str:
        if await self._operator.is_directory(path):
            if view_range:
                raise ToolError("`view_range` is not allowed when viewing a directory.")
            # Simplified directory view for now, avoiding complex 'find' command from original
            # This lists immediate children. Original used 'find -maxdepth 2'.
            children = []
            for item in sorted(os.listdir(path)):
                if not item.startswith('.'): # exclude hidden files/dirs
                    children.append(item)
            output = f"Directory listing for {path}:\n" + "\n".join(children)
            return maybe_truncate(output)

        else: # It's a file
            file_content = await self._operator.read_file(path)
            lines = file_content.splitlines() # Keep newlines for accurate line count and rejoining
            
            start_idx = 0
            end_idx = len(lines)

            if view_range:
                if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                    raise ToolError("`view_range` must be a list of two integers [start_line, end_line].")
                
                req_start_line, req_end_line = view_range
                
                if req_start_line < 1:
                    raise ToolError(f"View range start_line ({req_start_line}) must be 1 or greater.")
                
                start_idx = req_start_line - 1

                if req_end_line == -1:
                    end_idx = len(lines)
                elif req_end_line < req_start_line:
                    raise ToolError(f"View range end_line ({req_end_line}) must be >= start_line ({req_start_line}) or -1.")
                else:
                    end_idx = req_end_line
                
                if start_idx >= len(lines) and len(lines) > 0 : # start_idx is 0-based, req_start_line is 1-based
                     raise ToolError(f"View range start_line ({req_start_line}) is beyond the end of file ({len(lines)} lines).")

                lines_to_show = lines[start_idx:end_idx]
            else:
                lines_to_show = lines
            
            content_to_format = "\n".join(lines_to_show)
            return self._format_output_with_lines(content_to_format, str(path), start_line=start_idx + 1)


    async def _str_replace(self, path: Path, old_str: str, new_str: str) -> str:
        file_content = await self._operator.read_file(path)
        # Original tool expanded tabs. For simplicity in porting, ensure consistency or make it a user choice.
        # Let's assume for now inputs and file content are used as-is regarding tabs.

        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ToolError(f"Pattern `{old_str}` not found in {path}.")
        if occurrences > 1:
            # Could enhance to show line numbers like original, but for now, simpler error.
            raise ToolError(f"Pattern `{old_str}` found multiple times in {path}. Replacement aborted for safety. Make `old_str` more specific.")

        self._file_history[path].append(file_content) # Save current content before modification
        
        new_file_content = file_content.replace(old_str, new_str, 1) # Replace only the first (which is unique)
        await self._operator.write_file(path, new_file_content)
        
        # Snippet logic (simplified)
        # Find line where replacement happened for snippet context
        # This is approximate; original had more complex snippet logic
        # For simplicity, just confirm edit. A more advanced snippet could be added later.
        return f"Successfully replaced content in {path}."


    async def _insert(self, path: Path, insert_line: int, new_text: str) -> str:
        file_content = await self._operator.read_file(path)
        lines = file_content.splitlines(True) # Keep line endings

        if not (0 <= insert_line <= len(lines)): # insert_line is 1-based for user, 0-based for list index (inserting *after* line means index)
                                                # Or 0 means before first line. Let's clarify: "AFTER the line insert_line"
                                                # If insert_line is 0, means insert at beginning.
                                                # If insert_line is N, means insert after Nth line (index N).
            raise ToolError(f"Insert line {insert_line} is out of range for file with {len(lines)} lines. Valid range: 0 to {len(lines)} (0 for start).")

        self._file_history[path].append(file_content)
        
        # Ensure new_text ends with a newline if it doesn't, to maintain line structure
        # if not new_text.endswith('\n') and insert_line < len(lines): # Avoid adding if inserting at very end as a new line
        #    new_text_to_insert = new_text + '\n'
        # else:
        #    new_text_to_insert = new_text
        # Simpler: let user control newlines in new_text. Split new_text into lines.
        new_text_lines = new_text.splitlines(True)


        new_lines = lines[:insert_line] + new_text_lines + lines[insert_line:]
        new_file_content = "".join(new_lines)
        
        await self._operator.write_file(path, new_file_content)
        return f"Successfully inserted text into {path} after line {insert_line}."

    async def _undo_edit(self, path: Path) -> str:
        if not self._file_history[path]:
            raise ToolError(f"No edit history for {path} to undo.")
        
        last_content = self._file_history[path].pop()
        await self._operator.write_file(path, last_content)
        return f"Last edit to {path} undone. File reverted to previous state."

# Need to import asyncio for LocalFileOperator.run_command
import asyncio
