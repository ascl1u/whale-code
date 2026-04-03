"""Tool definitions for native LLM tool calling."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_commands",
            "description": "Execute commands in the terminal with your analysis and plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "string",
                        "description": (
                            "Analyze the current state based on the terminal output. "
                            "What has been accomplished? What still needs to be done?"
                        ),
                    },
                    "plan": {
                        "type": "string",
                        "description": (
                            "Describe your plan for the next steps. "
                            "What commands will you run and why?"
                        ),
                    },
                    "commands": {
                        "type": "array",
                        "description": "Commands to execute. Can be empty to wait without acting.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "keystrokes": {
                                    "type": "string",
                                    "description": (
                                        "Exact keystrokes to send to the terminal, used verbatim. "
                                        "Most bash commands should end with \\n to execute. "
                                        "For special keys use tmux-style escapes: C-c for Ctrl+C, C-d for Ctrl+D."
                                    ),
                                },
                                "duration": {
                                    "type": "number",
                                    "description": (
                                        "Seconds to wait for completion. "
                                        "Immediate tasks (cd, ls, echo): 0.1s. "
                                        "Normal commands (gcc, find): 1.0s. "
                                        "Slow commands (make, wget): set as needed. "
                                        "Never exceed 60s — poll intermediate status instead."
                                    ),
                                },
                            },
                            "required": ["keystrokes"],
                        },
                    },
                },
                "required": ["analysis", "plan", "commands"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Signal that the task is complete.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a bounded slice of a file from disk via the environment (not tmux scrollback). "
                "Use for source files, logs, configs when you need exact text. "
                "Path should be absolute when possible (e.g. /app/...)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read.",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Max bytes to read (default 32768, max 262144).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_read",
            "description": (
                "Read and analyze an image file visually. "
                "Use ONLY for image files (PNG, JPG, GIF, WEBP). "
                "Do NOT use for text files — use cat/head instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the image file.",
                    },
                    "image_read_instruction": {
                        "type": "string",
                        "description": "What information to extract from the image. Be specific.",
                    },
                },
                "required": ["file_path", "image_read_instruction"],
            },
        },
    },
]
