"""Tests for built-in coding tools and tool registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from skillengine.tools import (
    ToolRegistry,
    ToolDefinition,
    ReadTool,
    WriteTool,
    EditTool,
    create_coding_tools,
    create_read_only_tools,
    create_all_tools,
)


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_creation(self) -> None:
        """Should create a ToolDefinition with all fields."""
        td = ToolDefinition(
            name="test",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

        assert td.name == "test"
        assert td.description == "A test tool"
        assert td.parameters == {"type": "object", "properties": {}}
        assert td.handler is None

    def test_creation_with_handler(self) -> None:
        """Should accept a handler callable."""

        async def my_handler(args: dict) -> str:
            return "ok"

        td = ToolDefinition(
            name="test",
            description="desc",
            parameters={},
            handler=my_handler,
        )

        assert td.handler is my_handler


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self) -> None:
        """Should register a tool and retrieve it by name."""
        registry = ToolRegistry()
        tool = ToolDefinition(name="read", description="Read files", parameters={})

        registry.register(tool)
        result = registry.get("read")

        assert result is not None
        assert result.name == "read"
        assert result.description == "Read files"

    def test_get_nonexistent(self) -> None:
        """Should return None for a tool that does not exist."""
        registry = ToolRegistry()

        assert registry.get("nonexistent") is None

    def test_unregister(self) -> None:
        """Should unregister a tool and return True."""
        registry = ToolRegistry()
        tool = ToolDefinition(name="write", description="Write files", parameters={})
        registry.register(tool)

        result = registry.unregister("write")

        assert result is True
        assert registry.get("write") is None

    def test_unregister_nonexistent(self) -> None:
        """Should return False when unregistering a tool that does not exist."""
        registry = ToolRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_list_tools(self) -> None:
        """Should list all registered tools."""
        registry = ToolRegistry()
        tool_a = ToolDefinition(name="a", description="Tool A", parameters={})
        tool_b = ToolDefinition(name="b", description="Tool B", parameters={})
        registry.register(tool_a)
        registry.register(tool_b)

        tools = registry.list_tools()

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"a", "b"}

    def test_list_tools_empty(self) -> None:
        """Should return empty list when no tools registered."""
        registry = ToolRegistry()

        assert registry.list_tools() == []

    def test_get_definitions_format(self) -> None:
        """Should return tool definitions in OpenAI function calling format."""
        registry = ToolRegistry()
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        )
        registry.register(tool)

        definitions = registry.get_definitions()

        assert len(definitions) == 1
        defn = definitions[0]
        assert defn["type"] == "function"
        assert defn["function"]["name"] == "test_tool"
        assert defn["function"]["description"] == "A test tool"
        assert defn["function"]["parameters"]["type"] == "object"

    def test_register_overwrites(self) -> None:
        """Should overwrite a tool with the same name."""
        registry = ToolRegistry()
        tool_v1 = ToolDefinition(name="tool", description="v1", parameters={})
        tool_v2 = ToolDefinition(name="tool", description="v2", parameters={})
        registry.register(tool_v1)
        registry.register(tool_v2)

        result = registry.get("tool")

        assert result is not None
        assert result.description == "v2"
        assert len(registry.list_tools()) == 1


class TestCreateTools:
    """Tests for tool factory functions."""

    def test_create_coding_tools_returns_four(self) -> None:
        """Should return 4 coding tools: read, write, edit, bash."""
        tools = create_coding_tools()

        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"read", "write", "edit", "bash"}

    def test_create_coding_tools_are_tool_definitions(self) -> None:
        """Should return ToolDefinition instances."""
        tools = create_coding_tools()

        for tool in tools:
            assert isinstance(tool, ToolDefinition)
            assert tool.handler is not None

    def test_create_read_only_tools_returns_four(self) -> None:
        """Should return 4 read-only tools: read, grep, find, ls."""
        tools = create_read_only_tools()

        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"read", "grep", "find", "ls"}

    def test_create_all_tools_returns_seven(self) -> None:
        """Should return all 7 built-in tools."""
        tools = create_all_tools()

        assert len(tools) == 7
        assert set(tools.keys()) == {"read", "write", "edit", "bash", "grep", "find", "ls"}

    def test_create_all_tools_returns_dict(self) -> None:
        """Should return a dict mapping name to ToolDefinition."""
        tools = create_all_tools()

        assert isinstance(tools, dict)
        for name, defn in tools.items():
            assert isinstance(defn, ToolDefinition)
            assert defn.name == name

    def test_create_tools_with_cwd(self, tmp_path: Path) -> None:
        """Should pass cwd through to tools."""
        tools = create_coding_tools(cwd=str(tmp_path))

        assert len(tools) == 4
        for tool in tools:
            assert isinstance(tool, ToolDefinition)


class TestReadTool:
    """Tests for ReadTool."""

    def test_name_and_description(self) -> None:
        """Should have correct name and description."""
        tool = ReadTool()

        assert tool.name == "read"
        assert "Read" in tool.description or "read" in tool.description.lower()

    def test_parameters_schema(self) -> None:
        """Should have correct parameter schema."""
        tool = ReadTool()
        params = tool.parameters

        assert params["type"] == "object"
        assert "file_path" in params["properties"]
        assert params["required"] == ["file_path"]

    def test_definition(self) -> None:
        """Should produce a valid ToolDefinition."""
        tool = ReadTool()
        defn = tool.definition()

        assert isinstance(defn, ToolDefinition)
        assert defn.name == "read"
        assert defn.handler is not None

    async def test_execute_reads_file(self, tmp_path: Path) -> None:
        """Should read a text file and include line numbers."""
        test_file = tmp_path / "hello.txt"
        test_file.write_text("line one\nline two\nline three\n")

        tool = ReadTool(cwd=str(tmp_path))
        result = await tool.execute({"file_path": str(test_file)})

        assert "line one" in result
        assert "line two" in result
        assert "line three" in result
        # Line numbers should be present
        assert "1" in result
        assert "2" in result
        assert "3" in result

    async def test_execute_file_not_found(self, tmp_path: Path) -> None:
        """Should return an error when file does not exist."""
        tool = ReadTool(cwd=str(tmp_path))
        result = await tool.execute({"file_path": str(tmp_path / "missing.txt")})

        assert "Error" in result
        assert "not found" in result

    async def test_execute_directory_error(self, tmp_path: Path) -> None:
        """Should return an error when path is a directory."""
        tool = ReadTool(cwd=str(tmp_path))
        result = await tool.execute({"file_path": str(tmp_path)})

        assert "Error" in result
        assert "directory" in result

    async def test_execute_with_line_range(self, tmp_path: Path) -> None:
        """Should read a specific line range with offset and limit."""
        test_file = tmp_path / "lines.txt"
        lines = [f"line {i}" for i in range(1, 11)]
        test_file.write_text("\n".join(lines) + "\n")

        tool = ReadTool(cwd=str(tmp_path))
        result = await tool.execute({
            "file_path": str(test_file),
            "offset": 3,
            "limit": 2,
        })

        assert "line 3" in result
        assert "line 4" in result
        # Should indicate partial read
        assert "Showing lines" in result

    async def test_execute_empty_file(self, tmp_path: Path) -> None:
        """Should handle empty files."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        tool = ReadTool(cwd=str(tmp_path))
        result = await tool.execute({"file_path": str(test_file)})

        assert "empty" in result.lower() or "0 lines" in result

    async def test_execute_relative_path(self, tmp_path: Path) -> None:
        """Should resolve relative paths using cwd."""
        test_file = tmp_path / "relative.txt"
        test_file.write_text("relative content\n")

        tool = ReadTool(cwd=str(tmp_path))
        result = await tool.execute({"file_path": "relative.txt"})

        assert "relative content" in result

    async def test_execute_missing_file_path(self) -> None:
        """Should return error when file_path is empty."""
        tool = ReadTool()
        result = await tool.execute({"file_path": ""})

        assert "Error" in result


class TestEditTool:
    """Tests for EditTool."""

    def test_name_and_description(self) -> None:
        """Should have correct name and description."""
        tool = EditTool()

        assert tool.name == "edit"
        assert "replace" in tool.description.lower() or "replacement" in tool.description.lower()

    async def test_execute_replacement(self, tmp_path: Path) -> None:
        """Should replace a unique string in a file."""
        test_file = tmp_path / "code.py"
        test_file.write_text("def hello():\n    return 'hello'\n")

        tool = EditTool(cwd=str(tmp_path))
        result = await tool.execute({
            "file_path": str(test_file),
            "old_string": "return 'hello'",
            "new_string": "return 'world'",
        })

        assert "Edited" in result
        assert "1 replacement" in result
        updated = test_file.read_text()
        assert "return 'world'" in updated
        assert "return 'hello'" not in updated

    async def test_execute_not_found_error(self, tmp_path: Path) -> None:
        """Should return error when old_string not found in file."""
        test_file = tmp_path / "code.py"
        test_file.write_text("def hello():\n    pass\n")

        tool = EditTool(cwd=str(tmp_path))
        result = await tool.execute({
            "file_path": str(test_file),
            "old_string": "nonexistent string",
            "new_string": "replacement",
        })

        assert "Error" in result
        assert "not found" in result

    async def test_execute_non_unique_error(self, tmp_path: Path) -> None:
        """Should return error when old_string appears multiple times."""
        test_file = tmp_path / "code.py"
        test_file.write_text("x = 1\nx = 1\n")

        tool = EditTool(cwd=str(tmp_path))
        result = await tool.execute({
            "file_path": str(test_file),
            "old_string": "x = 1",
            "new_string": "x = 2",
        })

        assert "Error" in result
        assert "2 times" in result
        # File should be unchanged
        assert test_file.read_text() == "x = 1\nx = 1\n"

    async def test_execute_replace_all(self, tmp_path: Path) -> None:
        """Should replace all occurrences when replace_all is True."""
        test_file = tmp_path / "code.py"
        test_file.write_text("x = 1\nx = 1\n")

        tool = EditTool(cwd=str(tmp_path))
        result = await tool.execute({
            "file_path": str(test_file),
            "old_string": "x = 1",
            "new_string": "x = 2",
            "replace_all": True,
        })

        assert "Edited" in result
        assert "2 replacements" in result
        assert test_file.read_text() == "x = 2\nx = 2\n"

    async def test_execute_file_not_found(self, tmp_path: Path) -> None:
        """Should return error when the target file does not exist."""
        tool = EditTool(cwd=str(tmp_path))
        result = await tool.execute({
            "file_path": str(tmp_path / "missing.py"),
            "old_string": "a",
            "new_string": "b",
        })

        assert "Error" in result
        assert "not found" in result

    async def test_execute_same_string_error(self) -> None:
        """Should return error when old_string equals new_string."""
        tool = EditTool()
        result = await tool.execute({
            "file_path": "/tmp/any.py",
            "old_string": "same",
            "new_string": "same",
        })

        assert "Error" in result
        assert "different" in result


class TestWriteTool:
    """Tests for WriteTool."""

    def test_name_and_description(self) -> None:
        """Should have correct name and description."""
        tool = WriteTool()

        assert tool.name == "write"
        assert "write" in tool.description.lower() or "Write" in tool.description

    async def test_execute_creates_file(self, tmp_path: Path) -> None:
        """Should create a new file with the given content."""
        tool = WriteTool(cwd=str(tmp_path))
        file_path = str(tmp_path / "new_file.txt")
        result = await tool.execute({
            "file_path": file_path,
            "content": "Hello, world!\n",
        })

        assert "Created" in result
        created_file = Path(file_path)
        assert created_file.exists()
        assert created_file.read_text() == "Hello, world!\n"

    async def test_execute_overwrites_file(self, tmp_path: Path) -> None:
        """Should overwrite an existing file."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("old content")

        tool = WriteTool(cwd=str(tmp_path))
        result = await tool.execute({
            "file_path": str(test_file),
            "content": "new content",
        })

        assert "Wrote" in result or "overwrote" in result.lower()
        assert test_file.read_text() == "new content"

    async def test_execute_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Should create parent directories automatically."""
        tool = WriteTool(cwd=str(tmp_path))
        nested_path = str(tmp_path / "a" / "b" / "c" / "file.txt")
        result = await tool.execute({
            "file_path": nested_path,
            "content": "nested content\n",
        })

        assert "Created" in result
        assert Path(nested_path).exists()
        assert Path(nested_path).read_text() == "nested content\n"

    async def test_execute_empty_path_error(self) -> None:
        """Should return error when file_path is empty."""
        tool = WriteTool()
        result = await tool.execute({"file_path": "", "content": "data"})

        assert "Error" in result

    async def test_execute_reports_line_count(self, tmp_path: Path) -> None:
        """Should report line count in the result."""
        tool = WriteTool(cwd=str(tmp_path))
        result = await tool.execute({
            "file_path": str(tmp_path / "lines.txt"),
            "content": "line1\nline2\nline3\n",
        })

        assert "3 lines" in result
