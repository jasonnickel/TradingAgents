"""LLM client backed by locally installed Claude Code or Codex CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable

from .base_client import BaseLLMClient
from .validators import validate_model


def _emit_timing(event: str, **fields: Any) -> None:
    """Emit a single [TIMING] line to stderr for the verdict-agent runner to capture.

    Stderr is the right channel because: (a) stdout is reserved for the runner's
    final JSON payload, and (b) the parent verdict_agent.py writes stderr to a
    per-ticker artifacts log so we can audit pacing after a successful run, not
    only on failure. Format is single-line key=value so a quick `grep TIMING`
    over the log reproduces the timeline.
    """
    parts = [f"[TIMING] {event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    sys.stderr.write(" ".join(parts) + "\n")
    sys.stderr.flush()


def _schema_for_model(schema: Any) -> dict[str, Any]:
    if hasattr(schema, "model_json_schema"):
        raw_schema = schema.model_json_schema()
    else:
        raw_schema = schema.schema()
    return _codex_compatible_schema(raw_schema)


def _codex_compatible_schema(schema: Any) -> Any:
    """Codex CLI requires object schemas to explicitly forbid extra fields."""
    if isinstance(schema, dict):
        converted = {
            key: _codex_compatible_schema(value)
            for key, value in schema.items()
        }
        if converted.get("type") == "object":
            converted.setdefault("additionalProperties", False)
        return converted
    if isinstance(schema, list):
        return [_codex_compatible_schema(item) for item in schema]
    return schema


def _parse_structured_json(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


class CLIChatModel(Runnable[Any, AIMessage]):
    """Small LangChain-like adapter for non-interactive local agent CLIs.

    Subclasses :class:`langchain_core.runnables.Runnable` so the standard
    ``prompt | llm.bind_tools(...)`` pipe pattern composes correctly in
    LangGraph chains. Without that base, the pipe operator rejects the
    instance with "unsupported type" even though it implements ``invoke``.
    """

    is_cli_llm = True

    def __init__(
        self,
        provider: str,
        model: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
        effort: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model
        self.cwd = cwd or os.getcwd()
        self.timeout = timeout or 600
        self.effort = effort

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        prompt = self._input_to_prompt(input)
        start = time.monotonic()
        _emit_timing(
            "invoke_start",
            provider=self.provider,
            model=self.model,
            effort=self.effort or "default",
            prompt_chars=len(prompt),
        )
        try:
            content = self._run_cli(prompt)
        finally:
            _emit_timing(
                "invoke_end",
                provider=self.provider,
                model=self.model,
                elapsed_s=f"{time.monotonic() - start:.2f}",
            )
        return AIMessage(content=content)

    def bind_tools(self, tools: Any) -> "CLIChatModel":
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "CLIStructuredChatModel":
        return CLIStructuredChatModel(self, schema)

    def _input_to_prompt(self, input_: Any) -> str:
        if isinstance(input_, str):
            return input_

        if hasattr(input_, "to_messages"):
            return self._messages_to_prompt(input_.to_messages())

        if isinstance(input_, list):
            return self._messages_to_prompt(input_)

        return str(input_)

    def _messages_to_prompt(self, messages: list[Any]) -> str:
        parts = []
        for message in messages:
            if isinstance(message, BaseMessage):
                role = message.type
                content = message.content
            elif isinstance(message, dict):
                role = message.get("role", "message")
                content = message.get("content", "")
            else:
                role = "message"
                content = str(message)
            parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(parts)

    def _run_cli(self, prompt: str, schema: Optional[dict[str, Any]] = None) -> str:
        if self.provider == "claude_cli":
            return self._run_claude(prompt, schema)
        if self.provider == "codex_cli":
            return self._run_codex(prompt, schema)
        raise ValueError(f"Unsupported CLI provider: {self.provider}")

    def _run_claude(self, prompt: str, schema: Optional[dict[str, Any]]) -> str:
        cmd = [
            "claude",
            "--print",
            "--model",
            self.model,
            "--output-format",
            "text",
            "--no-session-persistence",
        ]
        if self.effort:
            cmd.extend(["--effort", self.effort])
        if schema is not None:
            cmd.extend(["--json-schema", json.dumps(schema)])
            prompt = (
                "Return only valid JSON that satisfies the supplied schema. "
                "Do not include markdown fences or explanatory text.\n\n"
                + prompt
            )
        cmd.append(prompt)
        return self._run_command(cmd)

    def _run_codex(self, prompt: str, schema: Optional[dict[str, Any]]) -> str:
        cmd = [
            "codex",
            "exec",
            "--cd",
            self.cwd,
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--color",
            "never",
            "--model",
            self.model,
        ]
        temp_paths: list[str] = []
        if schema is not None:
            schema_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            )
            with schema_file:
                json.dump(schema, schema_file)
            output_file = tempfile.NamedTemporaryFile(delete=False)
            output_file.close()
            temp_paths.extend([schema_file.name, output_file.name])
            cmd.extend(["--output-schema", schema_file.name])
            cmd.extend(["--output-last-message", output_file.name])
            prompt = (
                "Return only valid JSON that satisfies the output schema. "
                "Do not include markdown fences or explanatory text.\n\n"
                + prompt
            )
            cmd.append("-")
            self._run_command(cmd, input_text=prompt)
            try:
                return Path(output_file.name).read_text(encoding="utf-8")
            finally:
                for path in temp_paths:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        cmd.append("-")
        return self._run_command(cmd, input_text=prompt)

    def _run_command(self, cmd: list[str], input_text: Optional[str] = None) -> str:
        bin_name = cmd[0] if cmd else "?"
        start = time.monotonic()
        _emit_timing("subprocess_start", bin=bin_name, model=self.model)
        try:
            result = subprocess.run(
                cmd,
                cwd=self.cwd,
                input=input_text,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            _emit_timing(
                "subprocess_timeout",
                bin=bin_name,
                elapsed_s=f"{time.monotonic() - start:.2f}",
                timeout_s=self.timeout,
            )
            raise
        elapsed = time.monotonic() - start
        _emit_timing(
            "subprocess_end",
            bin=bin_name,
            rc=result.returncode,
            elapsed_s=f"{elapsed:.2f}",
            stdout_chars=len(result.stdout or ""),
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"{cmd[0]} exited with status {result.returncode}: {stderr}"
            )
        return result.stdout.strip()


class CLIStructuredChatModel:
    def __init__(self, llm: CLIChatModel, schema: Any):
        self.llm = llm
        self.schema = schema

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        prompt = self.llm._input_to_prompt(input)
        raw = self.llm._run_cli(prompt, _schema_for_model(self.schema))
        data = _parse_structured_json(raw)
        if hasattr(self.schema, "model_validate"):
            return self.schema.model_validate(data)
        return self.schema.parse_obj(data)


class CLIClient(BaseLLMClient):
    """Client factory for Claude Code and Codex command-line providers."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "claude_cli",
        **kwargs: Any,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        return CLIChatModel(
            provider=self.provider,
            model=self.model,
            cwd=self.kwargs.get("cwd") or os.getcwd(),
            timeout=self.kwargs.get("timeout"),
            effort=self.kwargs.get("effort"),
        )

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)
