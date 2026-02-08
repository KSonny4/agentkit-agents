# Docker-Deployable Agentkit with Telegram I/O

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make agentkit Docker-deployable as a SaaS-like daemon. Each agent = text files (identity.md, tools.md, evaluation.md). Telegram is primary I/O. Claude Code CLI runs inside the container.

**Architecture:** Docker container (Node.js + Python + Claude CLI). Telegram bot receives messages → mailbox → agent processes via Claude CLI → result sent back. Endless loop. One container per agent (goodoracle, nanoclaw, game-monitor, etc).

**Tech Stack:** Python 3.11+, python-telegram-bot>=21.0, Claude Code CLI (npm), Docker

---

## Context

agentkit core exists and works (42 tests, ~478 LOC). This plan adds:
- **Telegram as core** (was a skill, now built-in — primary I/O channel)
- **Daemon mode** (endless Telegram polling loop)
- **Docker deployment** (one command: `docker-compose up`)
- **Agent = text files** (mount profiles/ as volume → agent is defined)

User deploys any agent with:
```bash
mkdir profiles/myagent/   # write identity.md, tools.md, evaluation.md
docker-compose up -d      # done
```

---

## Task 1: Add python-telegram-bot dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1:** Change `dependencies = []` to `dependencies = ["python-telegram-bot>=21.0"]`

**Step 2:** Run: `source .venv/bin/activate && pip install -e . && python -c "import telegram; print(telegram.__version__)"`

**Step 3:** Commit: `git add pyproject.toml && git commit -m "feat: add python-telegram-bot dependency"`

---

## Task 2: Create telegram_bot.py + tests (TDD)

**Files:**
- Create: `agentkit/telegram_bot.py`
- Create: `tests/test_telegram.py`

**Step 1: Write tests**

```python
# tests/test_telegram.py
"""Tests for Telegram integration."""

from unittest.mock import AsyncMock, patch

from agentkit.telegram_bot import TelegramBot, MAX_MESSAGE_LENGTH


def test_telegram_bot_init():
    bot = TelegramBot("fake-token", "fake-chat-id")
    assert bot.token == "fake-token"
    assert bot.chat_id == "fake-chat-id"


def test_truncate_short_message():
    assert TelegramBot._truncate("hello") == "hello"


def test_truncate_exact_limit():
    msg = "x" * MAX_MESSAGE_LENGTH
    assert TelegramBot._truncate(msg) == msg


def test_truncate_long_message():
    result = TelegramBot._truncate("x" * 5000)
    assert len(result) <= MAX_MESSAGE_LENGTH
    assert "truncated" in result


def test_send_sync_calls_bot_api():
    bot = TelegramBot("fake-token", "123")
    with patch.object(bot._bot, "send_message", new_callable=AsyncMock) as mock_send:
        bot.send_sync("hello sync")
        mock_send.assert_called_once_with(chat_id="123", text="hello sync")


def test_send_sync_swallows_errors():
    bot = TelegramBot("fake-token", "123")
    with patch.object(bot._bot, "send_message", new_callable=AsyncMock, side_effect=Exception("net")):
        bot.send_sync("hello")  # must not raise
```

**Step 2: Run tests — expect FAIL** (module doesn't exist)

Run: `pytest tests/test_telegram.py -v`

**Step 3: Write implementation**

```python
# agentkit/telegram_bot.py
"""Telegram integration — send and receive messages."""

import asyncio
import logging

from telegram import Bot

log = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4096


class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._bot = Bot(token=token)

    async def send(self, text: str, chat_id: str | None = None) -> None:
        """Send a message to the specified or default chat."""
        chat_id = chat_id or self.chat_id
        text = self._truncate(text)
        try:
            await self._bot.send_message(chat_id=chat_id, text=text)
        except Exception as e:
            log.error("Failed to send Telegram message: %s", e)

    def send_sync(self, text: str, chat_id: str | None = None) -> None:
        """Synchronous wrapper for send."""
        asyncio.run(self.send(text, chat_id))

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_MESSAGE_LENGTH:
            return text
        return text[: MAX_MESSAGE_LENGTH - 20] + "\n\n... (truncated)"
```

**Step 4: Run tests — expect PASS**

Run: `pytest tests/test_telegram.py -v` → 6 pass

**Step 5:** Commit: `feat: telegram_bot.py with TelegramBot class`

---

## Task 3: Update agent.py — TELEGRAM: directive + process_next returns str|None (TDD)

**Files:**
- Modify: `agentkit/agent.py`
- Modify: `tests/test_agent.py`

Key changes:
1. Add `self.pending_messages: list[str] = []` to `__init__`
2. Add TELEGRAM: parsing to `_process_response()`
3. Add `_clean_response()` static method (strips MEMORY:/TELEGRAM: lines)
4. Change `process_next()` return: `str | None` (clean response or None)
5. Reset `pending_messages` at start of each `process_next()`

**Step 1: Replace test file**

```python
# tests/test_agent.py
"""Tests for core agent loop."""

from unittest.mock import patch

from agentkit.agent import Agent
from agentkit.claude import ToolMode, ClaudeError
from agentkit.config import Config


def _make_agent(tmp_path):
    (tmp_path / "profiles" / "test").mkdir(parents=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    config = Config(profile="test", project_root=tmp_path)
    return Agent(config)


@patch("agentkit.agent.invoke_claude")
def test_agent_process_task_returns_response(mock_claude, tmp_path):
    mock_claude.return_value = "task completed successfully"
    agent = _make_agent(tmp_path)
    agent.mailbox.enqueue("do something", source="test")
    result = agent.process_next()
    assert result == "task completed successfully"
    mock_claude.assert_called_once()


def test_agent_process_empty_queue_returns_none(tmp_path):
    agent = _make_agent(tmp_path)
    assert agent.process_next() is None


@patch("agentkit.agent.invoke_claude")
def test_agent_updates_memory_on_memory_prefix(mock_claude, tmp_path):
    mock_claude.return_value = "analysis done\nMEMORY: important finding"
    agent = _make_agent(tmp_path)
    agent.mailbox.enqueue("analyze data", source="test")
    agent.process_next()
    assert "important finding" in agent.memory.read_long_term()


@patch("agentkit.agent.invoke_claude")
def test_agent_process_with_tool_mode(mock_claude, tmp_path):
    mock_claude.return_value = "done"
    agent = _make_agent(tmp_path)
    agent.mailbox.enqueue("write code", source="test")
    agent.process_next(tool_mode=ToolMode.READWRITE)
    _, kwargs = mock_claude.call_args
    assert kwargs["tool_mode"] == ToolMode.READWRITE


@patch("agentkit.agent.invoke_claude")
def test_agent_collects_telegram_directives(mock_claude, tmp_path):
    mock_claude.return_value = "result\nTELEGRAM: hello from agent\nMEMORY: a fact"
    agent = _make_agent(tmp_path)
    agent.mailbox.enqueue("do work", source="test")
    agent.process_next()
    assert agent.pending_messages == ["hello from agent"]


@patch("agentkit.agent.invoke_claude")
def test_agent_strips_directives_from_response(mock_claude, tmp_path):
    mock_claude.return_value = "useful answer\nMEMORY: fact\nTELEGRAM: notify\nmore text"
    agent = _make_agent(tmp_path)
    agent.mailbox.enqueue("task", source="test")
    result = agent.process_next()
    assert "MEMORY:" not in result
    assert "TELEGRAM:" not in result
    assert "useful answer" in result
    assert "more text" in result


@patch("agentkit.agent.invoke_claude")
def test_agent_pending_messages_reset_each_process(mock_claude, tmp_path):
    mock_claude.return_value = "TELEGRAM: first"
    agent = _make_agent(tmp_path)
    agent.mailbox.enqueue("task1", source="test")
    agent.mailbox.enqueue("task2", source="test")
    agent.process_next()
    assert agent.pending_messages == ["first"]
    mock_claude.return_value = "no directives"
    agent.process_next()
    assert agent.pending_messages == []


@patch("agentkit.agent.invoke_claude")
def test_agent_run_all(mock_claude, tmp_path):
    mock_claude.return_value = "done"
    agent = _make_agent(tmp_path)
    agent.mailbox.enqueue("t1", source="test")
    agent.mailbox.enqueue("t2", source="test")
    assert agent.run_all() == 2


@patch("agentkit.agent.invoke_claude")
def test_agent_error_returns_none(mock_claude, tmp_path):
    mock_claude.side_effect = ClaudeError("boom")
    agent = _make_agent(tmp_path)
    agent.mailbox.enqueue("task", source="test")
    assert agent.process_next() is None
```

**Step 2: Run tests — expect FAIL** (pending_messages doesn't exist, return type wrong)

**Step 3: Replace agent.py**

```python
# agentkit/agent.py
"""Core agent loop — gather -> act -> verify -> iterate."""

import logging

from agentkit.claude import ClaudeError, ToolMode, invoke_claude
from agentkit.config import Config
from agentkit.context import ContextBuilder
from agentkit.mailbox import Mailbox
from agentkit.memory import Memory

log = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.memory = Memory(config.memory_dir)
        self.mailbox = Mailbox(config.db_path)
        self.context = ContextBuilder(config, self.memory)
        self.pending_messages: list[str] = []

    def process_next(self, *, tool_mode: ToolMode = ToolMode.READONLY) -> str | None:
        """Process the next task. Returns cleaned response or None."""
        task = self.mailbox.dequeue()
        if task is None:
            return None

        self.pending_messages = []
        log.info("Processing task %d: %s", task["id"], task["content"][:80])

        try:
            system_prompt = self.context.build_system_prompt()
            task_prompt = self.context.build_task_prompt(task["content"])
            response = invoke_claude(
                task_prompt, system_prompt=system_prompt, tool_mode=tool_mode
            )
            self._process_response(response)
            clean = self._clean_response(response)
            self.mailbox.complete(task["id"], result=response[:500])
            self.memory.append_today(f"Task: {task['content'][:100]}\nResult: {response[:200]}")
            log.info("Task %d completed", task["id"])
            return clean
        except ClaudeError as e:
            self.mailbox.fail(task["id"], error=str(e))
            log.error("Task %d failed: %s", task["id"], e)
            return None

    def _process_response(self, response: str) -> None:
        """Extract directives from Claude's response."""
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.startswith("MEMORY:"):
                self.memory.append_long_term(stripped[7:].strip())
            elif stripped.startswith("TELEGRAM:"):
                self.pending_messages.append(stripped[9:].strip())

    @staticmethod
    def _clean_response(response: str) -> str:
        """Strip directive lines from response."""
        lines = []
        for line in response.split("\n"):
            stripped = line.strip()
            if not stripped.startswith("MEMORY:") and not stripped.startswith("TELEGRAM:"):
                lines.append(line)
        return "\n".join(lines).strip()

    def run_all(self, *, tool_mode: ToolMode = ToolMode.READONLY) -> int:
        """Process all pending tasks. Returns count processed."""
        count = 0
        while self.process_next(tool_mode=tool_mode):
            count += 1
        return count
```

**Step 4: Run tests — expect PASS**

Run: `pytest tests/test_agent.py -v` → 9 pass

**Step 5:** Commit: `feat: TELEGRAM: directive + process_next returns str|None`

---

## Task 4: Update config.py — Config.from_env() (TDD)

**Files:**
- Modify: `agentkit/config.py`
- Modify: `tests/test_config.py`

**Step 1: Add tests**

Add to `tests/test_config.py`:

```python
def test_from_env_default(monkeypatch):
    monkeypatch.delenv("AGENT_PROFILE", raising=False)
    config = Config.from_env()
    assert config.profile == "playground"


def test_from_env_custom(monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE", "nanoclaw")
    config = Config.from_env()
    assert config.profile == "nanoclaw"
```

**Step 2: Add classmethod to config.py**

```python
@classmethod
def from_env(cls) -> "Config":
    """Create Config from environment variables."""
    profile = os.environ.get("AGENT_PROFILE", "playground")
    return cls(profile=profile)
```

**Step 3:** Run: `pytest tests/test_config.py -v` → 6 pass

**Step 4:** Commit: `feat: Config.from_env() with AGENT_PROFILE support`

---

## Task 5: Update cli.py — add run command + _send_pending (TDD)

**Files:**
- Modify: `agentkit/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Replace test file**

```python
# tests/test_cli.py
"""Tests for CLI."""

from unittest.mock import patch, MagicMock

from agentkit.cli import create_parser, _send_pending


def test_parser_task_command():
    parser = create_parser()
    args = parser.parse_args(["task", "do something"])
    assert args.command == "task"
    assert args.prompt == "do something"


def test_parser_task_write_flag():
    parser = create_parser()
    args = parser.parse_args(["task", "--write", "do something"])
    assert args.write is True


def test_parser_task_readonly_default():
    parser = create_parser()
    args = parser.parse_args(["task", "do something"])
    assert args.write is False


def test_parser_evaluate_command():
    parser = create_parser()
    args = parser.parse_args(["evaluate", "--profile", "trading"])
    assert args.command == "evaluate"
    assert args.profile == "trading"


def test_parser_default_profile():
    parser = create_parser()
    args = parser.parse_args(["task", "test"])
    assert args.profile == "playground"


def test_parser_run_command():
    parser = create_parser()
    args = parser.parse_args(["run", "--profile", "nanoclaw"])
    assert args.command == "run"
    assert args.profile == "nanoclaw"


def test_parser_run_default_profile():
    parser = create_parser()
    args = parser.parse_args(["run"])
    assert args.profile == "playground"


def test_send_pending_with_messages():
    config = MagicMock()
    config.telegram_bot_token = "tok"
    config.telegram_chat_id = "123"
    agent = MagicMock()
    agent.pending_messages = ["hello", "world"]
    with patch("agentkit.cli.TelegramBot") as MockBot:
        _send_pending(config, agent)
        assert MockBot.return_value.send_sync.call_count == 2


def test_send_pending_no_token():
    config = MagicMock()
    config.telegram_bot_token = ""
    agent = MagicMock()
    agent.pending_messages = ["hello"]
    with patch("agentkit.cli.TelegramBot") as MockBot:
        _send_pending(config, agent)
        MockBot.assert_not_called()
```

**Step 2: Replace cli.py**

```python
# agentkit/cli.py
"""CLI entry point — task, evaluate, run."""

import argparse
import logging
import os
from pathlib import Path

from agentkit.agent import Agent
from agentkit.claude import ToolMode
from agentkit.config import Config
from agentkit.telegram_bot import TelegramBot


def _send_pending(config: Config, agent: Agent) -> None:
    """Send pending TELEGRAM: messages if configured."""
    if agent.pending_messages and config.telegram_bot_token:
        bot = TelegramBot(config.telegram_bot_token, config.telegram_chat_id)
        for msg in agent.pending_messages:
            bot.send_sync(msg)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentkit", description="Autonomous agent framework")
    sub = parser.add_subparsers(dest="command")

    task_cmd = sub.add_parser("task", help="Process a single task")
    task_cmd.add_argument("prompt", help="Task to process")
    task_cmd.add_argument("--profile", default="playground")
    task_cmd.add_argument("--write", action="store_true", help="Enable READWRITE mode")

    eval_cmd = sub.add_parser("evaluate", help="Run evaluation cycle (always READONLY)")
    eval_cmd.add_argument("--profile", default="playground")

    run_cmd = sub.add_parser("run", help="Start daemon (Telegram polling)")
    run_cmd.add_argument(
        "--profile", default=os.environ.get("AGENT_PROFILE", "playground")
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "task":
        config = Config(profile=args.profile, project_root=Path.cwd())
        agent = Agent(config)
        tool_mode = ToolMode.READWRITE if args.write else ToolMode.READONLY
        agent.mailbox.enqueue(args.prompt, source="cli")
        agent.process_next(tool_mode=tool_mode)
        _send_pending(config, agent)

    elif args.command == "evaluate":
        config = Config(profile=args.profile, project_root=Path.cwd())
        eval_path = config.profile_dir / "evaluation.md"
        if not eval_path.exists():
            print(f"No evaluation.md found at {eval_path}")
            return
        agent = Agent(config)
        eval_template = eval_path.read_text()
        agent.mailbox.enqueue(eval_template, source="cron-evaluate")
        agent.process_next(tool_mode=ToolMode.READONLY)
        _send_pending(config, agent)

    elif args.command == "run":
        from agentkit.daemon import Daemon
        config = Config(profile=args.profile, project_root=Path.cwd())
        daemon = Daemon(config)
        daemon.run()

    else:
        parser.print_help()
```

**Step 3:** Run: `pytest tests/test_cli.py -v` → 9 pass

**Step 4:** Commit: `feat: run command + _send_pending in CLI`

---

## Task 6: Create daemon.py + tests (TDD)

**Files:**
- Create: `agentkit/daemon.py`
- Create: `tests/test_daemon.py`

**Step 1: Write tests**

```python
# tests/test_daemon.py
"""Tests for daemon mode."""

from unittest.mock import patch

import pytest

from agentkit.claude import ClaudeError
from agentkit.config import Config
from agentkit.daemon import Daemon


def _make_daemon(tmp_path):
    (tmp_path / "profiles" / "test").mkdir(parents=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    config = Config(profile="test", project_root=tmp_path)
    return Daemon(config)


def test_daemon_init(tmp_path):
    daemon = _make_daemon(tmp_path)
    assert daemon.agent is not None
    assert daemon.config.profile == "test"


@patch("agentkit.agent.invoke_claude")
def test_handle_message_enqueues_and_processes(mock_claude, tmp_path):
    mock_claude.return_value = "I helped you\nMEMORY: user asked for help"
    daemon = _make_daemon(tmp_path)
    result = daemon.handle_message("help me")
    assert "I helped you" in result
    assert "MEMORY:" not in result


@patch("agentkit.agent.invoke_claude")
def test_handle_message_returns_none_on_error(mock_claude, tmp_path):
    mock_claude.side_effect = ClaudeError("fail")
    daemon = _make_daemon(tmp_path)
    assert daemon.handle_message("hello") is None


@patch("agentkit.agent.invoke_claude")
def test_handle_message_collects_pending(mock_claude, tmp_path):
    mock_claude.return_value = "ok\nTELEGRAM: notification"
    daemon = _make_daemon(tmp_path)
    daemon.handle_message("do thing")
    assert daemon.agent.pending_messages == ["notification"]


def test_daemon_validate_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    daemon = _make_daemon(tmp_path)
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        daemon.validate()


def test_daemon_validate_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    daemon = _make_daemon(tmp_path)
    daemon.validate()  # should not raise
```

**Step 2: Write implementation**

```python
# agentkit/daemon.py
"""Daemon mode — long-running Telegram-connected agent."""

import asyncio
import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters

from agentkit.agent import Agent
from agentkit.config import Config
from agentkit.telegram_bot import TelegramBot

log = logging.getLogger(__name__)


class Daemon:
    def __init__(self, config: Config):
        self.config = config
        self.agent = Agent(config)

    def validate(self) -> None:
        """Validate required config for daemon mode."""
        if not self.config.telegram_bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required for daemon mode")

    def handle_message(self, text: str, source: str = "telegram") -> str | None:
        """Enqueue message, process, return clean response."""
        self.agent.mailbox.enqueue(text, source=source)
        return self.agent.process_next()

    def run(self) -> None:
        """Start Telegram polling loop. Blocks forever."""
        self.validate()
        log.info("Starting daemon with profile=%s", self.config.profile)

        app = ApplicationBuilder().token(self.config.telegram_bot_token).build()

        async def on_message(update: Update, context) -> None:
            if not update.message or not update.message.text:
                return

            user_text = update.message.text
            chat_id = str(update.message.chat_id)
            log.info("Received from chat %s: %s", chat_id, user_text[:80])

            # Run blocking Claude CLI call in thread pool
            response = await asyncio.to_thread(self.handle_message, user_text)

            # Send main response back to user's chat
            if response:
                bot = TelegramBot(self.config.telegram_bot_token, chat_id)
                await bot.send(response)

            # Send TELEGRAM: directive messages to notification channel
            if self.agent.pending_messages and self.config.telegram_chat_id:
                notify_bot = TelegramBot(
                    self.config.telegram_bot_token, self.config.telegram_chat_id
                )
                for msg in self.agent.pending_messages:
                    await notify_bot.send(msg)

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
        log.info("Telegram polling started — endless loop")
        app.run_polling()
```

**Step 3:** Run: `pytest tests/test_daemon.py -v` → 6 pass

**Step 4:** Commit: `feat: daemon.py with Telegram polling`

---

## Task 7: Create playground profile

**Files:**
- Create: `profiles/playground/identity.md`
- Create: `profiles/playground/tools.md`
- Create: `profiles/playground/evaluation.md`

```markdown
# profiles/playground/identity.md
# Playground Agent

You are a helpful assistant for testing and development of the agentkit framework.

- Answer questions clearly and concisely
- Use the MEMORY: prefix for observations worth remembering
- Use the TELEGRAM: prefix for messages that should be sent as notifications
```

```markdown
# profiles/playground/tools.md
# Available Tools

- **Read/Glob/Grep**: Search and read files
- **WebSearch/WebFetch**: Search the web

Write tools (Edit, Write, Bash) are only available in READWRITE mode (--write flag).
```

```markdown
# profiles/playground/evaluation.md
# Evaluation Cycle

Review current state and provide assessment:

1. **Health Check**: Any obvious issues?
2. **Memory Review**: Any patterns in recent daily memory?
3. **Improvement**: Suggest one concrete improvement.

Format:
- Status: [healthy/degraded/broken]
- Summary: [1-2 sentences]
- MEMORY: [observations worth remembering]
```

**Commit:** `feat: playground profile`

---

## Task 8: Create Dockerfile + entrypoint

**Files:**
- Create: `docker/entrypoint.sh`
- Create: `Dockerfile`

```bash
#!/usr/bin/env bash
# docker/entrypoint.sh
set -euo pipefail

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is required"
    exit 1
fi

if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
    echo "ERROR: TELEGRAM_BOT_TOKEN is required"
    exit 1
fi

echo "=== agentkit daemon ==="
echo "Profile: ${AGENT_PROFILE:-playground}"
echo "Starting..."

exec python3 -m agentkit "$@"
```

```dockerfile
# Dockerfile
FROM node:20-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app

COPY pyproject.toml .
COPY agentkit/ agentkit/
RUN python3 -m pip install --break-system-packages -e .

RUN mkdir -p /app/profiles /app/memory /app/data

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["run"]
```

**Verify:** `docker build -t agentkit .`

**Commit:** `feat: Dockerfile + entrypoint`

---

## Task 9: Create docker-compose.yml + update .env.example

**Files:**
- Create: `docker-compose.yml`
- Modify: `.env.example`

```yaml
# docker-compose.yml
version: "3.8"

services:
  agent:
    build: .
    env_file: .env
    environment:
      - AGENT_PROFILE=${AGENT_PROFILE:-playground}
    volumes:
      - ./profiles:/app/profiles:ro
      - ./memory:/app/memory
      - ./data:/app/data
    restart: unless-stopped
```

```
# .env.example — updated
ANTHROPIC_API_KEY=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
AGENT_PROFILE=playground
```

**Commit:** `feat: docker-compose.yml for single-command deploy`

---

## Task 10: Update CLAUDE.md + .gitignore

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.gitignore`

Add to CLAUDE.md architecture diagram:
```
CLI / Cron / Telegram → Mailbox (SQLite) → Agent Loop → Context Builder →
Claude CLI → Result Parser → Memory Update + Telegram notifications
```

Add daemon + Docker sections. Update directives (TELEGRAM: is now core).

Add to .gitignore: `docker-compose.override.yml`

**Commit:** `docs: update CLAUDE.md for Docker + Telegram`

---

## Task 11: Full test suite + lint + push

**Steps:**
1. `pytest -v` → ~57 tests pass (42 original - 4 old agent + 9 new agent + 6 telegram + 6 daemon + 2 config = ~61)
2. `ruff check .` → clean
3. `docker build -t agentkit .` → builds
4. `docker-compose config` → validates
5. `git push`

---

## Verification

```bash
# Unit tests
cd /Users/petr.kubelka/git_projects/agentkit
source .venv/bin/activate && pytest -v && ruff check .

# Docker build
docker build -t agentkit .

# Docker compose validation
docker-compose config

# Live test (requires real keys)
cp .env.example .env  # fill in keys
docker-compose up
# → send message to Telegram bot → get response
```

## Critical Files

| File | Action | LOC delta |
|------|--------|-----------|
| `agentkit/agent.py` | Modify | +20 |
| `agentkit/telegram_bot.py` | Create | +35 |
| `agentkit/daemon.py` | Create | +55 |
| `agentkit/cli.py` | Modify | +20 |
| `agentkit/config.py` | Modify | +6 |
| `Dockerfile` | Create | +18 |
| `docker-compose.yml` | Create | +14 |
| `docker/entrypoint.sh` | Create | +15 |
| Core total | | ~600 LOC |
