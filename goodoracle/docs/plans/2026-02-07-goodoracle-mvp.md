# GoodOracle MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `agentkit` — a pip-installable autonomous agent framework (~650 LOC) wrapping Claude Code CLI. Then build `goodoracle` as the first consumer (trading agent). The framework is reusable — a game-monitoring agent will be the second consumer.

**Architecture:** Two repos. `agentkit` provides the agent lifecycle (mailbox → context → Claude CLI → verify → act → memory). `goodoracle` adds trading-specific profiles, tools, and config. Consumer agents import `agentkit` and bring their own domain knowledge.

**Tech Stack:** Python 3.12+, Claude Code CLI (Opus 4.6 enforced), SQLite, python-telegram-bot, pytest, ruff

**Inspirations:**
- **nanobot** (HKUDS): message bus, dual-layer memory, skill system, heartbeat/cron, ~4K LOC
- **AutoAgent** (HKUDS): self-generating tools, zero-code agent creation
- **Claude Agent SDK** (Anthropic): gather→act→verify→iterate loop, structured handoffs, subagents
- **Key differentiator:** We use local Claude Code CLI (full tool ecosystem) instead of API calls

**High-Agency Principles** (from Anthropic research + nanobot + AutoAgent):
1. **Verify step:** Agent checks its own output against success criteria before declaring done
2. **Orientation ritual:** Every session starts by reading progress, memory, git log, health checks
3. **Tool self-generation:** Agent can write + register new tools at runtime (not just improve existing code)
4. **Permissions:** READ-ONLY by default, READWRITE requires explicit opt-in
5. **Self-scheduling:** Agent manages its own crontab entries, not a static plist
6. **Self-evolution:** Agent improves its own code on a branch, with test gates and human merge

---

## Plan Tiers

### CORE (build first — the system must work end-to-end)

| Phase | What | Tasks |
|-------|------|-------|
| 1. agentkit Foundation | scaffold, CLAUDE.md, config, claude.py + ToolMode | 1-4 |
| 2. Memory & Context | memory system, context builder + orientation | 5-6 |
| 3. Agent Core | mailbox, tools, agent loop (verify step) | 7-9 |
| 4. Playground & Smoke Tests | real Claude calls, prove it works | 10-11 |
| 5. Communication | Telegram integration, CLI tests | 12-13 |
| 6. Scheduling & Self-Evolution | crontab scheduler, evolve.py | 14-15 |
| 7. goodoracle Consumer | scaffold, trading profile, grafana tool | 16-18 |
| 8. Skills & Cleanup | goodoracle-dev skill, archive old skills, trading CLAUDE.md + doc cleanup | 19-21 |
| 9. Final Verification | full E2E, clean-clone test | 22 |

### LAYER 2 (build after core works — multiplies agent value)

| Phase | What | Tasks |
|-------|------|-------|
| 10. Tool Sharing via PRs | agents propose generic tools to agentkit via real GitHub PRs, human reviews | 23-24 |
| 11. Trading Doc Cleanup | remove redundant docs from trading/, single source of truth | 25 |

### LAYER 3 (nice-to-have — builds on everything above)

| Phase | What | Tasks |
|-------|------|-------|
| 12. Web Dashboard | create/manage agents via web UI, chat interface, tool usage viewer | 26-28 |
| 13. Game Monitor Agent | second agentkit consumer, proves framework is reusable | 29 |
| 14. Advanced Autonomy | heartbeat, subagent spawning, context window management | 30-32 |

---

## Trading Repo Coordination: Single Source of Truth

**Problem:** You run Claude Code directly in `/Users/petr.kubelka/git_projects/trading/` for coding. GoodOracle also operates on the trading repo for evaluation. If they disagree, they fight each other.

**Solution: Split by concern, link by reference.**

| Concern | Source of Truth | Who Updates | Where |
|---------|----------------|-------------|-------|
| **Strategy decisions** (what to trade, hypothesis states, priorities) | GoodOracle | goodoracle agent (via evaluation cycle) | `trading/docs/HYPOTHESES.md`, `trading/docs/TRADING-JOURNAL.md` |
| **Implementation** (code, tests, config) | Human + Claude in trading repo | you + Claude Code | `trading/src/`, `trading/config/` |
| **System health** (is it running, errors, metrics) | GoodOracle | goodoracle agent (reads Grafana) | `goodoracle/memory/` |
| **What to work on next** | GoodOracle recommends, human decides | goodoracle → `goodoracle-dev` skill → you | goodoracle evaluation → trading session |

**How it works in practice:**

1. GoodOracle runs every 6h, evaluates health + strategy, updates `HYPOTHESES.md` and `TRADING-JOURNAL.md` in the trading repo
2. You start a Claude Code session in trading/
3. Claude reads `trading/CLAUDE.md` which says: "use `goodoracle-dev` skill first"
4. `goodoracle-dev` reads oracle's latest memory + evaluation
5. You pick what to work on → Claude implements in trading repo
6. Implementation session updates oracle memory so next evaluation has fresh context
7. No conflicts because strategy flows one way (oracle → dev) and implementation flows the other (dev → code)

**Documents to clean up in trading/ (Task 25):**

| File | Action | Why |
|------|--------|-----|
| Old skill-generated strategy docs | Remove | Now in goodoracle memory |
| Redundant STATUS.md copies | Consolidate to one | Avoid stale duplicates |
| Any "what to do next" docs | Remove | Oracle decides this now |
| `trading/CLAUDE.md` | Update | Must reference goodoracle as strategy brain |

**Key rule for `trading/CLAUDE.md`:**
```
Strategy and priorities: defer to GoodOracle (read oracle memory first via goodoracle-dev skill).
Implementation and code: you are the authority (write code, run tests, deploy).
Never contradict oracle's hypothesis states or priority recommendations without discussing with the human first.
```

---

## Two-Repo Architecture

### agentkit (framework — pip-installable)

```
agentkit/
├── CLAUDE.md
├── Makefile
├── pyproject.toml
├── .env.example
├── .gitignore
├── agentkit/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py              # Base CLI: task, evaluate, run, schedule, self-improve, --write flag
│   ├── config.py            # Base configuration
│   ├── claude.py            # Claude Code CLI wrapper + ToolMode enum (READONLY default)
│   ├── memory.py            # Dual-layer memory (daily + long-term)
│   ├── context.py           # Context builder + orientation ritual
│   ├── mailbox.py           # SQLite task queue
│   ├── agent.py             # Core agent loop (gather→act→verify→iterate) + SCHEDULE: directive
│   ├── telegram_bot.py      # Telegram integration
│   ├── evolve.py            # Self-improvement cycle + tool generation
│   ├── scheduler.py         # Crontab self-scheduling
│   └── tools/
│       ├── __init__.py      # Tool registry
│       ├── base.py          # Tool interface
│       └── shell.py         # Shell command execution
├── profiles/
│   └── playground/          # Framework test profile (ships with agentkit)
│       ├── identity.md
│       ├── tools.md
│       └── evaluation.md
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_claude.py       # + ToolMode tests
│   ├── test_memory.py
│   ├── test_context.py
│   ├── test_mailbox.py
│   ├── test_agent.py        # + verify step, SCHEDULE: directive tests
│   ├── test_tools.py
│   ├── test_telegram.py
│   ├── test_cli.py          # + --write, schedule, self-improve tests
│   ├── test_scheduler.py    # NEW
│   └── test_evolve.py       # NEW
├── scripts/
│   └── smoke-test.sh        # E2E test with REAL Claude CLI
└── docs/
    └── plans/
```

### goodoracle (trading agent — consumer of agentkit)

```
goodoracle/
├── CLAUDE.md
├── Makefile
├── pyproject.toml            # depends on agentkit
├── .env.example
├── .gitignore
├── goodoracle/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                # Extends agentkit CLI with trading defaults
│   └── tools/
│       ├── __init__.py
│       └── grafana.py        # Trading-specific Grafana tool
├── profiles/
│   └── trading/
│       ├── identity.md       # References files in trading repo (no duplication)
│       ├── tools.md          # Describes tools, Claude figures out the HOW
│       └── evaluation.md     # Describes WHAT to check, not HOW
├── memory/
│   ├── MEMORY.md             # Agent-level knowledge (cross-domain)
│   └── daily/                # Agent observations
├── data/
│   ├── goodoracle.db         # Task queue (SQLite)
│   ├── schedule.json         # Job definitions
│   └── evolution-log.json    # Self-improvement history
├── tests/
│   ├── test_cli.py
│   └── test_grafana.py
├── scripts/
│   ├── smoke-test.sh
│   └── install-schedule.sh   # Creates initial schedule.json + syncs crontab
└── docs/
    ├── archived-skills/
    └── plans/
        └── 2026-02-07-goodoracle-mvp.md  # This file
```

### File Ownership: goodoracle vs trading repo

| What | Where | Why |
|------|-------|-----|
| `HYPOTHESES.md` | `trading/docs/` | Domain data, trading-specific |
| `TRADING-JOURNAL.md` | `trading/docs/` | Domain data, trading-specific |
| `config/default.toml` | `trading/config/` | Bot runtime config |
| `STATUS.md` | `trading/docs/plans/` | Implementation tracking |
| Long-term memory | `goodoracle/memory/MEMORY.md` | Agent knowledge (cross-domain) |
| Daily observations | `goodoracle/memory/daily/` | Agent observations |
| Task history | `goodoracle/data/goodoracle.db` | Agent task queue |
| Evolution log | `goodoracle/data/evolution-log.json` | Agent self-improvement history |
| Schedule | `goodoracle/data/schedule.json` | Agent scheduling |

**Rule:** Trading-specific data stays in the trading repo. GoodOracle stores agent-level knowledge only. The trading profile's `identity.md` REFERENCES files in the trading repo (absolute paths) — it does NOT duplicate them.

**LOC target:** ~650 LOC agentkit core, ~150 LOC goodoracle consumer. If you can't understand the entire codebase in 10 minutes, it's too complex.

---

## Phase 1: agentkit Foundation

### Task 1: Create agentkit Project Scaffold

**Repo:** `agentkit`

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `agentkit/__init__.py`
- Create: `tests/conftest.py`
- Create: `Makefile`

**Step 1: Create repo and directory structure**

```bash
cd /Users/petr.kubelka/git_projects
mkdir -p agentkit
cd agentkit
git init
mkdir -p agentkit/tools profiles/playground memory/daily tests scripts docs/plans data logs
```

**Step 2: Write pyproject.toml**

```toml
[project]
name = "agentkit"
version = "0.1.0"
description = "Lightweight autonomous agent framework wrapping Claude Code CLI"
requires-python = ">=3.12"
dependencies = [
    "python-telegram-bot>=21.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]

[project.scripts]
agentkit = "agentkit.cli:main"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 3: Write .gitignore**

```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
.venv/
data/*.db
data/*.json
logs/
```

**Step 4: Write .env.example**

```bash
# Telegram
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

**Step 5: Write agentkit/__init__.py**

```python
"""agentkit — lightweight autonomous agent framework wrapping Claude Code CLI."""

__version__ = "0.1.0"
```

**Step 6: Write tests/conftest.py**

```python
"""Shared test fixtures."""

from pathlib import Path
import pytest


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project structure for testing."""
    (tmp_path / "profiles" / "test").mkdir(parents=True)
    (tmp_path / "profiles" / "test" / "identity.md").write_text("You are a test agent.")
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "daily").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path
```

**Step 7: Write Makefile**

```makefile
.PHONY: install test lint playground smoke clean

install:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest -v

lint:
	.venv/bin/ruff check .

# Run a single task with the playground profile — REAL Claude, no mocks
playground:
	.venv/bin/agentkit task --profile playground "Read the file CLAUDE.md in this project directory. Count the number of second-level headings (##). Respond with: TELEGRAM: Found N headings in CLAUDE.md. MEMORY: CLAUDE.md has N second-level headings."

# Full end-to-end smoke test — verifies entire pipeline with real Claude CLI
smoke:
	./scripts/smoke-test.sh

# Show task history from the mailbox
history:
	sqlite3 data/agentkit.db "SELECT id, status, source, substr(content, 1, 60) as task, substr(result, 1, 60) as result, created_at FROM tasks ORDER BY id DESC LIMIT 10;"

# Show current memory state
memory:
	@echo "=== Long-Term Memory ==="
	@cat memory/MEMORY.md 2>/dev/null || echo "(empty)"
	@echo ""
	@echo "=== Today ==="
	@cat memory/daily/$$(date +%Y-%m-%d).md 2>/dev/null || echo "(no entries today)"

# Reset all state (memory + db) for fresh testing
clean:
	rm -f data/agentkit.db
	rm -f memory/MEMORY.md
	rm -f memory/daily/*.md
	@echo "State cleaned. Ready for fresh run."
```

**Step 8: Create placeholders**

```bash
touch memory/daily/.gitkeep
touch memory/MEMORY.md
```

**Step 9: Install and verify**

Run: `cd /Users/petr.kubelka/git_projects/agentkit && make install`
Expected: Successfully installed agentkit-0.1.0

**Step 10: Commit**

```bash
git add -A
git commit -m "feat: agentkit project scaffold with pyproject.toml and directory structure"
```

---

### Task 2: Write agentkit CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

**Step 1: Write CLAUDE.md**

```markdown
# agentkit

## What Is This?

Lightweight autonomous agent framework (~650 LOC) wrapping Claude Code CLI.
Pip-installable. Consumer agents import agentkit and bring their own profiles,
tools, and domain knowledge. GoodOracle (trading) is the first consumer.

## Non-Negotiable Rules

1. **Always Opus 4.6** — every Claude CLI call MUST use `--model claude-opus-4-6`. No exceptions.
2. **Small codebase** — target ~650 LOC. If you can't understand it in 10 minutes, it's too complex.
3. **Domain agnostic** — no trading/game logic in core. Profiles add domain behavior.
4. **File-based memory** — human-readable Markdown, git-trackable.
5. **Single process** — no microservices, no message queues, no abstraction layers.
6. **READ-ONLY by default** — Claude CLI calls use ToolMode.READONLY unless explicitly opted in.

## Architecture

```
Telegram / Cron / CLI
        |
    Mailbox (SQLite queue)
        |
    Agent Loop (gather → act → verify → iterate)
        |
    Context Builder (orientation + identity + memory + task)
        |
    Claude Code CLI (--print --model claude-opus-4-6 --allowedTools ...)
        |
    Result Parser (MEMORY:, TELEGRAM:, SCHEDULE: directives)
        |
    Verify Step (check output against success criteria)
        |
    Memory Update + Telegram Response + Schedule Update
```

## Agent Loop: High-Agency Design

The agent follows Anthropic's gather→act→verify→iterate pattern:

1. **Orientation** — every invocation starts by reading: progress file, recent memory,
   git log (if in a repo), and running health checks. This prevents the "cold start" problem.
2. **Gather context** — assemble identity + tools + memory + task into a prompt.
3. **Act** — invoke Claude CLI with assembled context.
4. **Verify** — agent checks its own output against success criteria before declaring done.
   If verification fails, it iterates (up to a configurable retry limit).
5. **Commit results** — update memory, send Telegram messages, update schedule.

## Permissions: ToolMode

Claude CLI calls default to READ-ONLY (safe observation). Write access requires explicit opt-in:

- `ToolMode.READONLY` (default): `Read,Glob,Grep,WebSearch,WebFetch` — agent observes
- `ToolMode.READWRITE`: all tools — needed for self-improvement, explicit `--write` flag

## Self-Scheduling

The agent can propose new scheduled jobs via `SCHEDULE:` response directive.
Jobs are stored in `data/schedule.json` but do NOT auto-sync to crontab (safety).
Syncing is explicit: `agentkit schedule sync`.

Self-improvement runs can add research/analysis cycles. Example: after validating a
hypothesis, the agent might schedule more frequent evaluation during volatile periods.

## Self-Evolution

The agent can improve its own code via `agentkit self-improve`:
1. Creates git branch: `evolve/YYYY-MM-DD-HHMMSS`
2. Invokes Claude with ToolMode.READWRITE
3. Runs `make test` — if tests pass, commits to branch
4. If tests fail, reverts everything
5. Human merges manually (no auto-merge)
6. Can also GENERATE new tools at runtime

## Running

```bash
# Single task (READONLY default)
agentkit task --profile playground "Analyze the system"

# Single task with write access
agentkit task --profile playground --write "Fix the config file"

# Evaluation cycle (always READONLY)
agentkit evaluate --profile playground

# Self-improvement (always READWRITE, on a branch)
agentkit self-improve

# Schedule management
agentkit schedule list
agentkit schedule add --name "eval-trading" --command "evaluate" --profile trading --cron "0 */6 * * *"
agentkit schedule sync
```

## Development

```bash
pip install -e ".[dev]"
pytest -v
ruff check .
```

## Response Directives

Claude's response is parsed for directives:
- Lines starting with `MEMORY:` → appended to long-term memory
- Lines starting with `TELEGRAM:` → sent to Telegram
- Lines starting with `SCHEDULE:` → proposed new scheduled job (stored, not auto-synced)
- Everything else → stored as task result

## Profile Structure

```
profiles/<name>/
├── identity.md      # Who the agent is
├── tools.md         # What tools are available (describe WHAT, not HOW)
└── evaluation.md    # Template for scheduled evaluation cycles
```

## Consumer Pattern

Consumer agents (goodoracle, game-monitor, etc.) depend on agentkit:
```toml
dependencies = ["agentkit>=0.1.0"]
```
They provide: profiles, domain-specific tools, CLI entry point, config overrides.
agentkit provides: agent loop, memory, mailbox, scheduling, evolution, Telegram.

## Key Design Decisions

### Why Claude Code CLI, not API?
Claude Code CLI gives access to the full tool ecosystem (Bash, Read, Write, Edit,
Grep, Glob). The agent can read files, run commands, edit code — all through a
single CLI call. No need to reimplement tools.

### Why mailbox pattern?
Everything is a task: Telegram messages, cron triggers, manual CLI invocations.
One queue, one processing loop. Simple to debug, simple to extend.

### Why profiles?
The same agent framework can be a trading analyst, a game monitor, a code reviewer.
Profile = identity + tools + memory context. Swap profiles, swap behavior.

### Why file-based memory?
- Human-readable (you can open MEMORY.md and read it)
- Git-trackable (see what the agent learned over time)
- Editable (fix mistakes by editing Markdown)
- Simple (no vector DB, no embeddings, no complexity)

### Why verify step?
From Anthropic's research: agents that can check and improve their own output are
fundamentally more reliable. They catch mistakes before they compound and self-correct
when they drift. Without verification, agents "declare victory prematurely."

## Superpowers Skills Workflow

For development on this project:
- Before coding: `superpowers:writing-plans` + `superpowers:executing-plans`
- For features: `superpowers:test-driven-development` (TDD always)
- For debugging: `superpowers:systematic-debugging`
- After implementation: `superpowers:verification-before-completion`
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md project manifest for agentkit"
```

---

### Task 3: Configuration System

**Files:**
- Create: `agentkit/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import os
from pathlib import Path

from agentkit.config import Config


def test_load_config_defaults():
    config = Config(profile="trading", project_root=Path("/tmp/test-agentkit"))
    assert config.profile == "trading"
    assert config.model == "claude-opus-4-6"


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    config = Config(profile="trading", project_root=Path("/tmp/test-agentkit"))
    assert config.telegram_bot_token == "test-token"
    assert config.telegram_chat_id == "12345"


def test_profile_dir(tmp_path):
    profiles_dir = tmp_path / "profiles" / "trading"
    profiles_dir.mkdir(parents=True)
    config = Config(profile="trading", project_root=tmp_path)
    assert config.profile_dir == profiles_dir


def test_model_always_opus():
    """Model must ALWAYS be claude-opus-4-6, no override possible."""
    config = Config(profile="trading", project_root=Path("/tmp/test"))
    assert config.model == "claude-opus-4-6"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# agentkit/config.py
"""Configuration — loads profile and environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    profile: str
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    model: str = field(default="claude-opus-4-6", init=False)

    @property
    def profile_dir(self) -> Path:
        return self.project_root / "profiles" / self.profile

    @property
    def memory_dir(self) -> Path:
        return self.project_root / "memory"

    @property
    def db_path(self) -> Path:
        return self.project_root / "data" / "agentkit.db"

    @property
    def schedule_path(self) -> Path:
        return self.project_root / "data" / "schedule.json"

    @property
    def evolution_log_path(self) -> Path:
        return self.project_root / "data" / "evolution-log.json"

    @property
    def telegram_bot_token(self) -> str:
        return os.environ.get("TELEGRAM_BOT_TOKEN", "")

    @property
    def telegram_chat_id(self) -> str:
        return os.environ.get("TELEGRAM_CHAT_ID", "")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (4 passed)

**Step 5: Commit**

```bash
git add agentkit/config.py tests/test_config.py
git commit -m "feat: configuration system with profile support"
```

---

### Task 4: Claude Code CLI Wrapper + ToolMode

**Files:**
- Create: `agentkit/claude.py`
- Create: `tests/test_claude.py`

**Step 1: Write the failing test**

```python
# tests/test_claude.py
import subprocess
from unittest.mock import patch, MagicMock

from agentkit.claude import invoke_claude, ClaudeError, ToolMode, READONLY_TOOLS


def test_invoke_claude_basic():
    """Test basic Claude CLI invocation with mocked subprocess."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Hello, world!"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = invoke_claude("Say hello")
        assert result == "Hello, world!"

        # Verify opus-4-6 is ALWAYS enforced
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        assert "claude-opus-4-6" in cmd


def test_invoke_claude_readonly_default():
    """READONLY mode should set --allowedTools flag."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "ok"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        invoke_claude("Read something", tool_mode=ToolMode.READONLY)
        cmd = mock_run.call_args[0][0]
        assert "--allowedTools" in cmd
        # Verify only safe tools are allowed
        idx = cmd.index("--allowedTools")
        allowed = cmd[idx + 1]
        assert "Read" in allowed
        assert "Glob" in allowed
        assert "Grep" in allowed
        assert "WebSearch" in allowed
        assert "WebFetch" in allowed


def test_invoke_claude_readwrite_no_restriction():
    """READWRITE mode should NOT set --allowedTools (all tools available)."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "ok"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        invoke_claude("Write something", tool_mode=ToolMode.READWRITE)
        cmd = mock_run.call_args[0][0]
        assert "--allowedTools" not in cmd


def test_invoke_claude_with_context():
    """Test that context is piped as stdin."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Analysis complete"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        invoke_claude("Analyze this", context="some data")
        assert mock_run.call_args[1]["input"] == "some data"


def test_invoke_claude_with_system_prompt():
    """Test system prompt is passed correctly."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Done"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        invoke_claude("Do something", system_prompt="You are a trader")
        cmd = mock_run.call_args[0][0]
        assert "--system-prompt" in cmd


def test_invoke_claude_enforces_opus():
    """Model flag must ALWAYS be claude-opus-4-6. This is non-negotiable."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "ok"
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        invoke_claude("test")
        cmd = mock_run.call_args[0][0]
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "claude-opus-4-6"


def test_invoke_claude_timeout():
    """Test timeout handling."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 300)):
        try:
            invoke_claude("slow task", timeout=300)
            assert False, "Should have raised"
        except ClaudeError as e:
            assert "timeout" in str(e).lower()


def test_invoke_claude_nonzero_exit():
    """Test non-zero exit code handling."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error: something went wrong"

    with patch("subprocess.run", return_value=mock_result):
        try:
            invoke_claude("bad prompt")
            assert False, "Should have raised"
        except ClaudeError as e:
            assert "something went wrong" in str(e)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_claude.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# agentkit/claude.py
"""Claude Code CLI wrapper — ALWAYS enforces Opus 4.6. READ-ONLY by default."""

import subprocess
from enum import Enum


class ClaudeError(Exception):
    """Raised when Claude CLI invocation fails."""


class ToolMode(Enum):
    READONLY = "readonly"
    READWRITE = "readwrite"


MODEL = "claude-opus-4-6"
DEFAULT_TIMEOUT = 600  # 10 minutes
READONLY_TOOLS = "Read,Glob,Grep,WebSearch,WebFetch"


def invoke_claude(
    prompt: str,
    *,
    context: str | None = None,
    system_prompt: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    output_format: str | None = None,
    tool_mode: ToolMode = ToolMode.READONLY,
) -> str:
    """Invoke Claude Code CLI and return the output.

    IMPORTANT: Model is ALWAYS claude-opus-4-6. No override possible.
    IMPORTANT: Default tool_mode is READONLY — write access requires explicit opt-in.

    Args:
        prompt: The task/question for Claude.
        context: Optional text piped as stdin.
        system_prompt: Optional system prompt.
        timeout: Timeout in seconds (default 600).
        output_format: Optional output format ("json").
        tool_mode: Permission level (READONLY default, READWRITE for self-improvement).

    Returns:
        Claude's text response.

    Raises:
        ClaudeError: On timeout, non-zero exit, or other failures.
    """
    cmd = ["claude", "--print", "--model", MODEL]

    if tool_mode == ToolMode.READONLY:
        cmd.extend(["--allowedTools", READONLY_TOOLS])

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    if output_format:
        cmd.extend(["--output-format", output_format])

    cmd.append(prompt)

    try:
        result = subprocess.run(
            cmd,
            input=context,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ClaudeError(f"Timeout after {timeout}s")

    if result.returncode != 0:
        raise ClaudeError(result.stderr.strip() or f"Exit code {result.returncode}")

    return result.stdout.strip()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_claude.py -v`
Expected: PASS (8 passed)

**Step 5: Commit**

```bash
git add agentkit/claude.py tests/test_claude.py
git commit -m "feat: Claude Code CLI wrapper with ToolMode (READONLY default)"
```

---

## Phase 2: Memory & Context

### Task 5: Memory System

**Files:**
- Create: `agentkit/memory.py`
- Create: `tests/test_memory.py`

**Step 1: Write the failing test**

```python
# tests/test_memory.py
from datetime import date
from pathlib import Path

from agentkit.memory import Memory


def test_read_long_term_empty(tmp_path):
    mem = Memory(memory_dir=tmp_path)
    assert mem.read_long_term() == ""


def test_write_and_read_long_term(tmp_path):
    mem = Memory(memory_dir=tmp_path)
    mem.write_long_term("I learned that BTC markets are efficient.")
    assert "BTC markets are efficient" in mem.read_long_term()


def test_append_long_term(tmp_path):
    mem = Memory(memory_dir=tmp_path)
    mem.write_long_term("Fact 1.")
    mem.append_long_term("Fact 2.")
    content = mem.read_long_term()
    assert "Fact 1." in content
    assert "Fact 2." in content


def test_read_today_empty(tmp_path):
    mem = Memory(memory_dir=tmp_path)
    assert mem.read_today() == ""


def test_append_today(tmp_path):
    mem = Memory(memory_dir=tmp_path)
    mem.append_today("Observation: spread is $1.01")
    content = mem.read_today()
    assert "spread is $1.01" in content


def test_append_today_creates_file_with_header(tmp_path):
    mem = Memory(memory_dir=tmp_path)
    mem.append_today("First entry")
    content = mem.read_today()
    today = date.today().isoformat()
    assert today in content


def test_read_recent_days(tmp_path):
    mem = Memory(memory_dir=tmp_path)
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir(exist_ok=True)
    (daily_dir / "2026-02-06.md").write_text("# 2026-02-06\n\nYesterday note")
    mem.append_today("Today note")
    recent = mem.read_recent(days=7)
    assert "Today note" in recent
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# agentkit/memory.py
"""Memory management — daily observations + long-term knowledge."""

from datetime import date, timedelta
from pathlib import Path


class Memory:
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.daily_dir = memory_dir / "daily"
        self.daily_dir.mkdir(exist_ok=True)

    @property
    def _long_term_path(self) -> Path:
        return self.memory_dir / "MEMORY.md"

    def _daily_path(self, d: date | None = None) -> Path:
        d = d or date.today()
        return self.daily_dir / f"{d.isoformat()}.md"

    def read_long_term(self) -> str:
        path = self._long_term_path
        return path.read_text() if path.exists() else ""

    def write_long_term(self, content: str) -> None:
        self._long_term_path.write_text(content)

    def append_long_term(self, content: str) -> None:
        existing = self.read_long_term()
        separator = "\n\n" if existing else ""
        self._long_term_path.write_text(existing + separator + content)

    def read_today(self) -> str:
        path = self._daily_path()
        return path.read_text() if path.exists() else ""

    def append_today(self, content: str) -> None:
        path = self._daily_path()
        if not path.exists():
            header = f"# {date.today().isoformat()}\n\n"
            path.write_text(header + content + "\n")
        else:
            existing = path.read_text()
            path.write_text(existing + "\n" + content + "\n")

    def read_recent(self, days: int = 7) -> str:
        parts = []
        today = date.today()
        for i in range(days):
            d = today - timedelta(days=i)
            path = self._daily_path(d)
            if path.exists():
                parts.append(path.read_text())
        return "\n\n---\n\n".join(parts)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory.py -v`
Expected: PASS (7 passed)

**Step 5: Commit**

```bash
git add agentkit/memory.py tests/test_memory.py
git commit -m "feat: memory system with daily + long-term storage"
```

---

### Task 6: Context Builder + Orientation Ritual

**Files:**
- Create: `agentkit/context.py`
- Create: `tests/test_context.py`

The context builder assembles prompts from identity, memory, and tools. It includes an **orientation ritual** — every invocation starts by reading recent memory and progress, preventing the "cold start" problem identified in Anthropic's long-running agent research.

**Step 1: Write the failing test**

```python
# tests/test_context.py
from pathlib import Path

from agentkit.context import ContextBuilder
from agentkit.config import Config
from agentkit.memory import Memory


def test_build_context_includes_identity(tmp_path):
    profile_dir = tmp_path / "profiles" / "test"
    profile_dir.mkdir(parents=True)
    (profile_dir / "identity.md").write_text("You are a helpful analyst.")

    config = Config(profile="test", project_root=tmp_path)
    memory = Memory(memory_dir=tmp_path / "memory")
    ctx = ContextBuilder(config=config, memory=memory)
    system_prompt = ctx.build_system_prompt()
    assert "helpful analyst" in system_prompt


def test_build_context_includes_memory(tmp_path):
    profile_dir = tmp_path / "profiles" / "test"
    profile_dir.mkdir(parents=True)
    (profile_dir / "identity.md").write_text("You are an agent.")

    memory = Memory(memory_dir=tmp_path / "memory")
    memory.write_long_term("Key fact: markets close at 4pm.")
    memory.append_today("Observation: volatility is high.")

    config = Config(profile="test", project_root=tmp_path)
    ctx = ContextBuilder(config=config, memory=memory)
    system_prompt = ctx.build_system_prompt()
    assert "markets close at 4pm" in system_prompt
    assert "volatility is high" in system_prompt


def test_build_context_includes_tools(tmp_path):
    profile_dir = tmp_path / "profiles" / "test"
    profile_dir.mkdir(parents=True)
    (profile_dir / "identity.md").write_text("Agent")
    (profile_dir / "tools.md").write_text("You can query Grafana.")

    config = Config(profile="test", project_root=tmp_path)
    memory = Memory(memory_dir=tmp_path / "memory")
    ctx = ContextBuilder(config=config, memory=memory)
    system_prompt = ctx.build_system_prompt()
    assert "query Grafana" in system_prompt


def test_build_context_includes_orientation(tmp_path):
    """Orientation ritual: recent memory summary is always included."""
    profile_dir = tmp_path / "profiles" / "test"
    profile_dir.mkdir(parents=True)
    (profile_dir / "identity.md").write_text("Agent")

    memory = Memory(memory_dir=tmp_path / "memory")
    memory.append_today("Earlier today: deployed v2.1")

    config = Config(profile="test", project_root=tmp_path)
    ctx = ContextBuilder(config=config, memory=memory)
    system_prompt = ctx.build_system_prompt()
    assert "deployed v2.1" in system_prompt


def test_build_task_prompt(tmp_path):
    profile_dir = tmp_path / "profiles" / "test"
    profile_dir.mkdir(parents=True)
    (profile_dir / "identity.md").write_text("Agent")

    config = Config(profile="test", project_root=tmp_path)
    memory = Memory(memory_dir=tmp_path / "memory")
    ctx = ContextBuilder(config=config, memory=memory)
    prompt = ctx.build_task_prompt("Analyze BTC conditions")
    assert "Analyze BTC conditions" in prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_context.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentkit/context.py
"""Context builder — assembles prompts from identity, memory, and tools.

Includes an orientation ritual: every invocation starts by reading recent
memory and progress, preventing the 'cold start' problem.
"""

from agentkit.config import Config
from agentkit.memory import Memory


class ContextBuilder:
    def __init__(self, config: Config, memory: Memory):
        self.config = config
        self.memory = memory

    def _read_profile_file(self, name: str) -> str:
        path = self.config.profile_dir / name
        return path.read_text() if path.exists() else ""

    def build_system_prompt(self) -> str:
        sections = []

        # Orientation ritual — what happened recently?
        recent = self.memory.read_recent(days=3)
        if recent:
            sections.append(f"## Recent Context (Orientation)\n\n{recent}")

        identity = self._read_profile_file("identity.md")
        if identity:
            sections.append(f"## Identity\n\n{identity}")

        tools = self._read_profile_file("tools.md")
        if tools:
            sections.append(f"## Available Tools\n\n{tools}")

        long_term = self.memory.read_long_term()
        if long_term:
            sections.append(f"## Long-Term Memory\n\n{long_term}")

        return "\n\n---\n\n".join(sections)

    def build_task_prompt(self, task: str) -> str:
        return (
            f"## Task\n\n{task}\n\n"
            "## Instructions\n\n"
            "1. Analyze the task using your identity, memory, and tools.\n"
            "2. Take action or provide analysis.\n"
            "3. Verify your output is correct before responding.\n"
            "4. State any observations worth remembering (prefix with MEMORY:).\n"
            "5. State any messages to send to the user (prefix with TELEGRAM:).\n"
            "6. State any jobs to schedule (prefix with SCHEDULE:).\n"
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_context.py -v`
Expected: PASS (5 passed)

**Step 5: Commit**

```bash
git add agentkit/context.py tests/test_context.py
git commit -m "feat: context builder with orientation ritual"
```

---

## Phase 3: Agent Core

### Task 7: Mailbox (Task Queue)

**Files:**
- Create: `agentkit/mailbox.py`
- Create: `tests/test_mailbox.py`

**Step 1: Write the failing test**

```python
# tests/test_mailbox.py
from agentkit.mailbox import Mailbox, TaskStatus


def test_enqueue_and_dequeue(tmp_path):
    db_path = tmp_path / "test.db"
    mb = Mailbox(db_path)
    task_id = mb.enqueue("Analyze BTC market", source="telegram")
    assert task_id > 0

    task = mb.dequeue()
    assert task is not None
    assert task["content"] == "Analyze BTC market"
    assert task["source"] == "telegram"
    assert task["status"] == TaskStatus.PROCESSING


def test_dequeue_empty(tmp_path):
    db_path = tmp_path / "test.db"
    mb = Mailbox(db_path)
    assert mb.dequeue() is None


def test_complete_task(tmp_path):
    db_path = tmp_path / "test.db"
    mb = Mailbox(db_path)
    mb.enqueue("Do something", source="cli")
    task = mb.dequeue()
    mb.complete(task["id"], result="Done successfully")

    history = mb.history(limit=1)
    assert len(history) == 1
    assert history[0]["status"] == TaskStatus.DONE
    assert history[0]["result"] == "Done successfully"


def test_fail_task(tmp_path):
    db_path = tmp_path / "test.db"
    mb = Mailbox(db_path)
    mb.enqueue("Bad task", source="cron")
    task = mb.dequeue()
    mb.fail(task["id"], error="Something broke")

    history = mb.history(limit=1)
    assert history[0]["status"] == TaskStatus.FAILED


def test_fifo_order(tmp_path):
    db_path = tmp_path / "test.db"
    mb = Mailbox(db_path)
    mb.enqueue("First", source="cli")
    mb.enqueue("Second", source="cli")

    task1 = mb.dequeue()
    task2 = mb.dequeue()
    assert task1["content"] == "First"
    assert task2["content"] == "Second"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_mailbox.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentkit/mailbox.py
"""Mailbox — SQLite-backed task queue."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Mailbox:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def enqueue(self, content: str, source: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            "INSERT INTO tasks (content, source, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (content, source, TaskStatus.PENDING, now, now),
        )
        self.conn.commit()
        return cursor.lastrowid

    def dequeue(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM tasks WHERE status = ? ORDER BY id ASC LIMIT 1",
            (TaskStatus.PENDING,),
        ).fetchone()
        if row is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
            (TaskStatus.PROCESSING, now, row["id"]),
        )
        self.conn.commit()
        return dict(row) | {"status": TaskStatus.PROCESSING}

    def complete(self, task_id: int, result: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE tasks SET status = ?, result = ?, updated_at = ? WHERE id = ?",
            (TaskStatus.DONE, result, now, task_id),
        )
        self.conn.commit()

    def fail(self, task_id: int, error: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE tasks SET status = ?, result = ?, updated_at = ? WHERE id = ?",
            (TaskStatus.FAILED, error, now, task_id),
        )
        self.conn.commit()

    def history(self, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_mailbox.py -v`
Expected: PASS (5 passed)

**Step 5: Commit**

```bash
git add agentkit/mailbox.py tests/test_mailbox.py
git commit -m "feat: SQLite-backed mailbox task queue"
```

---

### Task 8: Tool System

**Files:**
- Create: `agentkit/tools/__init__.py`
- Create: `agentkit/tools/base.py`
- Create: `agentkit/tools/shell.py`
- Create: `tests/test_tools.py`

**Step 1: Write the failing test**

```python
# tests/test_tools.py
from agentkit.tools.base import Tool, ToolRegistry
from agentkit.tools.shell import ShellTool


def test_tool_registry():
    registry = ToolRegistry()
    tool = ShellTool()
    registry.register(tool)
    assert "shell" in registry.list_tools()
    assert registry.get("shell") is tool


def test_tool_registry_describe():
    registry = ToolRegistry()
    registry.register(ShellTool())
    desc = registry.describe_all()
    assert "shell" in desc.lower()


def test_shell_tool_execute():
    tool = ShellTool()
    result = tool.execute(command="echo hello")
    assert "hello" in result


def test_shell_tool_metadata():
    tool = ShellTool()
    assert tool.name == "shell"
    assert tool.description
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentkit/tools/base.py
"""Tool interface and registry."""

from abc import ABC, abstractmethod


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def execute(self, **kwargs) -> str: ...


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def describe_all(self) -> str:
        lines = []
        for tool in self._tools.values():
            lines.append(f"- **{tool.name}**: {tool.description}")
        return "\n".join(lines)
```

```python
# agentkit/tools/shell.py
"""Shell command execution tool."""

import subprocess
from agentkit.tools.base import Tool


class ShellTool(Tool):
    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "Execute a shell command and return stdout."

    def execute(self, **kwargs) -> str:
        command = kwargs.get("command", "")
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout.strip() or result.stderr.strip()
        except subprocess.TimeoutExpired:
            return "ERROR: Command timed out after 30s"
```

```python
# agentkit/tools/__init__.py
"""Tool registry and built-in tools."""

from agentkit.tools.base import Tool, ToolRegistry
from agentkit.tools.shell import ShellTool

__all__ = ["Tool", "ToolRegistry", "ShellTool"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools.py -v`
Expected: PASS (4 passed)

**Step 5: Commit**

```bash
git add agentkit/tools/ tests/test_tools.py
git commit -m "feat: tool system with registry and shell tool"
```

---

### Task 9: Agent Core Loop (with Verify Step + SCHEDULE: Directive)

**Files:**
- Create: `agentkit/agent.py`
- Create: `tests/test_agent.py`

The agent loop follows the **gather→act→verify→iterate** pattern from Anthropic's research. The verify step checks the output against basic success criteria before declaring the task done. The agent also parses `SCHEDULE:` directives alongside `MEMORY:` and `TELEGRAM:`.

**Step 1: Write the failing test**

```python
# tests/test_agent.py
from pathlib import Path
from unittest.mock import patch

from agentkit.agent import Agent
from agentkit.claude import ToolMode
from agentkit.config import Config


def _setup_agent(tmp_path: Path) -> Agent:
    """Create an agent with a minimal profile."""
    profile_dir = tmp_path / "profiles" / "test"
    profile_dir.mkdir(parents=True)
    (profile_dir / "identity.md").write_text("You are a test agent.")
    (tmp_path / "data").mkdir(exist_ok=True)

    config = Config(profile="test", project_root=tmp_path)
    return Agent(config)


def test_agent_process_task(tmp_path):
    agent = _setup_agent(tmp_path)
    agent.mailbox.enqueue("What is 2+2?", source="test")

    with patch("agentkit.claude.invoke_claude", return_value="The answer is 4."):
        processed = agent.process_next()
        assert processed is True


def test_agent_process_empty_queue(tmp_path):
    agent = _setup_agent(tmp_path)
    processed = agent.process_next()
    assert processed is False


def test_agent_updates_memory_on_memory_prefix(tmp_path):
    agent = _setup_agent(tmp_path)
    agent.mailbox.enqueue("Learn something", source="test")

    response = "Analysis done.\nMEMORY: BTC spread is consistently $1.01"
    with patch("agentkit.claude.invoke_claude", return_value=response):
        agent.process_next()

    long_term = agent.memory.read_long_term()
    assert "BTC spread" in long_term


def test_agent_captures_telegram_messages(tmp_path):
    agent = _setup_agent(tmp_path)
    agent.mailbox.enqueue("Report status", source="test")

    response = "Here's the report.\nTELEGRAM: System is healthy, no action needed."
    with patch("agentkit.claude.invoke_claude", return_value=response):
        agent.process_next()

    assert len(agent.pending_messages) == 1
    assert "System is healthy" in agent.pending_messages[0]


def test_agent_captures_schedule_directives(tmp_path):
    agent = _setup_agent(tmp_path)
    agent.mailbox.enqueue("Set up monitoring", source="test")

    response = "Done.\nSCHEDULE: evaluate --profile trading 0 */6 * * *"
    with patch("agentkit.claude.invoke_claude", return_value=response):
        agent.process_next()

    assert len(agent.pending_schedules) == 1
    assert "evaluate" in agent.pending_schedules[0]


def test_agent_process_with_tool_mode(tmp_path):
    agent = _setup_agent(tmp_path)
    agent.mailbox.enqueue("Read something", source="test")

    with patch("agentkit.claude.invoke_claude", return_value="Done.") as mock_claude:
        agent.process_next(tool_mode=ToolMode.READWRITE)
        _, kwargs = mock_claude.call_args
        assert kwargs.get("tool_mode") == ToolMode.READWRITE
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentkit/agent.py
"""Core agent loop — gather → act → verify → iterate."""

import logging
from agentkit.claude import invoke_claude, ClaudeError, ToolMode
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
        self.pending_schedules: list[str] = []

    def process_next(self, *, tool_mode: ToolMode = ToolMode.READONLY) -> bool:
        """Process the next task from the mailbox. Returns True if a task was processed."""
        task = self.mailbox.dequeue()
        if task is None:
            return False

        log.info("Processing task %d: %s", task["id"], task["content"][:80])

        try:
            system_prompt = self.context.build_system_prompt()
            task_prompt = self.context.build_task_prompt(task["content"])
            response = invoke_claude(
                task_prompt, system_prompt=system_prompt, tool_mode=tool_mode
            )
            self._process_response(response)
            self.mailbox.complete(task["id"], result=response[:500])
            self.memory.append_today(f"Task: {task['content'][:100]}\nResult: {response[:200]}")
            log.info("Task %d completed", task["id"])
        except ClaudeError as e:
            self.mailbox.fail(task["id"], error=str(e))
            log.error("Task %d failed: %s", task["id"], e)

        return True

    def _process_response(self, response: str) -> None:
        """Extract MEMORY:, TELEGRAM:, and SCHEDULE: directives from Claude's response."""
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.startswith("MEMORY:"):
                content = stripped[len("MEMORY:"):].strip()
                self.memory.append_long_term(content)
            elif stripped.startswith("TELEGRAM:"):
                content = stripped[len("TELEGRAM:"):].strip()
                self.pending_messages.append(content)
            elif stripped.startswith("SCHEDULE:"):
                content = stripped[len("SCHEDULE:"):].strip()
                self.pending_schedules.append(content)

    def run_all(self, *, tool_mode: ToolMode = ToolMode.READONLY) -> int:
        """Process all pending tasks. Returns count of tasks processed."""
        count = 0
        while self.process_next(tool_mode=tool_mode):
            count += 1
        return count
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py -v`
Expected: PASS (6 passed)

**Step 5: Commit**

```bash
git add agentkit/agent.py tests/test_agent.py
git commit -m "feat: core agent loop with verify step + SCHEDULE: directive"
```

---

## Phase 4: Playground & Smoke Tests

**Philosophy:** Validate the framework with REAL Claude calls as soon as the core agent loop works. No mocks for integration tests. This was originally Phase 8 — moved up because once the core works, we should prove it before adding Telegram, scheduling, etc.

### Task 10: Playground Profile

The playground profile is a safe, self-contained test environment. No external dependencies (no Grafana, no SSH, no Telegram needed). Just Claude + filesystem. Ships with agentkit.

**Files:**
- Create: `profiles/playground/identity.md`
- Create: `profiles/playground/tools.md`
- Create: `profiles/playground/evaluation.md`

**Step 1: Write playground identity.md**

```markdown
# Playground Agent

You are a test agent for verifying the agentkit framework works correctly.

## Rules

1. Always respond concisely and precisely.
2. Always include at least one MEMORY: line with a factual observation from your work.
3. Always include at least one TELEGRAM: line summarizing your response in one sentence.
4. When asked to read files, read them and report exact data (line counts, headings, etc.).
5. Never make up data — if a file doesn't exist, say so.
6. Verify your output before responding — check that counts are accurate and facts are correct.

## Your Project

You are running inside the agentkit framework project.

Key files you can read:
- CLAUDE.md — project manifest
- pyproject.toml — project config
- memory/MEMORY.md — your long-term memory
- memory/daily/ — your daily observations
```

**Step 2: Write playground tools.md**

```markdown
# Available Tools

## File System

You can read any file in the project directory using your Read, Glob, and Grep tools.

## Web Search

You can search the web for information using WebSearch and WebFetch tools.

Note: In READONLY mode (default), you can only observe. You cannot modify files or run shell commands.
```

**Step 3: Write playground evaluation.md**

```markdown
# Playground Evaluation

Run a self-check of the agentkit framework. This verifies memory, file access, and response directives all work.

## Steps

1. Read CLAUDE.md and count the number of second-level headings (## lines).
2. Read memory/MEMORY.md and summarize what long-term knowledge exists (or note it's empty).
3. Check if data/agentkit.db exists and report.
4. Read today's daily memory file if it exists.
5. Verify your findings are accurate before responding.

## Output

- MEMORY: Playground evaluation ran successfully. Found N headings in CLAUDE.md, system operational.
- TELEGRAM: Playground self-check passed. CLAUDE.md has N headings, system operational.
```

**Step 4: Commit**

```bash
git add profiles/playground/
git commit -m "feat: playground profile for real integration testing"
```

---

### Task 11: Smoke Test Script

The smoke test runs the REAL full pipeline: Claude CLI is invoked, memory is written, mailbox is updated. No mocks. This validates agentkit works end-to-end.

**Note:** This requires a minimal CLI (`agentkit task` command). We implement a minimal `cli.py` here, then extend it fully in Phase 5.

**Files:**
- Create: `agentkit/cli.py` (minimal — just `task` and `evaluate`)
- Create: `agentkit/__main__.py`
- Create: `scripts/smoke-test.sh`

**Step 1: Write minimal CLI**

```python
# agentkit/cli.py
"""CLI entry point — task, evaluate, run, schedule, self-improve."""

import argparse
import logging
from pathlib import Path

from agentkit.agent import Agent
from agentkit.claude import ToolMode
from agentkit.config import Config
from agentkit.telegram_bot import TelegramBot


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentkit", description="Autonomous agent framework")

    sub = parser.add_subparsers(dest="command")

    # agentkit task "prompt"
    task_cmd = sub.add_parser("task", help="Process a single task")
    task_cmd.add_argument("prompt", help="Task to process")
    task_cmd.add_argument("--profile", default="playground")
    task_cmd.add_argument("--write", action="store_true", help="Enable READWRITE mode")

    # agentkit evaluate
    eval_cmd = sub.add_parser("evaluate", help="Run evaluation cycle (always READONLY)")
    eval_cmd.add_argument("--profile", default="playground")

    # agentkit run
    run_cmd = sub.add_parser("run", help="Run in polling mode")
    run_cmd.add_argument("--profile", default="playground")

    # agentkit self-improve
    improve_cmd = sub.add_parser("self-improve", help="Run self-improvement cycle (READWRITE)")

    # agentkit schedule
    sched_cmd = sub.add_parser("schedule", help="Manage scheduled jobs")
    sched_sub = sched_cmd.add_subparsers(dest="schedule_action")
    sched_sub.add_parser("list", help="List all scheduled jobs")
    add_cmd = sched_sub.add_parser("add", help="Add a scheduled job")
    add_cmd.add_argument("--name", required=True)
    add_cmd.add_argument("--command", required=True)
    add_cmd.add_argument("--profile", default="playground")
    add_cmd.add_argument("--cron", required=True)
    sched_sub.add_parser("remove", help="Remove a scheduled job").add_argument("--name", required=True)
    sched_sub.add_parser("sync", help="Sync schedule to crontab")

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    project_root = Path(__file__).parent.parent
    config = Config(
        profile=getattr(args, "profile", "playground"),
        project_root=project_root,
    )

    if args.command == "task":
        tool_mode = ToolMode.READWRITE if args.write else ToolMode.READONLY
        agent = Agent(config)
        agent.mailbox.enqueue(args.prompt, source="cli")
        agent.process_next(tool_mode=tool_mode)
        _send_pending(config, agent)

    elif args.command == "evaluate":
        agent = Agent(config)
        evaluation_path = config.profile_dir / "evaluation.md"
        if evaluation_path.exists():
            prompt = evaluation_path.read_text()
        else:
            prompt = "Run a full evaluation of the current system state. Report findings."
        agent.mailbox.enqueue(prompt, source="cron")
        agent.process_next(tool_mode=ToolMode.READONLY)
        _send_pending(config, agent)

    elif args.command == "self-improve":
        from agentkit.evolve import self_improve
        self_improve(config)

    elif args.command == "schedule":
        from agentkit.scheduler import Scheduler
        scheduler = Scheduler(config.schedule_path)
        if args.schedule_action == "list":
            scheduler.print_jobs()
        elif args.schedule_action == "add":
            scheduler.add_job(args.name, args.command, args.profile, args.cron)
        elif args.schedule_action == "remove":
            scheduler.remove_job(args.name)
        elif args.schedule_action == "sync":
            scheduler.sync_crontab()

    elif args.command == "run":
        print("Polling mode not yet implemented. Use 'task' or 'evaluate'.")

    else:
        parser.print_help()


def _send_pending(config: Config, agent: Agent) -> None:
    if agent.pending_messages and config.telegram_bot_token:
        bot = TelegramBot(token=config.telegram_bot_token, chat_id=config.telegram_chat_id)
        for msg in agent.pending_messages:
            bot.send_sync(msg)
        agent.pending_messages.clear()
```

```python
# agentkit/__main__.py
"""Allow running as: python -m agentkit"""

from agentkit.cli import main

main()
```

**Step 2: Write the smoke test**

```bash
#!/usr/bin/env bash
# scripts/smoke-test.sh — Full end-to-end verification with REAL Claude CLI
#
# This script tests the entire agentkit pipeline:
#   1. Claude CLI is available and responds
#   2. A task can be enqueued, processed, and completed
#   3. Memory is updated (both daily and long-term)
#   4. The evaluation cycle works
#   5. Task history is queryable
#
# NO MOCKS. If something fails here, it's broken for real.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

AGENTKIT=".venv/bin/agentkit"
PASS=0
FAIL=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "==========================================="
echo "  agentkit Smoke Test (REAL, NO MOCKS)"
echo "==========================================="
echo ""

# Clean state for reproducible test
echo "Cleaning previous state..."
make clean 2>/dev/null || true
echo ""

# --- Test 1: Claude CLI available ---
echo "1. Checking Claude CLI..."
if claude --version > /dev/null 2>&1; then
    pass "Claude CLI found: $(claude --version 2>&1 | head -1)"
else
    fail "Claude CLI not on PATH"
    echo "   Cannot continue without Claude CLI. Install it first."
    exit 1
fi
echo ""

# --- Test 2: agentkit CLI installed ---
echo "2. Checking agentkit CLI..."
if $AGENTKIT --help > /dev/null 2>&1; then
    pass "agentkit CLI installed"
else
    fail "agentkit not installed. Run: make install"
    exit 1
fi
echo ""

# --- Test 3: Simple task with playground profile ---
echo "3. Running playground task (REAL Claude call)..."
echo "   This calls Claude Opus 4.6 — may take 30-60 seconds..."
$AGENTKIT task --profile playground \
    "Read the file CLAUDE.md in this project directory and count the second-level headings (lines starting with ##). Respond with exactly: MEMORY: CLAUDE.md has N second-level headings. TELEGRAM: Counted N headings in CLAUDE.md."

# Verify memory was updated
if [ -f "memory/daily/$(date +%Y-%m-%d).md" ]; then
    pass "Daily memory file created"
else
    fail "Daily memory file NOT created"
fi

if grep -q "CLAUDE.md" "memory/MEMORY.md" 2>/dev/null; then
    pass "Long-term memory updated with CLAUDE.md observation"
else
    fail "Long-term memory NOT updated (expected CLAUDE.md mention)"
fi
echo ""

# --- Test 4: Task recorded in mailbox ---
echo "4. Checking mailbox database..."
if [ -f "data/agentkit.db" ]; then
    TASK_COUNT=$(sqlite3 data/agentkit.db "SELECT COUNT(*) FROM tasks WHERE status = 'done';")
    if [ "$TASK_COUNT" -ge 1 ]; then
        pass "Found $TASK_COUNT completed task(s) in mailbox"
    else
        fail "No completed tasks found in mailbox"
    fi
else
    fail "Database not created at data/agentkit.db"
fi
echo ""

# --- Test 5: Evaluation cycle ---
echo "5. Running playground evaluation (REAL Claude call)..."
echo "   Another real Claude call — 30-60 seconds..."
$AGENTKIT evaluate --profile playground

EVAL_COUNT=$(sqlite3 data/agentkit.db "SELECT COUNT(*) FROM tasks WHERE source = 'cron' AND status = 'done';")
if [ "$EVAL_COUNT" -ge 1 ]; then
    pass "Evaluation task completed successfully"
else
    fail "Evaluation task did not complete"
fi
echo ""

# --- Test 6: Second task accumulates memory ---
echo "6. Running second task to verify memory accumulation..."
$AGENTKIT task --profile playground \
    "Read memory/MEMORY.md and tell me how many lines of long-term memory exist. Respond with: MEMORY: Memory system is accumulating correctly. TELEGRAM: Memory has N lines of knowledge."

MEMORY_LINES=$(wc -l < memory/MEMORY.md 2>/dev/null || echo "0")
if [ "$MEMORY_LINES" -ge 2 ]; then
    pass "Memory accumulating: $MEMORY_LINES lines in long-term memory"
else
    fail "Memory not accumulating (only $MEMORY_LINES lines)"
fi
echo ""

# --- Test 7: Task history queryable ---
echo "7. Verifying task history..."
TOTAL_TASKS=$(sqlite3 data/agentkit.db "SELECT COUNT(*) FROM tasks;")
DONE_TASKS=$(sqlite3 data/agentkit.db "SELECT COUNT(*) FROM tasks WHERE status = 'done';")
FAILED_TASKS=$(sqlite3 data/agentkit.db "SELECT COUNT(*) FROM tasks WHERE status = 'failed';")

if [ "$TOTAL_TASKS" -ge 3 ]; then
    pass "Task history: $TOTAL_TASKS total, $DONE_TASKS done, $FAILED_TASKS failed"
else
    fail "Expected at least 3 tasks, found $TOTAL_TASKS"
fi
echo ""

# --- Summary ---
echo "==========================================="
echo "  Results: $PASS passed, $FAIL failed"
echo "==========================================="
echo ""

# Show state for inspection
echo "Current memory state:"
echo "--- Long-term (memory/MEMORY.md) ---"
cat memory/MEMORY.md 2>/dev/null || echo "(empty)"
echo ""
echo "--- Today (memory/daily/$(date +%Y-%m-%d).md) ---"
cat "memory/daily/$(date +%Y-%m-%d).md" 2>/dev/null || echo "(empty)"
echo ""
echo "Task history:"
sqlite3 -header -column data/agentkit.db \
    "SELECT id, status, source, substr(content, 1, 50) as task, created_at FROM tasks ORDER BY id;"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "Some tests failed. Check output above."
    exit 1
else
    echo "All smoke tests passed. agentkit is operational."
    exit 0
fi
```

**Step 3: Make executable and commit**

```bash
chmod +x scripts/smoke-test.sh
git add agentkit/cli.py agentkit/__main__.py scripts/smoke-test.sh
git commit -m "feat: CLI + real E2E smoke test — no mocks, real Claude CLI"
```

---

## Phase 5: Communication

### Task 12: Telegram Integration

**Files:**
- Create: `agentkit/telegram_bot.py`
- Create: `tests/test_telegram.py`

**Step 1: Write the failing test**

```python
# tests/test_telegram.py
from unittest.mock import AsyncMock
import pytest

from agentkit.telegram_bot import TelegramBot


def test_telegram_bot_init():
    bot = TelegramBot(token="fake-token", chat_id="12345")
    assert bot.chat_id == "12345"


@pytest.mark.asyncio
async def test_send_message():
    bot = TelegramBot(token="fake-token", chat_id="12345")
    mock_bot = AsyncMock()
    bot._bot = mock_bot

    await bot.send("Hello from agentkit")
    mock_bot.send_message.assert_called_once_with(
        chat_id="12345", text="Hello from agentkit"
    )


def test_truncate_long_message():
    bot = TelegramBot(token="fake-token", chat_id="12345")
    long_msg = "x" * 5000
    truncated = bot._truncate(long_msg)
    assert len(truncated) <= 4096
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_telegram.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

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

    async def send(self, text: str) -> None:
        """Send a message to the configured chat."""
        text = self._truncate(text)
        try:
            await self._bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            log.error("Failed to send Telegram message: %s", e)

    def send_sync(self, text: str) -> None:
        """Synchronous wrapper for send."""
        asyncio.run(self.send(text))

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_MESSAGE_LENGTH:
            return text
        return text[: MAX_MESSAGE_LENGTH - 20] + "\n\n... (truncated)"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_telegram.py -v`
Expected: PASS (3 passed)

**Step 5: Commit**

```bash
git add agentkit/telegram_bot.py tests/test_telegram.py
git commit -m "feat: Telegram bot integration for sending messages"
```

---

### Task 13: CLI Tests

**Files:**
- Create: `tests/test_cli.py`

**Step 1: Write tests for the full CLI**

```python
# tests/test_cli.py
from agentkit.cli import create_parser


def test_parser_task_command():
    parser = create_parser()
    args = parser.parse_args(["task", "--profile", "trading", "Analyze BTC"])
    assert args.command == "task"
    assert args.profile == "trading"
    assert args.prompt == "Analyze BTC"


def test_parser_task_write_flag():
    parser = create_parser()
    args = parser.parse_args(["task", "--write", "Fix something"])
    assert args.write is True


def test_parser_task_readonly_default():
    parser = create_parser()
    args = parser.parse_args(["task", "Read something"])
    assert args.write is False


def test_parser_evaluate_command():
    parser = create_parser()
    args = parser.parse_args(["evaluate", "--profile", "trading"])
    assert args.command == "evaluate"
    assert args.profile == "trading"


def test_parser_run_command():
    parser = create_parser()
    args = parser.parse_args(["run", "--profile", "trading"])
    assert args.command == "run"


def test_parser_self_improve_command():
    parser = create_parser()
    args = parser.parse_args(["self-improve"])
    assert args.command == "self-improve"


def test_parser_schedule_list():
    parser = create_parser()
    args = parser.parse_args(["schedule", "list"])
    assert args.command == "schedule"
    assert args.schedule_action == "list"


def test_parser_schedule_sync():
    parser = create_parser()
    args = parser.parse_args(["schedule", "sync"])
    assert args.schedule_action == "sync"


def test_parser_default_profile():
    parser = create_parser()
    args = parser.parse_args(["task", "Do something"])
    assert args.profile == "playground"  # default
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (9 passed)

**Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: CLI parser tests for all commands"
```

---

## Phase 6: Scheduling & Self-Evolution

### Task 14: Crontab-Based Self-Scheduler

**Files:**
- Create: `agentkit/scheduler.py`
- Create: `tests/test_scheduler.py`

The agent manages its own crontab entries. No more static launchd plist. Schedule is stored in JSON, synced to crontab explicitly. Safe — never touches non-agentkit cron entries.

**Step 1: Write the failing test**

```python
# tests/test_scheduler.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentkit.scheduler import Scheduler


def test_load_empty_schedule(tmp_path):
    path = tmp_path / "schedule.json"
    sched = Scheduler(path)
    assert sched.jobs == []


def test_add_job(tmp_path):
    path = tmp_path / "schedule.json"
    sched = Scheduler(path)
    sched.add_job("eval-trading", "evaluate", "trading", "0 */6 * * *")
    assert len(sched.jobs) == 1
    assert sched.jobs[0]["name"] == "eval-trading"


def test_add_job_persists(tmp_path):
    path = tmp_path / "schedule.json"
    sched = Scheduler(path)
    sched.add_job("eval-trading", "evaluate", "trading", "0 */6 * * *")

    sched2 = Scheduler(path)
    assert len(sched2.jobs) == 1


def test_remove_job(tmp_path):
    path = tmp_path / "schedule.json"
    sched = Scheduler(path)
    sched.add_job("eval-trading", "evaluate", "trading", "0 */6 * * *")
    sched.remove_job("eval-trading")
    assert len(sched.jobs) == 0


def test_duplicate_job_name_rejected(tmp_path):
    path = tmp_path / "schedule.json"
    sched = Scheduler(path)
    sched.add_job("eval", "evaluate", "trading", "0 */6 * * *")
    try:
        sched.add_job("eval", "task", "playground", "0 * * * *")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_sync_crontab_builds_correct_entries(tmp_path):
    path = tmp_path / "schedule.json"
    sched = Scheduler(path)
    sched.add_job("eval-trading", "evaluate", "trading", "0 */6 * * *")

    with patch("subprocess.run") as mock_run:
        # Mock reading existing crontab (empty)
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        sched.sync_crontab(agentkit_bin="/usr/local/bin/agentkit")

        # Verify crontab was written
        assert mock_run.call_count == 2  # read + write
        write_call = mock_run.call_args_list[1]
        written = write_call[1].get("input", "")
        assert "agentkit-managed" in written
        assert "0 */6 * * *" in written
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scheduler.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentkit/scheduler.py
"""Crontab-based self-scheduler — agent manages its own cron jobs."""

import json
import subprocess
from pathlib import Path


class Scheduler:
    CRON_TAG = "# agentkit-managed"

    def __init__(self, schedule_path: Path):
        self.schedule_path = schedule_path
        self.jobs = self._load()

    def _load(self) -> list[dict]:
        if self.schedule_path.exists():
            data = json.loads(self.schedule_path.read_text())
            return data.get("jobs", [])
        return []

    def _save(self) -> None:
        self.schedule_path.parent.mkdir(parents=True, exist_ok=True)
        self.schedule_path.write_text(json.dumps({"jobs": self.jobs}, indent=2))

    def add_job(self, name: str, command: str, profile: str, cron: str) -> None:
        if any(j["name"] == name for j in self.jobs):
            raise ValueError(f"Job '{name}' already exists")
        self.jobs.append({
            "name": name,
            "command": command,
            "profile": profile,
            "cron": cron,
            "enabled": True,
        })
        self._save()

    def remove_job(self, name: str) -> None:
        self.jobs = [j for j in self.jobs if j["name"] != name]
        self._save()

    def print_jobs(self) -> None:
        if not self.jobs:
            print("No scheduled jobs.")
            return
        for job in self.jobs:
            status = "enabled" if job.get("enabled", True) else "disabled"
            print(f"  {job['name']}: {job['command']} --profile {job['profile']} "
                  f"[{job['cron']}] ({status})")

    def sync_crontab(self, agentkit_bin: str = "agentkit") -> None:
        """Sync enabled jobs to crontab. Safe — only touches agentkit-managed lines."""
        # Read existing crontab
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True
        )
        existing = result.stdout if result.returncode == 0 else ""

        # Strip old agentkit-managed lines
        lines = [l for l in existing.splitlines() if self.CRON_TAG not in l]

        # Add current enabled jobs
        for job in self.jobs:
            if job.get("enabled", True):
                entry = (f"{job['cron']} {agentkit_bin} {job['command']} "
                         f"--profile {job['profile']} {self.CRON_TAG}")
                lines.append(entry)

        # Write back
        new_crontab = "\n".join(lines) + "\n" if lines else ""
        subprocess.run(
            ["crontab", "-"], input=new_crontab, text=True, check=True
        )
        print(f"Synced {len([j for j in self.jobs if j.get('enabled', True)])} jobs to crontab.")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scheduler.py -v`
Expected: PASS (6 passed)

**Step 5: Commit**

```bash
git add agentkit/scheduler.py tests/test_scheduler.py
git commit -m "feat: crontab-based self-scheduler — agent manages its own cron jobs"
```

---

### Task 15: Self-Evolution System + Tool Generation

**Files:**
- Create: `agentkit/evolve.py`
- Create: `tests/test_evolve.py`

The self-evolution system lets the agent improve its own code AND generate new tools. Safety guardrails: always on a branch, tests must pass, human merges.

**Step 1: Write the failing test**

```python
# tests/test_evolve.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from agentkit.evolve import self_improve, _log_evolution


def test_log_evolution_creates_file(tmp_path):
    log_path = tmp_path / "evolution-log.json"
    _log_evolution(log_path, branch="evolve/test", success=True, summary="Added feature X")

    data = json.loads(log_path.read_text())
    assert len(data) == 1
    assert data[0]["branch"] == "evolve/test"
    assert data[0]["success"] is True


def test_log_evolution_appends(tmp_path):
    log_path = tmp_path / "evolution-log.json"
    _log_evolution(log_path, branch="evolve/1", success=True, summary="First")
    _log_evolution(log_path, branch="evolve/2", success=False, summary="Second")

    data = json.loads(log_path.read_text())
    assert len(data) == 2


@patch("agentkit.evolve.invoke_claude")
@patch("agentkit.evolve.subprocess")
def test_self_improve_reverts_on_test_failure(mock_subprocess, mock_claude, tmp_path):
    """If tests fail after Claude's changes, everything should revert."""
    config = MagicMock()
    config.project_root = tmp_path
    config.evolution_log_path = tmp_path / "evolution-log.json"
    config.telegram_bot_token = ""

    # Mock git commands
    mock_subprocess.run.return_value = MagicMock(returncode=0, stdout="main\n")

    # Mock Claude response
    mock_claude.return_value = "I improved the error handling."

    # Mock test failure
    test_result = MagicMock()
    test_result.returncode = 1  # Tests fail
    mock_subprocess.run.side_effect = [
        MagicMock(returncode=0, stdout="main\n"),  # git branch --show-current
        MagicMock(returncode=0),  # git checkout -b
        MagicMock(returncode=0, stdout="ok"),  # claude call (handled by mock_claude)
        MagicMock(returncode=1),  # make test FAILS
        MagicMock(returncode=0),  # git checkout main
        MagicMock(returncode=0),  # git branch -D
    ]

    # Should not raise, should handle gracefully
    # (exact behavior depends on implementation, but should not crash)


@patch("agentkit.evolve.invoke_claude")
@patch("agentkit.evolve.subprocess")
def test_self_improve_commits_on_success(mock_subprocess, mock_claude, tmp_path):
    """If tests pass, changes should be committed to the branch."""
    config = MagicMock()
    config.project_root = tmp_path
    config.evolution_log_path = tmp_path / "evolution-log.json"
    config.telegram_bot_token = ""

    mock_claude.return_value = "I added input validation."
    mock_subprocess.run.return_value = MagicMock(returncode=0, stdout="")

    # Verify no crash — success path
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolve.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# agentkit/evolve.py
"""Self-evolution — agent improves its own code and generates new tools.

Safety guardrails:
- Always on a branch, never main
- Tests must pass or everything reverts
- Human merges manually (no auto-merge)
- Telegram notification on every attempt
- Evolution log tracks history
- System prompt limits to ONE change per cycle
"""

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from agentkit.claude import invoke_claude, ToolMode, ClaudeError
from agentkit.config import Config

log = logging.getLogger(__name__)

EVOLVE_SYSTEM_PROMPT = """You are improving the agentkit framework. Make ONE focused improvement.

You have READWRITE access to the codebase. You can:
1. Fix a bug you've noticed
2. Improve error handling or robustness
3. Create a new tool (write the tool file + register it)
4. Optimize an existing module
5. Add missing test coverage

Rules:
- ONE change per session. Small, focused, testable.
- All changes must be covered by tests.
- Do not break existing functionality.
- Do not modify this system prompt or the evolution system itself.
- After making changes, explain what you did and why in 2-3 sentences.
"""


def self_improve(config: Config) -> None:
    """Run one self-improvement cycle."""
    project_root = config.project_root
    log_path = config.evolution_log_path
    branch_name = f"evolve/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"

    log.info("Starting self-improvement cycle on branch %s", branch_name)

    # Get current branch
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True, cwd=project_root
    )
    original_branch = result.stdout.strip() or "main"

    # Create evolution branch
    subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=project_root, check=True
    )

    try:
        # Build context from recent state
        context_parts = []

        # Recent memory
        memory_file = project_root / "memory" / "MEMORY.md"
        if memory_file.exists():
            context_parts.append(f"## Recent Memory\n{memory_file.read_text()[:2000]}")

        # Recent errors from evolution log
        if log_path.exists():
            history = json.loads(log_path.read_text())
            recent_failures = [e for e in history[-5:] if not e.get("success")]
            if recent_failures:
                context_parts.append(
                    f"## Recent Failures\n" +
                    "\n".join(f"- {f['summary']}" for f in recent_failures)
                )

        context = "\n\n".join(context_parts) if context_parts else "No prior context."

        # Invoke Claude with READWRITE
        response = invoke_claude(
            "Review the codebase and make ONE focused improvement.",
            system_prompt=EVOLVE_SYSTEM_PROMPT,
            context=context,
            tool_mode=ToolMode.READWRITE,
            timeout=300,
        )

        # Run tests
        test_result = subprocess.run(
            ["make", "test"], capture_output=True, text=True, cwd=project_root
        )

        if test_result.returncode == 0:
            # Tests pass — commit
            subprocess.run(["git", "add", "-A"], cwd=project_root)
            subprocess.run(
                ["git", "commit", "-m", f"evolve: {response[:100]}"],
                cwd=project_root
            )
            _log_evolution(log_path, branch_name, success=True, summary=response[:200])
            _notify(config, f"Self-improvement succeeded on {branch_name}:\n{response[:300]}")
            log.info("Self-improvement succeeded: %s", response[:100])
        else:
            # Tests fail — revert
            subprocess.run(["git", "checkout", "."], cwd=project_root)
            subprocess.run(["git", "clean", "-fd"], cwd=project_root)
            _log_evolution(log_path, branch_name, success=False,
                          summary=f"Tests failed: {test_result.stderr[:200]}")
            _notify(config, f"Self-improvement failed on {branch_name}: tests did not pass")
            log.warning("Self-improvement failed: tests did not pass")

    except (ClaudeError, subprocess.CalledProcessError) as e:
        subprocess.run(["git", "checkout", "."], cwd=project_root)
        subprocess.run(["git", "clean", "-fd"], cwd=project_root)
        _log_evolution(log_path, branch_name, success=False, summary=str(e)[:200])
        _notify(config, f"Self-improvement error on {branch_name}: {e}")
        log.error("Self-improvement error: %s", e)

    finally:
        # Always switch back to original branch
        subprocess.run(["git", "checkout", original_branch], cwd=project_root)
        # If branch was never committed to, delete it
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=project_root, capture_output=True
        )


def _log_evolution(log_path: Path, branch: str, success: bool, summary: str) -> None:
    """Append to evolution log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    if log_path.exists():
        entries = json.loads(log_path.read_text())
    entries.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "success": success,
        "summary": summary,
    })
    log_path.write_text(json.dumps(entries, indent=2))


def _notify(config: Config, message: str) -> None:
    """Send Telegram notification if configured."""
    if config.telegram_bot_token:
        try:
            from agentkit.telegram_bot import TelegramBot
            bot = TelegramBot(token=config.telegram_bot_token, chat_id=config.telegram_chat_id)
            bot.send_sync(message)
        except Exception as e:
            log.warning("Failed to send evolution notification: %s", e)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_evolve.py -v`
Expected: PASS (4 passed)

**Step 5: Commit**

```bash
git add agentkit/evolve.py tests/test_evolve.py
git commit -m "feat: self-evolution system with tool generation and safety guardrails"
```

---

## Phase 7: goodoracle Consumer Setup

### Task 16: Create goodoracle Project Scaffold

**Repo:** `goodoracle` (separate repo from agentkit)

**Step 1: Create repo structure**

```bash
cd /Users/petr.kubelka/git_projects/goodoracle
# Already exists — we reorganize it

mkdir -p goodoracle/tools profiles/trading memory/daily data tests scripts
```

**Step 2: Write pyproject.toml**

```toml
[project]
name = "goodoracle"
version = "0.1.0"
description = "Autonomous trading agent built on agentkit"
requires-python = ">=3.12"
dependencies = [
    "agentkit>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]

[project.scripts]
goodoracle = "goodoracle.cli:main"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 3: Write goodoracle/__init__.py**

```python
"""GoodOracle — autonomous trading agent built on agentkit."""

__version__ = "0.1.0"
```

**Step 4: Write goodoracle/cli.py** (extends agentkit CLI with trading defaults)

```python
# goodoracle/cli.py
"""GoodOracle CLI — extends agentkit with trading defaults."""

import sys
from agentkit.cli import create_parser, main as agentkit_main


def main() -> None:
    """Run agentkit CLI with trading as default profile."""
    # Patch default profile to 'trading' for goodoracle
    parser = create_parser()
    parser.prog = "goodoracle"

    # If no --profile specified, default to trading
    if "--profile" not in sys.argv:
        for i, arg in enumerate(sys.argv):
            if arg in ("task", "evaluate", "run"):
                sys.argv.insert(i + 1, "--profile")
                sys.argv.insert(i + 2, "trading")
                break

    agentkit_main()
```

**Step 5: Write goodoracle/__main__.py**

```python
"""Allow running as: python -m goodoracle"""

from goodoracle.cli import main

main()
```

**Step 6: Write goodoracle CLAUDE.md**

```markdown
# GoodOracle

## What Is This?

Autonomous trading agent built on **agentkit**. Monitors Polymarket prediction
markets, evaluates strategies, maintains memory, and communicates via Telegram.

## Architecture

GoodOracle is a consumer of agentkit. It provides:
- Trading profile (identity, tools, evaluation cycle)
- Grafana query tool
- CLI with trading defaults

agentkit provides the framework: agent loop, memory, mailbox, scheduling, evolution, Telegram.

## Running

```bash
# Single task (READONLY default)
goodoracle task "Analyze BTC market conditions"

# Single task with write access
goodoracle task --write "Update the hypothesis file"

# Evaluation cycle (every 6h via cron, always READONLY)
goodoracle evaluate

# Self-improvement (READWRITE, on a branch)
goodoracle self-improve

# Schedule management
goodoracle schedule list
goodoracle schedule sync
```

## File Ownership

Trading-specific data stays in the trading repo. GoodOracle stores agent-level knowledge only.

| What | Where |
|------|-------|
| HYPOTHESES.md | trading/docs/ |
| TRADING-JOURNAL.md | trading/docs/ |
| config/default.toml | trading/config/ |
| Long-term memory | goodoracle/memory/MEMORY.md |
| Daily observations | goodoracle/memory/daily/ |
| Task history | goodoracle/data/goodoracle.db |
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: goodoracle consumer scaffold depending on agentkit"
```

---

### Task 17: Trading Profile

**Files:**
- Create: `profiles/trading/identity.md`
- Create: `profiles/trading/tools.md`
- Create: `profiles/trading/evaluation.md`

**Step 1: Write identity.md**

The trading profile REFERENCES files in the trading repo — it does NOT duplicate them.

```markdown
# Trading Oracle

You are an autonomous trading analyst managing a Polymarket prediction market system.

## Your Mission

Maximize profit on Polymarket prediction markets. You think in expected value:

    EV = (win_rate * avg_win) - (loss_rate * avg_loss) - fees - slippage

## Your System

- **Infrastructure:** Rust trading bots on AWS (t3.micro, eu-west-2)
- **Markets:** Polymarket BTC 15-minute prediction markets (expanding to ETH and others)
- **Data:** Binance real-time feed + Polymarket CLOB order book
- **Monitoring:** Grafana Cloud (Prometheus metrics + Loki logs)

## How You Think

1. **Hypothesis-driven:** Every strategy is a falsifiable hypothesis. Define pass/fail criteria BEFORE testing.
2. **Evidence-based:** Decisions come from data, not intuition. Use your tools to query Grafana, read files, check logs.
3. **ELI5 communication:** Explain everything simply. Numbers first, then interpretation.
4. **Risk-aware:** Know your edge, know when it decays, know when to stop.
5. **Verify before reporting:** Always check that your findings are accurate before responding.

## Your Counterparties

- Market makers (tight spreads, fast execution)
- LLM bots (herd behavior, exploitable via mean reversion)
- Retail humans (slow, emotional, predictable)
- Other arb bots (competing for same edge)

## Key Files (in trading repo — READ, do not duplicate)

These files live in the trading repo. Read them for context, update them when warranted:
- Hypotheses: /Users/petr.kubelka/git_projects/trading/docs/HYPOTHESES.md
- Journal: /Users/petr.kubelka/git_projects/trading/docs/TRADING-JOURNAL.md
- Config: /Users/petr.kubelka/git_projects/trading/config/default.toml
- Status: /Users/petr.kubelka/git_projects/trading/docs/plans/paper_trading_mvp/STATUS.md
- Implementation plan: /Users/petr.kubelka/git_projects/trading/docs/implementation-plan.md
- CLAUDE.md: /Users/petr.kubelka/git_projects/trading/CLAUDE.md
```

**Step 2: Write tools.md**

The tools description says WHAT is available, not HOW to use it. Claude figures out the HOW from its tool access (Read, Glob, Grep, Grafana tool, etc.).

```markdown
# Available Tools

## Grafana Queries

Query live infrastructure metrics via the Grafana API. Available queries:
- **pnl** — current paper trading P&L
- **latency** — execution latency metrics
- **health** — process health and uptime
- **errors** — recent error counts from Loki logs

## File System Access

Read and analyze files in both the trading repo and goodoracle repo:
- Trading project: `/Users/petr.kubelka/git_projects/trading/`
- Agent memory: `/Users/petr.kubelka/git_projects/goodoracle/memory/`

## Web Search

Search for market news, competitor analysis, strategy research, and regulatory updates.

## SSH to Server (when in READWRITE mode)

Check process health, read logs, inspect the running trading system on the AWS instance.
Connection details in: `/Users/petr.kubelka/git_projects/trading/deploy/.server-info`
```

**Step 3: Write evaluation.md**

The evaluation describes WHAT to check, not HOW. Claude has tool access during evaluation and uses its available tools to figure out the mechanics.

```markdown
# Evaluation Cycle

Run this evaluation every 6 hours. Report findings via Telegram.

## Steps

### 1. Health Check
- Are bot processes alive? (query infrastructure health)
- Are data feeds flowing? (check recent data counts)
- Any errors in last 6 hours? (query error logs)
- Memory/CPU stable? (query process metrics)

### 2. Performance Review
- Paper PnL since last evaluation
- Number of opportunities detected vs. acted on
- Current spread levels (combined YES+NO ask)
- Fill rate for paper trades

### 3. Hypothesis Update
- Read current hypotheses from the trading repo
- Check if new evidence supports state transitions
- Propose new hypotheses if market conditions changed
- Update hypothesis file if warranted

### 4. Market Analysis
- Current BTC volatility and direction
- Polymarket spread patterns (tightening or widening?)
- Any new market types worth exploring? (ETH, other events)
- External factors (exchange news, regulatory changes)

### 5. Roadmap Check
- What was the last development task completed?
- What should be worked on next?
- Any blocked tasks that need attention?
- Priority: what gives the highest EV improvement per hour of dev work?

### 6. Recommendations
For each recommendation, provide:
- **What:** Specific action to take
- **Why (ELI5):** Simple explanation with numbers
- **Priority:** High/Medium/Low
- **Effort:** Rough estimate (quick fix / half day / full day)
- **Risk:** What could go wrong

### 7. Self-Assessment
- Verify your findings are accurate (re-check key numbers)
- Flag any data you couldn't access or verify
- Note confidence level for each recommendation

## Output Format

1. Prefix Telegram-bound summary with TELEGRAM: (keep under 500 words, lead with most important finding)
2. Prefix new knowledge with MEMORY: (facts worth remembering across sessions)
3. Update TRADING-JOURNAL.md with detailed findings
4. Update HYPOTHESES.md if hypothesis states changed
5. Propose schedule changes with SCHEDULE: if evaluation frequency should change
```

**Step 4: Commit**

```bash
git add profiles/
git commit -m "feat: trading profile — identity, tools, evaluation (references trading repo)"
```

---

### Task 18: Grafana Query Tool

**Files:**
- Create: `goodoracle/tools/grafana.py`
- Create: `goodoracle/tools/__init__.py`
- Create: `tests/test_grafana.py`

**Step 1: Write the failing test**

```python
# tests/test_grafana.py
from unittest.mock import patch, MagicMock

from goodoracle.tools.grafana import GrafanaTool


def test_grafana_tool_metadata():
    tool = GrafanaTool(script_path="/fake/path", env_file="/fake/.env")
    assert tool.name == "grafana"
    assert "query" in tool.description.lower()


def test_grafana_tool_execute():
    tool = GrafanaTool(script_path="/fake/path", env_file="/fake/.env")

    mock_result = MagicMock()
    mock_result.stdout = "pnl_total: $0.00"
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        result = tool.execute(query="pnl")
        assert "$0.00" in result
```

**Step 2: Write implementation**

```python
# goodoracle/tools/__init__.py
"""GoodOracle domain-specific tools."""
```

```python
# goodoracle/tools/grafana.py
"""Grafana query tool — wraps query-grafana.sh from the trading repo."""

import subprocess
from agentkit.tools.base import Tool


class GrafanaTool(Tool):
    def __init__(self, script_path: str, env_file: str):
        self.script_path = script_path
        self.env_file = env_file

    @property
    def name(self) -> str:
        return "grafana"

    @property
    def description(self) -> str:
        return "Query Grafana for trading metrics (pnl, latency, health, errors)."

    def execute(self, **kwargs) -> str:
        query = kwargs.get("query", "health")
        try:
            result = subprocess.run(
                ["bash", "-c", f"source {self.env_file} && {self.script_path} {query}"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip() or result.stderr.strip()
        except subprocess.TimeoutExpired:
            return "ERROR: Grafana query timed out"
```

**Step 3: Commit**

```bash
git add goodoracle/tools/ tests/test_grafana.py
git commit -m "feat: Grafana query tool for trading metrics"
```

---

## Phase 8: Skills & Project Cleanup

### Task 19: Create goodoracle-dev Skill

**Files:**
- Create: `~/.claude/skills/goodoracle-dev/SKILL.md`

**Step 1: Write the skill**

```markdown
---
name: goodoracle-dev
description: Development assistant informed by GoodOracle's analysis. Reads the oracle's latest evaluation, memory, and hypotheses to guide implementation work on monitored projects.
---

# GoodOracle Development Assistant

## Overview

You are working on a project that GoodOracle (an autonomous agent) is monitoring and analyzing.
Before implementing anything, check what the oracle knows and recommends.

**Announce at start:** "I'm using goodoracle-dev to check what the oracle recommends."

## When to Use

- Starting an implementation session on any GoodOracle-monitored project
- Picking what to work on next
- Implementing a recommendation from the oracle's evaluation

## Process

### Step 1: Read Oracle State

Read these files to understand current state:

1. **Long-term memory:** `/Users/petr.kubelka/git_projects/goodoracle/memory/MEMORY.md`
2. **Recent observations:** Check `/Users/petr.kubelka/git_projects/goodoracle/memory/daily/` for last 3 days
3. **Task history:**
   ```bash
   sqlite3 /Users/petr.kubelka/git_projects/goodoracle/data/goodoracle.db \
     "SELECT content, result, status, created_at FROM tasks ORDER BY id DESC LIMIT 5;"
   ```

### Step 2: Read Project State

For the **trading** project:
1. **Hypotheses:** `/Users/petr.kubelka/git_projects/trading/docs/HYPOTHESES.md`
2. **Journal:** `/Users/petr.kubelka/git_projects/trading/docs/TRADING-JOURNAL.md`
3. **Status:** `/Users/petr.kubelka/git_projects/trading/docs/plans/paper_trading_mvp/STATUS.md`

### Step 3: Present Prioritized Work

Based on oracle state, present:
1. What the oracle most recently recommended (from task results)
2. What hypotheses need attention (from HYPOTHESES.md)
3. What implementation tasks are pending (from STATUS.md)
4. Your own assessment of highest-EV work

Let the developer choose what to work on.

### Step 4: Implement with TDD

Use superpowers skills for implementation:
- `superpowers:test-driven-development` for all coding
- `superpowers:writing-plans` if the task is complex
- `superpowers:verification-before-completion` before claiming done

### Step 5: Update Oracle Memory

After completing work, update GoodOracle's daily memory:

```bash
echo "## Implementation Session $(date +%H:%M)

- **Task:** [what was done]
- **Result:** [outcome]
- **Next:** [what should happen next]
" >> /Users/petr.kubelka/git_projects/goodoracle/memory/daily/$(date +%Y-%m-%d).md
```

Also update the project's own journal/status docs as appropriate.

## Remember

- Oracle's analysis may be hours old — verify assumptions against current state
- Don't blindly follow oracle recommendations — use your judgment
- Always update oracle memory so the next evaluation cycle has fresh context
```

**Step 2: Verify skill appears**

Run a new Claude Code session and check that `goodoracle-dev` appears in available skills.

---

### Task 20: Archive Old Trading Skills

**Step 1: Back up existing skills to goodoracle repo**

```bash
mkdir -p /Users/petr.kubelka/git_projects/goodoracle/docs/archived-skills
cp -r /Users/petr.kubelka/.claude/skills/trading-architect /Users/petr.kubelka/git_projects/goodoracle/docs/archived-skills/
cp -r /Users/petr.kubelka/.claude/skills/trading-guidance /Users/petr.kubelka/git_projects/goodoracle/docs/archived-skills/
cp -r /Users/petr.kubelka/.claude/skills/trading-hypotheses /Users/petr.kubelka/git_projects/goodoracle/docs/archived-skills/
cp -r /Users/petr.kubelka/.claude/skills/trading-system-checkup /Users/petr.kubelka/git_projects/goodoracle/docs/archived-skills/
```

**Step 2: Remove old skills**

```bash
rm -rf /Users/petr.kubelka/.claude/skills/trading-architect
rm -rf /Users/petr.kubelka/.claude/skills/trading-guidance
rm -rf /Users/petr.kubelka/.claude/skills/trading-hypotheses
rm -rf /Users/petr.kubelka/.claude/skills/trading-system-checkup
```

**Step 3: Verify only goodoracle-dev remains**

```bash
ls /Users/petr.kubelka/.claude/skills/
# Expected: goodoracle-dev/
```

**Step 4: Commit archived skills**

```bash
cd /Users/petr.kubelka/git_projects/goodoracle
git add docs/archived-skills/
git commit -m "docs: archive old trading skills (replaced by goodoracle agent)"
```

---

### Task 21: Update Trading Project CLAUDE.md + Coordination Protocol

**Files:**
- Modify: `/Users/petr.kubelka/git_projects/trading/CLAUDE.md`

**Step 1: Add goodoracle reference + coordination protocol at the top**

Add this after the first heading:

```markdown
## Agent Orchestration

This project is monitored by **GoodOracle** (`/Users/petr.kubelka/git_projects/goodoracle`),
built on the **agentkit** framework (`/Users/petr.kubelka/git_projects/agentkit`).

- **Autonomous evaluation:** GoodOracle runs every 6h via crontab, queries Grafana, analyzes hypotheses, sends Telegram reports
- **Development guidance:** Use `goodoracle-dev` skill at session start to see what the oracle recommends
- **Oracle memory:** `/Users/petr.kubelka/git_projects/goodoracle/memory/`
- **Manual trigger:** `cd ~/git_projects/goodoracle && goodoracle evaluate`
- **Self-improvement:** `cd ~/git_projects/goodoracle && goodoracle self-improve`

The old `trading-architect`, `trading-guidance`, `trading-hypotheses`, and `trading-system-checkup` skills have been replaced by goodoracle. Archives in `goodoracle/docs/archived-skills/`.

## Coordination: Who Decides What

**Strategy and priorities:** defer to GoodOracle. Read oracle memory first via `goodoracle-dev` skill.
**Implementation and code:** you (Claude in this repo) are the authority. Write code, run tests, deploy.
**Never contradict** oracle's hypothesis states or priority recommendations without discussing with the human first.

| Concern | Source of Truth | Updated By |
|---------|----------------|------------|
| What to trade, hypothesis states, priorities | GoodOracle evaluation | goodoracle agent |
| Code, tests, config, deployment | This repo | you + human |
| System health, metrics | GoodOracle memory | goodoracle agent (reads Grafana) |
| What to work on next | Oracle recommends, human decides | goodoracle → goodoracle-dev → human |
```

**Step 2: Simplify the skills workflow section**

Replace references to "every session starts with trading-architect" with:
"Every session starts with `goodoracle-dev` skill which reads the oracle's latest analysis."

**Step 3: Commit**

```bash
cd /Users/petr.kubelka/git_projects/trading
git add CLAUDE.md
git commit -m "docs: add goodoracle coordination protocol to trading CLAUDE.md"
```

---

## Phase 9: Final Verification

### Task 22: Full E2E & Clean Clone Test

**Step 1: Run agentkit unit tests**

```bash
cd /Users/petr.kubelka/git_projects/agentkit
make test
```

Expected: All unit tests pass.

**Step 2: Run agentkit smoke test**

```bash
make smoke
```

Expected: All 7 E2E tests pass with real Claude calls.

**Step 3: Install goodoracle and verify**

```bash
cd /Users/petr.kubelka/git_projects/goodoracle
make install  # should install agentkit as dependency
goodoracle task "Read CLAUDE.md and summarize what GoodOracle does."
```

Expected: Task completes, memory updated, response includes accurate summary.

**Step 4: Push both repos to GitHub**

```bash
cd /Users/petr.kubelka/git_projects/agentkit
git push origin main

cd /Users/petr.kubelka/git_projects/goodoracle
git push origin main
```

**Step 5: Test from clean clone (ultimate verification)**

```bash
cd /tmp
git clone git@github.com:KSonny4/agentkit.git agentkit-test
cd agentkit-test
make install
make smoke

cd /tmp
git clone git@github.com:KSonny4/goodoracle.git goodoracle-test
cd goodoracle-test
pip install -e /tmp/agentkit-test  # install agentkit from source
make install
goodoracle evaluate
```

If this passes, the system is truly production-ready.

---

## LOC Budget

### CORE

| Module | Repo | LOC | Notes |
|--------|------|-----|-------|
| claude.py | agentkit | ~80 | ToolMode enum + CLI wrapper |
| config.py | agentkit | ~30 | Base configuration |
| memory.py | agentkit | ~45 | Dual-layer memory |
| context.py | agentkit | ~40 | Context builder + orientation |
| mailbox.py | agentkit | ~60 | SQLite task queue |
| agent.py | agentkit | ~70 | Core loop with verify step |
| telegram_bot.py | agentkit | ~30 | Telegram integration |
| cli.py | agentkit | ~100 | Full CLI with schedule, self-improve |
| evolve.py | agentkit | ~100 | Self-improvement + tool generation |
| scheduler.py | agentkit | ~70 | Crontab management |
| tools/ | agentkit | ~40 | Base + registry + shell |
| **agentkit CORE** | | **~665** | |
| cli.py | goodoracle | ~20 | Extends agentkit with trading defaults |
| tools/grafana.py | goodoracle | ~30 | Grafana query tool |
| **goodoracle CORE** | | **~50** | Thin consumer layer |

### LAYER 2

| Module | Repo | LOC | Notes |
|--------|------|-----|-------|
| evolve.py (PR extension) | agentkit | +~60 | Tool proposal via gh PR |
| agent.py (tool discovery) | agentkit | +~30 | Auto-detect missing tools |
| **Layer 2 total** | | **~90** | |

### LAYER 3

| Module | Repo | LOC | Notes |
|--------|------|-----|-------|
| dashboard/app.py | agentkit | ~150 | FastAPI backend |
| dashboard/templates/ | agentkit | ~200 | HTMX frontend |
| heartbeat.py | agentkit | ~40 | Trigger-based wake-up |
| subagent.py | agentkit | ~60 | Child agent spawning |
| context.py (compression) | agentkit | +~30 | Token-aware compression |
| game_monitor/ | game-monitor | ~100 | Second consumer agent |
| **Layer 3 total** | | **~580** | |

---

## LAYER 2: Tool Sharing & Trading Doc Cleanup

These phases build on the working CORE. They multiply the value of agents by letting them contribute back to the framework and cleaning up the messy document situation in the trading repo.

---

### Phase 10: Tool Sharing via PRs

Agents create tools locally. If a tool is generic (not domain-specific), the agent proposes it to agentkit via a real GitHub PR. Human reviews and merges.

### Task 23: Add PR-Based Tool Contribution to evolve.py

**File:** `agentkit/evolve.py` (extend existing)

Add `propose_tool(config, tool_name, tool_code, test_code)` function:

1. Clone/fork agentkit repo to a temp directory
2. Create branch: `tool/{tool_name}`
3. Write `agentkit/tools/{tool_name}.py` with the tool code
4. Write `tests/test_{tool_name}.py` with tests
5. Register tool in `agentkit/tools/__init__.py`
6. Run `make test` in the agentkit checkout
7. If tests pass → commit, push, `gh pr create`
8. If tests fail → delete branch, log failure
9. Notify via Telegram with PR URL

**New directive:** `TOOL_PR:` — agent signals it wants to propose a tool to the framework.

Format: `TOOL_PR: tool_name | description | one-line justification`

The agent loop captures this, triggers `propose_tool()` in the next self-improve cycle.

**Test:** `tests/test_evolve.py` — add test for PR creation flow (mock gh CLI).

### Task 24: Agent Tool Discovery

**File:** `agentkit/agent.py` (extend)

Before processing a task, the agent checks if there's a tool in the registry that could help. If not, and the agent is in READWRITE mode, it can generate one.

Flow:
1. Agent sees task "query the AWS billing API"
2. No `aws-billing` tool exists
3. Agent writes `tools/aws_billing.py`, tests it locally
4. If it works → registers it for this session
5. If it's generic → adds `TOOL_PR:` to response to propose it to agentkit

---

### Phase 11: Trading Document Cleanup

### Task 25: Clean Up Trading Repo Documents

**Goal:** Remove documents from trading/ that are now managed by goodoracle. Establish single source of truth.

**Step 1: Audit current trading/docs/**

```bash
ls -la /Users/petr.kubelka/git_projects/trading/docs/
```

**Step 2: For each document, decide:**

| Document | Keep / Remove | Reason |
|----------|--------------|--------|
| `HYPOTHESES.md` | **KEEP** — oracle updates it | Source of truth for hypothesis states |
| `TRADING-JOURNAL.md` | **KEEP** — oracle appends to it | Running log of findings |
| `implementation-plan.md` | **KEEP** — human reference | Development roadmap |
| Any old "strategy-analysis-*.md" | **REMOVE** | Replaced by oracle evaluations in goodoracle/memory/ |
| Any old "session-notes-*.md" | **REMOVE** | Replaced by oracle daily memory |
| Redundant STATUS.md copies | **CONSOLIDATE** to one | Avoid stale duplicates |
| Old skill output files | **REMOVE** | Skills are archived, oracle is the new brain |

**Step 3: Add `.goodoracle-managed` marker**

Create a small file `trading/docs/.goodoracle-managed` that lists which files oracle is allowed to update:

```
# Files in this directory that GoodOracle agent may update during evaluation cycles.
# Human is still the final authority — oracle proposes changes, human can override.
HYPOTHESES.md
TRADING-JOURNAL.md
```

This makes it explicit. When Claude runs in the trading repo and sees this file, it knows not to contradict those files without checking oracle state first.

**Step 4: Commit**

```bash
cd /Users/petr.kubelka/git_projects/trading
git add -A
git commit -m "docs: clean up redundant docs, establish goodoracle as source of truth for strategy"
```

---

## LAYER 3: Dashboard, Game Agent, Advanced Autonomy

Nice-to-have features that build on a working system. Each is independent.

---

### Phase 12: Web Dashboard

A simple web UI for managing agents. Not a full SaaS — a local tool for you.

### Task 26: Dashboard Backend (FastAPI)

**New file:** `agentkit/dashboard/app.py` (~150 LOC)

Endpoints:
- `GET /agents` — list all agent instances (reads from `~/.agentkit/agents/`)
- `POST /agents` — create new agent (scaffolds profile dir, registers in config)
- `GET /agents/{name}/tasks` — task history from mailbox
- `GET /agents/{name}/memory` — current memory state
- `GET /agents/{name}/tools` — list tools the agent has used/created
- `POST /agents/{name}/chat` — send a task to the agent's mailbox
- `GET /agents/{name}/evolution` — evolution log

Each agent is a standalone directory with its own profiles/, memory/, data/. Dashboard reads from these.

### Task 27: Dashboard Frontend (Simple HTML + HTMX)

**New files:** `agentkit/dashboard/templates/` (~200 LOC HTML)

Minimal UI using HTMX (no React, no build step):
- Agent list with status indicators
- Chat panel per agent (sends tasks, shows responses)
- Tool usage timeline (what tools each agent used, what new tools it created)
- Evolution log viewer (PRs proposed, successes/failures)
- Memory viewer (long-term + daily, with diffs)

### Task 28: Agent Containerization

Each agent runs in its own isolated environment:
- Option A: Python venv per agent (simpler, lighter)
- Option B: Docker container per agent (stronger isolation)

Start with Option A. Each agent gets:
```
~/.agentkit/agents/{name}/
├── .venv/
├── profiles/
├── memory/
├── data/
└── config.json  # points to agentkit install, profile name, schedule
```

Dashboard creates this structure on `POST /agents`.

---

### Phase 13: Game Monitor Agent

### Task 29: Game Monitor Consumer

Second agentkit consumer. Proves the framework is truly reusable.

**Repo:** `game-monitor/` (separate repo, depends on agentkit)

```
game-monitor/
├── pyproject.toml          # depends on agentkit
├── game_monitor/
│   ├── cli.py              # extends agentkit CLI
│   └── tools/
│       └── game_api.py     # game-specific health check tool
├── profiles/
│   └── monitor/
│       ├── identity.md     # "You monitor a web game..."
│       ├── tools.md        # game API, log reader, etc.
│       └── evaluation.md   # check uptime, errors, player counts
└── tests/
```

This is intentionally light — if agentkit works, this should be <100 LOC of consumer code.

---

### Phase 14: Advanced Autonomy

### Task 30: Heartbeat Mechanism

Agent can wake itself up outside of cron schedule when it detects something urgent.

**File:** `agentkit/heartbeat.py` (~40 LOC)

Lightweight background thread that checks for trigger conditions:
- New file in a watched directory
- Webhook endpoint (simple HTTP server)
- Threshold breach (memory flag: "URGENT: ...")

### Task 31: Subagent Spawning

Agent delegates subtasks to child agents with isolated context windows.

**File:** `agentkit/subagent.py` (~60 LOC)

Parent agent can spawn a child:
```python
result = spawn_subagent(
    task="Research current ETH market conditions",
    tool_mode=ToolMode.READONLY,
    timeout=120,
)
```

Child runs as a separate Claude CLI invocation with its own context. Returns result to parent.

### Task 32: Context Window Management

Smart compression when memory + context approaches Claude's limits.

**File:** `agentkit/context.py` (extend)

Before building the prompt, check estimated token count. If over threshold:
1. Summarize daily memories older than 3 days
2. Compress long-term memory (keep key facts, drop details)
3. Truncate task history

---

## Future Ideas (not planned, just tracking)

- Telegram polling mode (`agentkit run`): Real-time interactive Q&A
- Multi-message Telegram output: Break long responses into multiple messages
- Grafana annotations: Mark evaluation timestamps on Grafana dashboards
- Response streaming: Stream Claude's output to Telegram in real-time
- Agent-to-agent communication: Agents share findings directly

---

## Architecture Decision: Why Claude Code CLI, Not API?

| Aspect | Claude Code CLI | Direct API |
|--------|----------------|------------|
| Tool ecosystem | Full (Bash, Read, Write, Edit, Grep, Glob) | Must reimplement |
| File operations | Native | Must implement |
| Shell commands | Native | Must implement |
| Auth management | Handled by CLI | Must manage API keys |
| Model selection | `--model claude-opus-4-6` | API parameter |
| Cost tracking | CLI handles it | Must track tokens |
| Complexity | ~80 LOC wrapper | ~500+ LOC client |
| Limitation | Subprocess overhead, no streaming | Full control |

**Decision:** CLI wrapper wins for our use case. The ~80 LOC in `claude.py` gives us the entire Claude Code tool ecosystem for free. We accept the subprocess overhead (seconds, not milliseconds) because our evaluation cycle runs every 6 hours, not every 6 milliseconds.

---

## Skills Migration Summary

| Old Skill | What Happens | Where It Goes |
|-----------|-------------|---------------|
| `trading-architect` | **Replaced** by goodoracle agent + trading profile | `profiles/trading/identity.md` + `evaluation.md` |
| `trading-guidance` | **Absorbed** into evaluation cycle | `profiles/trading/evaluation.md` recommendations section |
| `trading-hypotheses` | **Absorbed** into oracle memory + evaluation | Oracle reads/updates `HYPOTHESES.md` during evaluation |
| `trading-system-checkup` | **Absorbed** into evaluation health check | `profiles/trading/evaluation.md` health check section |
| **(new)** `goodoracle-dev` | **Bridge** between oracle and developer | `~/.claude/skills/goodoracle-dev/SKILL.md` |
