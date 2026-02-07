# agentkit-agents

Consumer agents built on [agentkit](https://github.com/KSonny4/agentkit).

Each agent = agentkit core + domain-specific profiles, tools, and configuration.

## Agents

| Agent | Description | Status |
|-------|-------------|--------|
| `goodoracle` | Trading analysis agent | Planned |
| `game-monitor` | Game monitoring agent | Planned |

## How to Add an Agent

1. Create a directory: `mkdir my-agent`
2. Add a profile: `profiles/<name>/identity.md`, `tools.md`, `evaluation.md`
3. Optionally add custom tools extending `agentkit.tools.base.Tool`
4. Add a README explaining the agent's purpose and setup

## Dependencies

Each agent depends on `agentkit`:

```bash
pip install agentkit
# or add to your pyproject.toml
```

## Running an Agent

```bash
cd my-agent
agentkit task --profile my-profile "your prompt"
agentkit evaluate --profile my-profile
```
