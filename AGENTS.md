# Codex Project Rules

This project uses the global Codex operating layer:

- `/Users/jasonnickel/.codex/AGENTS.md`
- `/Users/jasonnickel/.codex/agents/`
- `/Users/jasonnickel/.codex/skills/`

## Project Notes

Python multi-agent trading research framework. Treat trading, portfolio, broker, and execution-adjacent behavior as high-risk. Keep decisions framed as research or decision support unless the user explicitly approves otherwise.

## Codex Workflow

Before editing, classify the task as `ROUTINE`, `COMPLEX`, `ARCHITECTURAL`, or `SECURITY-SENSITIVE`.

Use `codex-scout` for read-only mapping and test discovery. Use `codex-routine-executor` for bounded 1-2 file routine edits. Use `codex-risk-reviewer` for architecture, LLM/provider behavior, trading/execution boundaries, public CLI contracts, security, or data-integrity risk. Use `codex-git-workflow` for git status, diff summaries, commit grouping, and PR text.

Keep final decisions, architecture, execution/broker boundaries, LLM/provider routing, and release-risk calls in the main high-reasoning Codex session.

Git/GitHub safety:
- Use `codex-git-workflow` for git status, git diff review, branch hygiene, commit grouping, commit message drafting, PR summary drafting, and merge-readiness checks.
- Never merge, push, force-push, delete branches, or rebase shared branches without explicit user approval.
- Escalate to high-reasoning Codex before merge/release when schema, migrations, security, auth, data integrity, public contracts, architecture, forensic/evidence files, broad refactors, failing tests, or unclear release readiness are involved.

## Verified Commands

Verified from repo files:

```bash
pytest
```

Do not invent verification commands. Do not edit secrets/provider config or change execution boundaries without explicit user approval.
