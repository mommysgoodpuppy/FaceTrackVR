# Project instructions

See `/home/ellie/docs` for extra machine-specific information.

Use `docs/` only for user-facing documentation, `markdown/` for plans and agent Markdown dumps, `markdown/issues/` for project-local issue management by agents (see its README), and `research/` for relevant reference projects and files. Keep `research/` out of commits on `experimental`; the `next-ultra-research` branch tracks its own `research/next_ultra/` prototype.

## Testing

Run tests from the repository root with the app directory on the path, as CI does:

```
PYTHONPATH=EyeTrackApp .venv/bin/python -m pytest -q tests
```

## Git management

- Inspect Git state before editing and preserve existing local changes and untracked files, including other worktrees such as `../FaceTrackVR-resonite`.
- Day-to-day work happens on `experimental`. A tag is also named `experimental`, so use `refs/heads/experimental` where Git reports the name as ambiguous.
- `origin` is this fork; `upstream` is the original EyeTrackVR project. Keep changes suitable for upstreaming where possible.
- Do not commit without asking. An explicit user instruction to commit a particular batch authorizes that batch.
- Keep commits focused, with descriptive messages explaining the change.
- Use descriptive branch names based on the work, not AI or model names.
- Commit as the configured Git identity. No `Co-authored-by` trailers or other co-author attribution.
- End every agent-made commit message with:

  ```
  Agent: <harness> <version>; model <exact model id>; thread <harness thread/session id>
  ```

  For example, `Agent: Claude Code 2.1.306; model claude-opus-5-5; thread 30097453-3823-4ba0-84b3-01b52fd4962e`. Use actual metadata; do not invent unavailable version, model, or session identifiers.

- Do not push, publish, tag releases, or create content on GitHub or other external sites without explicit user authorization. Discussing an issue (for example, "let's make a GitHub issue") is not authorization to immediately create it through the CLI.
