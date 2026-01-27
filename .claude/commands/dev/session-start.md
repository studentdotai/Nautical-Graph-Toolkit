---
description: Start development session with TODO and task overview
argument-hint: ""
allowed-tools:
  - Read
  - Write
---

Display session overview with current TODOs and active tasks.

## Instructions

1. Read the following files:
   - `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/todo/TODO.md`
   - `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/tasks/TASK_INDEX.md`
   - `/home/vikont/PythonProject/Nautical-Graph-Toolkit/.claude/dev.local.md` (if exists)

2. Parse TODO.md to extract TODO items by priority section:
   - Find sections matching `### P0 - `, `### P1 - `, `### P2 - `, `### P3 - `
   - Within each section, extract items matching `- [ ] **TODO-###**: <title>`
   - Extract estimates matching `Estimate: <size> (<time>)`

3. Parse TASK_INDEX.md to extract active tasks:
   - Find `## Active Tasks (#)` section
   - Extract task count from header
   - If count > 0 and not "None", extract task IDs and titles

4. Parse .claude/dev.local.md (if exists) for session tracking:
   - Extract `last_session:` timestamp from YAML frontmatter
   - Extract `session_count:` number
   - Extract `current_focus:` string
   - Calculate time since last session (days/hours/minutes ago)

5. Display output in Style B (Text/Structured) format:

```
=== Session Start: YYYY-MM-DD ===

TODO LIST (# items):
  P0 - Critical (# items):
    - TODO-###: <title> [<size>, <time>]
    (or "None" if empty)

  P1 - High (<section label>) (# items):
    - TODO-###: <title> [<size>, <time>]
    ...

  P2 - Medium (<section label>) (# items):
    - TODO-###: <title> [<size>, <time>]
    ...

  P3 - Low (<section label>) (# items):
    - TODO-###: <title> [<size>, <time>]
    ...

ACTIVE TASKS (#):
  - TASK-###: <title>
  (or "None currently - ready for next task!" if empty)

SESSION INFO:
  Last session: <time ago> (<date>)  (or "First session")
  Session count: #
  Current focus: <focus>  (or "None")
```

6. Update `.claude/dev.local.md` with new session start:
   - Increment `session_count` by 1
   - Update `session_start` to current timestamp
   - Keep `last_session` as previous session start (don't overwrite yet)
   - Keep existing `active_tasks` and `current_focus`

## Error Handling

- If TODO.md not found: Display error message with path
- If TASK_INDEX.md not found: Display error message with path
- If .claude/dev.local.md not found: Treat as first session (create new tracking file)

## Example Output

```
=== Session Start: 2026-01-02 ===

TODO LIST (6 items):
  P0 - Critical (0 items):
    None

  P1 - High (This Week) (2 items):
    - TODO-002: Add comprehensive type hints [M, 2-4h]
    - TODO-003: Create CLI entry point [M, 2-3h]

  P2 - Medium (This Month) (2 items):
    - TODO-004: Pre-commit hooks [S, <1h]
    - TODO-005: Improve test coverage [M, 2-3h]

  P3 - Low (When Possible) (2 items):
    - TODO-006: Docker Compose setup [M, 2-3h]
    - TODO-007: Video tutorial [L, 4h+]

ACTIVE TASKS (0):
  None currently - ready for next task!

SESSION INFO:
  Last session: 23 hours ago (2026-01-01)
  Session count: 6
  Current focus: None
```