---
description: End development session with summary and cleanup
argument-hint: ""
allowed-tools:
  - Read
  - Write
  - Edit
---

End the current development session with a summary of work completed.

## Instructions

1. Read session tracking file:
   - `/home/vikont/PythonProject/Nautical-Graph-Toolkit/.claude/dev.local.md`
   - Extract `session_start:` timestamp
   - Extract `active_tasks:` list
   - Extract `current_focus:` string

2. Read task files for active tasks (if any):
   - For each task in `active_tasks`, read `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/tasks/active/TASK-###-*.md`
   - Extract task title and current status/progress

3. Calculate session duration:
   - Current time minus `session_start` timestamp
   - Format as hours and minutes (e.g., "2 hours 15 minutes")

4. Display session summary in Style B format:

```
=== Session End: YYYY-MM-DD HH:MM ===

SESSION DURATION: <hours>h <minutes>m

ACTIVE TASKS WORKED ON:
  - TASK-###: <title> (Status: <status>)
  (or "None - no active tasks this session" if empty)

CURRENT FOCUS:
  <focus description>
  (or "None recorded" if not set)

---

Would you like to update DAILY_LOG.md with today's progress?
  [Yes] Update DAILY_LOG now
  [No] Skip for now
```

5. Prompt user for DAILY_LOG update:
   - If user says yes: Show them `/dev:log-progress` command can be used
   - If user says no: Proceed to cleanup

6. Update `.claude/dev.local.md` for session end:
   - Update `last_session:` to `session_start` value (archive this session)
   - Clear `session_start:` (no active session now)
   - Keep `active_tasks` and `current_focus` unchanged
   - Keep `session_count` unchanged

## Error Handling

- If .claude/dev.local.md not found: Display message "No active session found"
- If session_start not found in tracking file: Display message "Session start time not recorded"

## Example Output

```
=== Session End: 2026-01-02 16:45 ===

SESSION DURATION: 4h 30m

ACTIVE TASKS WORKED ON:
  - TASK-002: Add type hints to core/graph.py (Status: In Progress - 60% complete)
  - TASK-003: Create CLI entry point (Status: Completed)

CURRENT FOCUS:
  Implementing type annotations for graph generation functions

---

Would you like to update DAILY_LOG.md with today's progress?
  Use /dev:log-progress to record today's work
  Or skip for now
```