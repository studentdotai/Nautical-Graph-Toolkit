---
description: Complete a task and archive it
argument-hint: "TASK-###"
allowed-tools:
  - Read
  - Edit
  - Write
  - Bash
---

Mark a task as complete, archive it, and update related files.

## Usage

```
/dev:complete-task TASK-###
```

## Instructions

1. Parse argument to extract TASK ID:
   - Match pattern `TASK-(\d+)`
   - Validate format, display error if invalid

2. Find the task file:
   - Use Bash: `ls /home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/tasks/active/TASK-###-*.md`
   - If not found: Display error "TASK-### not found in active tasks"

3. Read the task file:
   - Extract metadata: Title, Priority, Created date, Estimated Effort, Related TODO
   - Extract description and current progress notes

4. Prompt user for completion summary (interactive):
   - "Describe what was accomplished:" (multi-line text)
   - "Actual effort spent:" (e.g., "4h", "2 days")
   - "Key outcomes:" (multi-line text, optional)
   - "Lessons learned:" (multi-line text, optional)

5. Update task file with completion metadata:
   - Update Status: Completed
   - Add Completed: YYYY-MM-DD
   - Add Actual Effort: <user input>
   - Update Progress: 100%
   - Add new `## Completion Summary` section at end:

```markdown
## Completion Summary

**Completed**: YYYY-MM-DD

**Work Accomplished**:
<user input from "describe what was accomplished">

**Actual Effort**: <user input>

**Key Outcomes**:
<user input from "key outcomes", or "None" if empty>

**Lessons Learned**:
<user input from "lessons learned", or "None" if empty>
```

6. Move task file to completed/ directory:
   - Use Bash: `mv /path/to/active/TASK-###-*.md /path/to/completed/`

7. Update TASK_INDEX.md:
   - Move task from "Active Tasks" section to "Recently Completed" table
   - Update statistics: Decrement Active count, increment Completed count
   - In "Recently Completed" table, add row at TOP (newest first):
     `| TASK-### | <title> | P# | YYYY-MM-DD | <duration> |`
   - Calculate duration: Completed date minus Created date

8. Mark related TODO as complete (if applicable):
   - If task has "Related TODO: TODO-###", run `/dev:complete-todo TODO-###`
   - Or update TODO.md directly using same logic as complete-todo command

9. Add entry to DAILY_LOG.md:
   - Check if today's entry exists
   - If not, create new entry for today
   - Under "Work completed" section, add:
     `- Completed TASK-###: <title> (<actual effort>)`

10. Display confirmation:

```
✅ TASK-### completed and archived

Title: <task title>
Completed: YYYY-MM-DD
Duration: <days/weeks> (<created> to <completed>)
Actual effort: <user input>

Updated files:
  - Moved task file to dev/tasks/completed/
  - Updated TASK_INDEX.md statistics
  - Marked TODO-### as complete (if applicable)
  - Added entry to DAILY_LOG.md

Task summary added to completed file.
```

## Error Handling

- If TASK ID format invalid: Display "Error: Invalid TASK ID format. Usage: /dev:complete-task TASK-###"
- If task file not found: Display "Error: TASK-### not found in active tasks"
- If TASK_INDEX.md not found: Display error with path
- If task already in completed/: Display "Error: TASK-### is already completed"

## Example

```bash
$ /dev:complete-task TASK-002

Describe what was accomplished:
> Implemented type hints for all functions in core/graph.py
> Updated tests to verify type correctness
> Added mypy to pre-commit hooks

Actual effort spent:
> 3h

Key outcomes:
> Better IDE support with autocomplete
> Caught 5 type errors during implementation
> Improved code documentation

Lessons learned:
> Type hints help catch bugs early
> Gradual typing approach works well

✅ TASK-002 completed and archived

Title: Add comprehensive type hints to core/graph.py
Completed: 2026-01-02
Duration: 1 day (2026-01-01 to 2026-01-02)
Actual effort: 3h

Updated files:
  - Moved task file to dev/tasks/completed/
  - Updated TASK_INDEX.md statistics (Active: 0, Completed: 2)
  - Marked TODO-002 as complete
  - Added entry to DAILY_LOG.md (2026-01-02)

Task summary added to completed file.
```