---
description: Update task status and progress
argument-hint: ""
allowed-tools:
  - Read
  - Edit
---

Update task progress, status, and add progress notes interactively.

## Usage

```
/dev:update-task
```

## Instructions

1. Read `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/tasks/TASK_INDEX.md`

2. List active tasks:
   - Extract from "Active Tasks" section
   - Display numbered list for selection

3. Prompt user to select task:
   - "Select task to update (1-N or TASK-###):"
   - Accept either number or TASK-ID format
   - Validate selection

4. Find and read task file:
   - Use Bash: `ls /home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/tasks/active/TASK-###-*.md`
   - Read current task file content

5. Display current task info:

```
TASK-###: <title>
Status: <current status>
Progress: <current progress %>
Priority: <priority>
```

6. Prompt for updates (interactive, all optional):
   - "New status (or press Enter to keep '<current>'):" (Planned/Active/Blocked)
   - "Progress percentage (0-100, or press Enter to keep <current>):" (number)
   - "Add progress notes:" (multi-line text, optional)
   - "Update current focus (for session tracking):" (single line, optional)

7. Update task file:
   - Update Status field if provided
   - Update Progress field if provided
   - If progress notes provided, add new entry to `## Progress Notes` section:

```markdown
### YYYY-MM-DD

<user progress notes>
```

8. Update session tracking if focus changed:
   - If user provided new focus, update `/home/vikont/PythonProject/Nautical-Graph-Toolkit/.claude/dev.local.md`
   - Update `current_focus:` field in YAML frontmatter

9. Display confirmation:

```
✅ TASK-### updated

Status: <old> → <new> (if changed)
Progress: <old>% → <new>% (if changed)
Current focus: <new focus> (if changed)

Progress notes added: <Yes/No>
```

## Error Handling

- If no active tasks: Display "No active tasks found. Use /dev:new-task to create one."
- If task selection invalid: Display "Error: Invalid selection. Choose 1-N or TASK-###"
- If task file not found: Display error with task ID
- If progress not 0-100: Display "Error: Progress must be between 0 and 100"

## Example

```bash
$ /dev:update-task

Active tasks:
  1. TASK-002: Add type hints to core/graph.py
  2. TASK-003: Create CLI entry point

Select task to update (1-2 or TASK-###):
> 1

TASK-002: Add type hints to core/graph.py
Status: Active
Progress: 40%
Priority: P1

New status (or press Enter to keep 'Active'):
> [Enter]

Progress percentage (0-100, or press Enter to keep 40):
> 60

Add progress notes:
> Completed type hints for BaseGraph class
> Updated tests to verify type correctness
> Next: FineGraph and H3Graph classes

Update current focus (for session tracking):
> Type annotations for graph generation

✅ TASK-002 updated

Status: Active (unchanged)
Progress: 40% → 60%
Current focus: Type annotations for graph generation

Progress notes added: Yes
```