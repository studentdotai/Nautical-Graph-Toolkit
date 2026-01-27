---
description: Change TODO item priority
argument-hint: "TODO-### P#"
allowed-tools:
  - Read
  - Edit
---

Change the priority of a TODO item by moving it to a different priority section.

## Usage

```
/dev:prioritize TODO-### P#
```

Where P# is: P0, P1, P2, or P3

## Instructions

1. Parse arguments:
   - Extract TODO ID matching pattern `TODO-(\d+)`
   - Extract new priority matching pattern `P(\d)` where digit is 0-3
   - Validate both formats, display error if invalid

2. Read `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/todo/TODO.md`

3. Find the TODO item in current priority section:
   - Search all active sections (P0-P3) for `- [ ] **TODO-###**:`
   - Extract full multi-line item
   - Note current priority section

4. Check if already at target priority:
   - If current priority == new priority: Display "TODO-### is already at P#" and exit

5. Find target priority section:
   - Locate `### P# - ` section header
   - Note section label (e.g., "High (This Week)", "Medium (This Month)")

6. Move item to new priority section:
   - Extract complete item from current section (preserve all indentation)
   - Insert at END of target section's item list (before next ### or end of section)
   - Delete from current section

7. Update related task priority (if applicable):
   - Read `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/tasks/TASK_INDEX.md`
   - Search for task with `Related TODO: TODO-###` or matching title
   - If found: Update priority column in task table to match new TODO priority
   - Save TASK_INDEX.md if modified

8. Display confirmation:

```
✅ TODO-### priority changed

From: P# - <old section label>
To: P# - <new section label>

Title: <todo title>
```

Optional note if related task found:
```
📋 Updated related TASK-###  priority to P#
```

## Error Handling

- If argument format invalid: Display "Error: Invalid format. Usage: /dev:prioritize TODO-### P#"
- If priority invalid: Display "Error: Invalid priority. Must be P0, P1, P2, or P3"
- If TODO.md not found: Display error with path
- If TODO not found: Display "Error: TODO-### not found in active TODO list"
- If already at target priority: Display info message and exit

## Example

```bash
$ /dev:prioritize TODO-005 P1

✅ TODO-005 priority changed

From: P2 - Medium (This Month)
To: P1 - High (This Week)

Title: Improve test coverage for utils/s57_classification.py

📋 Updated related TASK-005 priority to P1
```