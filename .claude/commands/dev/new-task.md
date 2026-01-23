---
description: Create new task from TODO item
argument-hint: ""
allowed-tools:
  - Read
  - Edit
  - Write
---

Create a new task file from a TODO item with interactive selection and autopopulation.

## Usage

```
/dev:new-task
```

## Instructions

1. Read `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/todo/TODO.md`

2. Parse and display active TODO items:
   - Extract all items matching `- [ ] **TODO-###**: <title>` from P0-P3 sections
   - Extract estimate for each item
   - Display numbered list grouped by priority:

```
Available TODOs:

P0 - Critical:
  (none)

P1 - High:
  1. TODO-002: Add comprehensive type hints [M, 2-4h]
  2. TODO-003: Create CLI entry point [M, 2-3h]

P2 - Medium:
  3. TODO-004: Pre-commit hooks [S, <1h]
  ...
```

3. Prompt user to select TODO:
   - "Select TODO to create task for (1-N or TODO-###):"
   - Accept either number or TODO-ID format
   - Validate selection

4. Read selected TODO item details:
   - Extract full multi-line item with all sub-content
   - Parse: Title, Estimate, Reason, Benefits, Created date
   - Note: Priority from section

5. Calculate next TASK ID:
   - Read `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/tasks/TASK_INDEX.md`
   - Find all TASK-### IDs
   - Get maximum number and increment (e.g., if max is TASK-001, next is TASK-002)

6. Prompt for additional task details (interactive):
   - "Task owner (or press Enter for 'Claude'):" (default: Claude)
   - "Additional description/context:" (multi-line, optional)
   - "Acceptance criteria:" (multi-line, optional - suggest based on TODO)

7. Create task filename:
   - Format: `TASK-###-<slug>.md`
   - Generate slug from title: lowercase, hyphens, max 50 chars
   - Example: `TASK-002-add-type-hints.md`

8. Create task file content using template:

```markdown
# TASK-###: <title from TODO>

<TODO reason or additional description>

## Metadata

- **ID**: TASK-###
- **Priority**: <priority from TODO section>
- **Status**: Planned
- **Created**: YYYY-MM-DD
- **Completed**: (not yet)
- **Owner**: <user input or Claude>
- **Estimated Effort**: <estimate from TODO (S/M/L with time)>
- **Actual Effort**: (to be determined)
- **Progress**: 0%
- **Related TODO**: TODO-###

## Description

<TODO title and reason>

<additional description from user if provided>

## Acceptance Criteria

<user input or auto-generated from TODO>

Example auto-generated:
- [ ] <main goal from TODO title>
- [ ] Tests pass
- [ ] Documentation updated (if applicable)

## Implementation Plan

*To be filled in during implementation*

## Progress Notes

*Progress updates will be added here*

## Skills Used

*To be documented*

## Outcomes

**Expected**:
- <TODO benefits or goals>

**Actual** (to be updated on completion):
- (to be determined)

## Lessons Learned

*To be filled on completion*

## Related Items

- **TODO**: TODO-###
- **Priority**: <priority>
- **Dependencies**: (if any)

## Cross-References

- **Project Knowledge**: `/dev/rules/CLAUDE.md`
- **Skills**: `.claude/skills/` (13 specialized skills)
- **TODO List**: `/dev/todo/TODO.md`
```

9. Write task file:
   - Save to `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/tasks/active/TASK-###-<slug>.md`

10. Update TASK_INDEX.md:
    - Increment "Active" count in statistics
    - Increment "Total Tasks" count
    - Add entry to "Active Tasks" section:
      - If section says "None", replace with task entry
      - Otherwise add to list
    - Format: `- TASK-###: <title> (Priority: P#, Created: YYYY-MM-DD)`

11. Display confirmation:

```
✅ TASK-### created

Title: <task title>
Priority: <priority>
Estimated effort: <estimate>
Created: YYYY-MM-DD
Related TODO: TODO-###

Task file: dev/tasks/active/TASK-###-<slug>.md
TASK_INDEX.md updated (Active: #, Total: #)

Next steps:
  - Use /dev:update-task to track progress
  - Use /dev:complete-task when done
```

## Error Handling

- If no active TODOs: Display "No active TODOs found. Use /dev:new-todo to create one first."
- If TODO.md not found: Display error with path
- If selection invalid: Display "Error: Invalid selection. Choose 1-N or TODO-###"
- If TASK_INDEX.md not found: Display error with path

## Example

```bash
$ /dev:new-task

Available TODOs:

P1 - High:
  1. TODO-002: Add comprehensive type hints [M, 2-4h]
  2. TODO-003: Create CLI entry point [M, 2-3h]

P2 - Medium:
  3. TODO-004: Pre-commit hooks [S, <1h]

Select TODO to create task for (1-3 or TODO-###):
> 1

Task owner (or press Enter for 'Claude'):
> [Enter]

Additional description/context:
> Focus on core/graph.py module first
> Use mypy for validation
> Add to pre-commit hooks when complete

Acceptance criteria:
> - [ ] All functions in core/graph.py have type hints
> - [ ] All classes have type hints
> - [ ] mypy passes with no errors
> - [ ] Tests updated to verify types
> - [ ] Pre-commit hook configured

✅ TASK-002 created

Title: Add comprehensive type hints to core/graph.py
Priority: P1
Estimated effort: M (2-4h)
Created: 2026-01-02
Related TODO: TODO-002

Task file: dev/tasks/active/TASK-002-add-type-hints.md
TASK_INDEX.md updated (Active: 1, Total: 2)

Next steps:
  - Use /dev:update-task to track progress
  - Use /dev:complete-task when done
```