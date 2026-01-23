---
description: Mark TODO item as complete
argument-hint: "TODO-###"
allowed-tools:
  - Read
  - Edit
---

Mark a TODO item as complete and move it to the "Recently Completed" section.

## Usage

```
/dev:complete-todo TODO-###
```

## Instructions

1. Parse argument to extract TODO ID:
   - Match pattern `TODO-(\d+)`
   - Validate format, display error if invalid

2. Read `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/todo/TODO.md`

3. Find the TODO item:
   - Search for `- [ ] **TODO-###**:` in active sections (P0-P3)
   - Extract the full multi-line item (including all indented sub-items)
   - Note the priority section it's in

4. Check if already completed:
   - Search in "Recently Completed" section for `- [x] **TODO-###**:`
   - If found: Display warning "TODO-### is already completed (completed on <date>)" and exit

5. Extract item details:
   - Title, estimate, reason, created date, benefits
   - Preserve all indented content

6. Move item to "Recently Completed" section:
   - Change checkbox from `[ ]` to `[x]`
   - Add completion metadata: `- Completed: YYYY-MM-DD`
   - Find `## Recently Completed (#)` section
   - Insert item at TOP of completed list (newest first)
   - Update count in section header

7. Remove item from active section:
   - Delete the original TODO item (all lines)
   - Clean up empty lines

8. Update statistics in header:
   - Find `## Active TODOs (# items)` header
   - Recalculate count by counting actual `- [ ]` items
   - Update the number in header

9. Save changes to TODO.md

10. Display confirmation:

```
✅ TODO-### marked as complete

Moved from: P# - <priority label>
Completed: YYYY-MM-DD

Title: <todo title>
```

## Error Handling

- If TODO ID format invalid: Display "Error: Invalid TODO ID format. Usage: /dev:complete-todo TODO-###"
- If TODO.md not found: Display error with path
- If TODO not found in active sections: Display "Error: TODO-### not found in active TODO list"
- If TODO already completed: Display warning with completion date

## Example

```bash
$ /dev:complete-todo TODO-002

✅ TODO-002 marked as complete

Moved from: P1 - High (This Week)
Completed: 2026-01-02

Title: Add comprehensive type hints to core/graph.py
```