---
description: Add a new TODO item
argument-hint: ""
allowed-tools:
  - Read
  - Edit
---

Add a new TODO item to the TODO list with interactive form.

## Usage

```
/dev:new-todo
```

## Instructions

1. Read `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/todo/TODO.md`

2. Calculate next TODO ID:
   - Find all TODO items matching `TODO-(\d+)` pattern
   - Get maximum number
   - Increment by 1 to get next ID (e.g., if max is TODO-007, next is TODO-008)

3. Prompt user for TODO details (interactive form):
   - "TODO title/description:" (single line, required)
   - "Priority (P0/P1/P2/P3):" (single character, required, validate)
   - "Estimated effort (S/M/L):" (single character, required)
     - S = Small (<1h)
     - M = Medium (1-4h)
     - L = Large (4h+)
   - "Time estimate:" (e.g., "2-4h", "<1h", "4h+")
   - "Reason/justification:" (multi-line, optional)
   - "Expected benefits:" (multi-line, optional)

4. Validate inputs:
   - Priority must be P0, P1, P2, or P3
   - Effort must be S, M, or L
   - Title must not be empty

5. Find target priority section in TODO.md:
   - Locate `### P# - ` header matching user's priority choice
   - Note section label

6. Format new TODO item:

```markdown
- [ ] **TODO-###**: <title>
  - Estimate: <effort> (<time>)
  - Reason: <reason or "TBD">
  - Created: YYYY-MM-DD
  - Benefits: <benefits or "TBD">
```

7. Insert into appropriate priority section:
   - Add at END of that priority section's item list
   - Before next section boundary (### or end of section)
   - Maintain proper indentation (2 spaces for sub-items)

8. Update statistics:
   - Find `## Active TODOs (#)` header
   - Recalculate count by counting all `- [ ]` items
   - Update number in header

9. Save changes to TODO.md

10. Display confirmation:

```
✅ TODO-### created

Priority: P# - <section label>
Title: <todo title>
Estimate: <effort> (<time>)
Created: YYYY-MM-DD

Added to TODO list under P# section.
```

## Error Handling

- If TODO.md not found: Display error with path
- If priority invalid: Display "Error: Priority must be P0, P1, P2, or P3"
- If effort invalid: Display "Error: Effort must be S (Small), M (Medium), or L (Large)"
- If title empty: Display "Error: TODO title cannot be empty"

## Example

```bash
$ /dev:new-todo

TODO title/description:
> Add Docker Compose setup for PostGIS

Priority (P0/P1/P2/P3):
> P2

Estimated effort (S/M/L):
> M

Time estimate:
> 2-3h

Reason/justification:
> Simplify development environment setup
> Make PostGIS testing easier for new contributors

Expected benefits:
> One-command PostGIS setup
> Reproducible environment
> Easier CI/CD integration

✅ TODO-008 created

Priority: P2 - Medium (This Month)
Title: Add Docker Compose setup for PostGIS
Estimate: M (2-3h)
Created: 2026-01-02

Added to TODO list under P2 section.
```