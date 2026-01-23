---
description: Add entry to daily progress log
argument-hint: ""
allowed-tools:
  - Read
  - Edit
  - Write
---

Add a new entry to DAILY_LOG.md for today's progress.

## Instructions

1. Get current date in YYYY-MM-DD format

2. Read `/home/vikont/PythonProject/Nautical-Graph-Toolkit/dev/progress/DAILY_LOG.md`

3. Check if entry for today already exists:
   - Search for `## YYYY-MM-DD` matching today's date
   - If exists: Inform user and ask if they want to append to existing entry

4. Prompt user for progress information (interactive form):
   - "What did you work on today?" (multi-line text)
   - "Key decisions made:" (multi-line text, optional)
   - "Blockers encountered:" (multi-line text, optional)
   - "Next steps:" (multi-line text, optional)

5. Format the new entry:

```markdown
## YYYY-MM-DD

**Work completed**:
<user input from "what did you work on">

**Decisions**:
<user input from "key decisions", or "None" if empty>

**Blockers**:
<user input from "blockers", or "None" if empty>

**Next steps**:
<user input from "next steps", or "None" if empty>

---
```

6. Insert the entry into DAILY_LOG.md:
   - If file is empty or has only header: Add after header
   - If has existing entries: Add at the TOP (after file header, before first entry)
   - Maintain reverse chronological order (newest first)

7. Confirm to user: "Progress logged for YYYY-MM-DD"

## Error Handling

- If DAILY_LOG.md not found: Display error with path
- If entry already exists: Ask user to confirm append or replace
- If user cancels: Exit without changes

## Entry Format Example

```markdown
## 2026-01-02

**Work completed**:
- Implemented session-start, session-end, and log-progress commands for dev plugin
- Created command file structure in .claude/commands/dev/
- Updated documentation for command interaction patterns

**Decisions**:
- Chose Style B (Text/Structured) output format for session-start
- Decided on mixed interaction pattern (some commands interactive, some argument-based)

**Blockers**:
None

**Next steps**:
- Implement remaining 6 commands (new-task, complete-task, update-task, new-todo, complete-todo, prioritize)
- Test each command individually
- Run workflow integration tests

---
```