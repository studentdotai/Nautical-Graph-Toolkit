# Task Index

Master list of all tasks - Active and completed.

**Purpose**: Central registry for tracking all development tasks. Provides overview of active work, completed achievements, and upcoming tasks.

Last Updated: YYYY-MM-DD

## Statistics

- **Total Tasks**: 0
- **Active**: 0
- **Blocked**: 0
- **Completed**: 0
- **Cancelled**: 0

## Active Tasks (0)

[No active tasks currently]

## Blocked Tasks (0)

[No blocked tasks currently]

## Recently Completed (0)

[No completed tasks yet]

---

## Example Index (After Some Work)

Last Updated: 2026-01-20

### Statistics

- **Total Tasks**: 5
- **Active**: 1
- **Blocked**: 0
- **Completed**: 4
- **Cancelled**: 0

### Active Tasks (1)

- TASK-005: Implement user authentication system (Priority: P1, Created: 2026-01-18)

### Blocked Tasks (0)

None currently

### Recently Completed (4)

| ID | Title | Priority | Completed | Duration |
|----|-------|----------|-----------|----------|
| TASK-004 | Add API documentation | P2 | 2026-01-17 | ~3h |
| TASK-003 | Optimize database queries | P1 | 2026-01-15 | ~5h |
| TASK-002 | Fix critical bug in auth module | P0 | 2026-01-12 | ~2h |
| TASK-001 | Setup development environment | P0 | 2026-01-10 | ~4h |

---

## Task Details

### TASK-005: Implement user authentication system ⏳

- **File**: `/dev/tasks/active/TASK-005-implement-user-authentication.md`
- **Priority**: P1 (High)
- **Status**: Active
- **Created**: 2026-01-18
- **Completed**: (in progress)
- **Owner**: [Your name or team name]
- **Estimated Effort**: L (6-8h)
- **Actual Effort**: ~3h so far
- **Dependencies**: TASK-004 (API docs completed)
- **Related**: TODO-015

**Summary**: Implement JWT-based authentication with role-based access control and session management.

**Progress**: 40% complete - Basic JWT signing/verification done, working on RBAC implementation.

### TASK-004: Add API documentation ✅

- **File**: `/dev/tasks/completed/TASK-004-add-api-documentation.md`
- **Priority**: P2 (Medium)
- **Status**: Completed
- **Created**: 2026-01-14
- **Completed**: 2026-01-17
- **Owner**: Development Team
- **Estimated Effort**: M (3-4h)
- **Actual Effort**: ~3h
- **Dependencies**: None
- **Related**: TODO-012

**Summary**: Created comprehensive OpenAPI documentation for all REST endpoints with examples.

**Results**:
- ✅ OpenAPI spec generated (45 endpoints documented)
- ✅ Interactive Swagger UI deployed
- ✅ Example requests/responses added
- ✅ Authentication flow documented

---

## Upcoming Tasks (Next 3)

Based on current priorities:

1. **TASK-006**: Implement password reset functionality (from TODO-016)
   - Estimated: M (2-3h)
   - Priority: P1
   - Owner: TBD
   - Depends on: TASK-005 (auth system)

2. **TASK-007**: Add email verification (from TODO-017)
   - Estimated: M (3-4h)
   - Priority: P2
   - Owner: TBD
   - Depends on: TASK-005 (auth system)

3. **TASK-008**: Optimize API response times (from TODO-019)
   - Estimated: L (5-6h)
   - Priority: P2
   - Owner: TBD
   - Depends on: None

---

## Task Naming Convention

`TASK-[NUMBER]-[descriptive-kebab-case-name].md`

**Examples**:
- `TASK-001-setup-development-environment.md`
- `TASK-042-implement-docker-compose.md`
- `TASK-123-add-time-series-routing.md`

**Guidelines**:
- Use zero-padded 3-digit numbers (001, 002, 042, 123)
- Descriptive kebab-case name (lowercase, hyphens)
- Keep name concise but clear (max 50 characters)
- Focus on the outcome, not the method

---

## Task Lifecycle

```
┌─────────┐
│  TODO   │  (Idea/requirement in TODO.md or BACKLOG.md)
└────┬────┘
     │
     │ Create task file
     ├──────────────────────► [If higher priority or quick win]
     │                        [Otherwise stays in BACKLOG]
     ▼
┌─────────┐
│ Planned │  (Task file created, not started)
└────┬────┘
     │
     │ Start work
     ▼
┌─────────┐
│ Active  │  (Currently working on it)  ◄──┐
└────┬────┘                                 │
     │                                      │
     ├─► ┌─────────┐                       │
     │   │ Blocked │  (Waiting on external)│ Resume work
     │   └────┬────┘                       │
     │        │ Resolve blocker            │
     │        └────────────────────────────┘
     │
     │ Complete work
     ▼
┌───────────┐
│ Completed │  (Done, archived)
└───────────┘
     │
     │ OR
     ▼
┌───────────┐
│ Cancelled │  (No longer relevant)
└───────────┘
```

**Flow Description**:
1. **TODO → Create Task**: Promote high-priority TODO to task
2. **Planned → Active**: Start working (move to active/)
3. **Active → Blocked**: Hit external dependency, mark blocked
4. **Blocked → Active**: Blocker resolved, resume work
5. **Active → Completed**: Finish work, mark complete (move to completed/)
6. **Any → Cancelled**: Task no longer relevant (move to completed/ with cancelled status)

---

## Status Definitions

- **Planned**: Task created, not yet started (still in planning or backlog)
- **Active**: Currently being worked on (limit to 1-2 active tasks per person)
- **Blocked**: Waiting on external dependency or blocker resolution
- **Completed**: Work finished, acceptance criteria met, outcomes documented
- **Cancelled**: No longer relevant, superseded by other work, or deprioritized

---

## Task Management Guidelines

### When to Create a Task

**Create a task when**:
- TODO item becomes high priority (P0 or P1)
- Work will take >2 hours (needs planning and tracking)
- Multiple work sessions required
- Significant impact on codebase or users
- Requires coordination with others
- Needs documentation of decisions and approach

**Don't create a task for**:
- Quick fixes (<30 min)
- Minor typos or formatting
- Routine maintenance
- Low-priority items that can stay in BACKLOG

### Limiting Active Tasks

**Best Practices**:
- Limit to 1-2 active tasks per person
- Focus on completion, not starting
- Finish tasks before starting new ones
- Move stalled work to blocked status
- Re-evaluate blocked tasks weekly

### Updating Task Status

**Update frequency**:
- Daily: Update progress notes in active task files
- At milestones: Update progress percentage
- When blocked: Immediately mark as blocked with reason
- At completion: Document outcomes and lessons learned
- Weekly: Review all active tasks, update TASK_INDEX

### Archiving Completed Tasks

**When to archive**:
- Task is 100% complete
- All acceptance criteria met
- Outcomes documented
- Code merged/deployed

**How to archive**:
1. Update task file with completion date and outcomes
2. Move file from `active/` to `completed/`
3. Update TASK_INDEX.md statistics
4. Add entry to "Recently Completed" table
5. Update related TODO items (mark as completed)

---

## Cross-References

- **TODO List**: `/dev/todo/TODO.md` (active work items, source for new tasks)
- **Priorities**: `/dev/todo/PRIORITIES.md` (priority framework and decision guide)
- **Backlog**: `/dev/todo/BACKLOG.md` (future work, low-priority items)
- **Daily Log**: `/dev/progress/DAILY_LOG.md` (daily progress updates)
- **Changelog**: `/dev/progress/CHANGELOG.md` (notable changes)
- **Milestones**: `/dev/progress/MILESTONES.md` (major achievements)
- **Project Rules**: `/dev/rules/CLAUDE.md` (project knowledge and standards)
