# TODO List

Active work items and priorities.

**Purpose**: Track active work items that are ready to be worked on. This is your working list of tasks that are higher priority than backlog items.

Last Updated: YYYY-MM-DD

## Active TODOs (0 items)

[No active TODOs currently - start adding your work items below]

---

## Example Active TODOs (After Adding Work Items)

Last Updated: 2026-01-15

### Active TODOs (5 items)

### P0 - Critical (Do Now)

[No P0 items - reserve for truly urgent work that blocks everything else]

### P1 - High (This Week)

- [ ] **TODO-002**: Add comprehensive type hints to core module
  - Estimate: M (2-4h)
  - Reason: Improve IDE support, catch type errors early, better code quality
  - Created: 2026-01-10
  - Benefits: Type safety, IntelliSense, documentation, fewer runtime errors

- [ ] **TODO-005**: Implement user authentication system
  - Estimate: L (6-8h)
  - Reason: Security requirement for multi-tenant deployment
  - Created: 2026-01-12
  - Benefits: Secure API endpoints, enable user management, RBAC
  - **Blocked**: Waiting for API documentation (TASK-004) to complete

### P2 - Medium (This Month)

- [ ] **TODO-008**: Add pre-commit hooks for code quality
  - Estimate: S (<1h)
  - Reason: Enforce code quality automatically before commits
  - Created: 2026-01-13
  - Benefits: Consistent style, catch issues early, reduce review time

- [ ] **TODO-010**: Improve test coverage for utility modules
  - Estimate: M (2-3h)
  - Current: ~60%, Target: >80%
  - Reason: Critical utilities need better coverage
  - Created: 2026-01-14
  - Benefits: Higher confidence in changes, catch edge cases

### P3 - Low (When Possible)

- [ ] **TODO-015**: Refactor database connection pooling
  - Estimate: M (3-4h)
  - Reason: Current implementation has memory leaks under high load
  - Created: 2026-01-15
  - Benefits: Better performance, reduced memory usage, improved stability
  - **Investigation Progress (2026-01-15)**:
    - Root cause: Pool not releasing connections properly
    - Affects: PostGIS backend only
    - Workaround: Manual connection cleanup in tests

---

## Recently Completed (3 items)

- [x] **TODO-001**: Setup development environment
  - Completed: 2026-01-09
  - Result: Created automated setup script
  - Outcome: Reduced onboarding from 2h to 10 minutes

- [x] **TODO-003**: Fix critical authentication bug
  - Completed: 2026-01-11
  - Result: Token validation error resolved
  - Outcome: Zero authentication failures in production

- [x] **TODO-007**: Update documentation for new API
  - Completed: 2026-01-14
  - Result: All endpoints documented with OpenAPI
  - Outcome: API docs complete, Swagger UI deployed

---

## Guidelines

### Managing Your TODO List

**Keep TODO list <20 items**:
- Move low priority items to BACKLOG.md
- Archive completed items after documenting outcome
- Remove stale items that are no longer relevant

**Estimate sizes**:
- **S (small)**: <1h - Quick fixes, minor updates
- **M (medium)**: 1-4h - Feature additions, moderate refactoring
- **L (large)**: 4-8h - Major features, significant refactoring
- **XL (extra large)**: >8h - Should be broken into smaller TODOs

**Priority levels**:
- **P0 (critical)**: Do now - Blocking other work, security issues, production bugs
- **P1 (high)**: This week - Important features, roadmap commitments, significant improvements
- **P2 (medium)**: This month - Nice-to-have features, code quality improvements
- **P3 (low)**: When possible - Minor enhancements, nice-to-haves, experiments

### Workflow

**Creating TODOs**:
1. Add new TODO with next available number (TODO-XXX)
2. Assign priority based on impact and urgency
3. Estimate effort realistically
4. Write clear reason (why this work matters)
5. List benefits (what improvements result)
6. Update "Active TODOs" count

**Starting Work**:
1. **Create TASK when starting work** on a TODO item (for tracking and planning)
2. Move TODO to "In Progress" in task file
3. Break down into implementation steps in task file
4. Track progress in task file, not TODO

**Completing Work**:
1. Mark TODO as [x] completed
2. Add completion date
3. Note brief outcome
4. Move to "Recently Completed" section
5. Archive after 1-2 weeks (keep recent completions visible)
6. Update related documentation

**Blocking TODOs**:
- Add **Blocked** note with reason
- Document blocker and what's needed to unblock
- Check weekly if blocker is resolved

### Update Frequency

- **Daily**: Mark completed items, add new urgent items
- **Weekly**: Review priorities, reorder based on current focus
- **Monthly**: Archive old completed items, move low-priority to backlog

### Retention Policy

- **Active items**: Keep until completed or moved to backlog
- **Recently completed**: Keep for 2-4 weeks for easy reference
- **Archived completed**: Move to CHANGELOG.md or project notes

---

## Decision Framework: TODO vs BACKLOG

**Add to TODO (active list) if**:
- You'll work on it this week or month
- Priority is P0, P1, or high-P2
- Blocks other work
- Clear requirements and ready to start
- Effort is reasonable (≤8h)

**Move to BACKLOG if**:
- Won't work on it for >1 month
- Priority is P3 or low-P2
- Dependencies not resolved
- Requirements unclear or needs research
- Effort is very large (>8h, should break down first)
- Nice-to-have without clear immediate value

---

## TODO Entry Template

```markdown
- [ ] **TODO-XXX**: [Clear, concise title]
  - Estimate: S/M/L/XL (<1h / 1-4h / 4-8h / >8h)
  - Reason: [Why this work is important]
  - Created: YYYY-MM-DD
  - Benefits: [What improvements or value this provides]
  - [Optional] Blocked: [What's blocking progress]
  - [Optional] Progress notes: [Investigation findings, decisions made]
```

**Example**:
```markdown
- [ ] **TODO-042**: Implement caching layer for API responses
  - Estimate: M (3-4h)
  - Reason: API response times >500ms under load, affecting user experience
  - Created: 2026-01-15
  - Benefits: Faster API responses (<100ms), reduced database load, better scalability
  - Progress: Evaluated Redis vs Memcached, chose Redis for persistence
```

---

## Cross-References

- **Backlog**: `/dev/todo/BACKLOG.md` (future items, low priority work)
- **Priorities**: `/dev/todo/PRIORITIES.md` (decision guide, weekly planning)
- **Task Index**: `/dev/tasks/TASK_INDEX.md` (active tasks, detailed work tracking)
- **Daily Log**: `/dev/progress/DAILY_LOG.md` (daily progress updates)
- **Milestones**: `/dev/progress/MILESTONES.md` (major achievements)
- **Project Rules**: `/dev/rules/CLAUDE.md` (project knowledge and standards)
