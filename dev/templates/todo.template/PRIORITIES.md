# Priorities

What to work on next - Daily decision-making guide.

**Purpose**: Help you decide what to work on each day/week. Provides framework for prioritizing work and tracking current focus areas.

Last Updated: YYYY-MM-DD

## Current Focus

**This Week (YYYY-MM-DD to YYYY-MM-DD)**
- Primary: [Main focus for the week]
- Secondary: [Secondary focus or backup work]

**This Month (Month YYYY)**
- [Key goal 1]
- [Key goal 2]
- [Key goal 3]

---

## Example Current Focus (After Setting Priorities)

**This Week (2026-01-15 to 2026-01-21)**
- Primary: Implement user authentication system (TODO-005, TASK-003)
- Secondary: Add API documentation (TODO-007)

**This Month (January 2026)**
- Complete authentication and authorization system
- Improve test coverage to >80%
- Set up CI/CD pipeline

---

## Top 3 Priorities (Right Now)

### 1. [Priority Title] (P0/P1) ⚡

- **Why**: [Why this is important right now]
- **Impact**: [What this will enable or improve]
- **Effort**: S/M/L/XL (<1h / 1-4h / 4-8h / >8h)
- **Blockers**: [Current blockers or "None"]
- **Progress**: X% complete (or "not started")
- **Next Step**: [Immediate next action]
- **Task**: TASK-XXX (or "Not yet created")

### 2. [Priority Title] (P1)

- **Why**: [Rationale for priority]
- **Impact**: [Expected outcomes]
- **Effort**: [Time estimate]
- **Blockers**: [Dependencies or blockers]
- **Next Step**: [What to do first]
- **Task**: [Task reference if exists]

### 3. [Priority Title] (P1/P2)

- **Why**: [Importance]
- **Impact**: [Benefits]
- **Effort**: [Estimate]
- **Blockers**: [Blockers]
- **Next Step**: [Action item]
- **Task**: [Task reference]

---

## Example Top 3 Priorities

### 1. Complete User Authentication System (P0) ⚡

- **Why**: Security requirement blocking multi-tenant deployment to production
- **Impact**: Secure API, enable user management, unlock RBAC features
- **Effort**: L (6-8h, ~3h remaining)
- **Blockers**: None
- **Progress**: 60% complete (JWT done, RBAC in progress)
- **Next Step**: Implement role-based access control decorators
- **Task**: TASK-003

### 2. Add Comprehensive API Documentation (P1)

- **Why**: Developers can't use API effectively without clear docs
- **Impact**: Better developer experience, reduced support burden, faster adoption
- **Effort**: M (3-4h)
- **Blockers**: Waiting for auth system to complete (endpoint permissions)
- **Next Step**: Generate OpenAPI spec from existing routes
- **Task**: Not yet created (create when starting)

### 3. Improve Test Coverage (P2)

- **Why**: Current 60% coverage insufficient for production confidence
- **Impact**: Higher confidence in changes, catch edge cases, reduce bugs
- **Effort**: M (2-3h per module, multiple modules)
- **Blockers**: None
- **Next Step**: Start with most critical module (auth module after completion)
- **Task**: Not yet created

---

## Weekly Goals

**Week 1 (YYYY-MM-DD to YYYY-MM-DD)**
- [ ] Goal 1
- [ ] Goal 2
- [ ] Goal 3

**Week 2 (YYYY-MM-DD to YYYY-MM-DD)**
- [ ] Goal 1
- [ ] Goal 2
- [ ] Goal 3

---

## Example Weekly Goals

**Week 1 (2026-01-15 to 2026-01-21)**
- [x] Complete JWT authentication implementation
- [ ] Implement RBAC system (in progress, 60% done)
- [ ] Write integration tests for auth flow
- [ ] Start API documentation generation

**Week 2 (2026-01-22 to 2026-01-28)**
- [ ] Complete API documentation
- [ ] Improve test coverage for auth module
- [ ] Set up CI/CD pipeline
- [ ] Performance optimization review

---

## Monthly Goals

**[Month YYYY]**
- [Goal 1 with measurable outcome]
- [Goal 2 with acceptance criteria]
- [Goal 3 with deadline]

**Example**:

**January 2026**
- Complete authentication and authorization system (all endpoints secured)
- Achieve >80% test coverage (current: 60%)
- Deploy CI/CD pipeline (automated testing on all PRs)
- Document all API endpoints (OpenAPI spec + Swagger UI)

**February 2026** (Tentative)
- Performance optimization sprint (target 50% improvement)
- Database scaling improvements (handle 10K requests/sec)
- Add monitoring and observability (Prometheus + Grafana)
- Write developer onboarding guide

---

## Focus Areas

Distribution of effort across categories:

- **[Category 1] (XX%)**: [What this includes]
- **[Category 2] (XX%)**: [What this includes]
- **[Category 3] (XX%)**: [What this includes]
- **[Category 4] (XX%)**: [What this includes]

**Example**:

- **Core Features (40%)**: New functionality, feature enhancements, user-facing improvements
- **Code Quality (30%)**: Testing, documentation, refactoring, tech debt reduction
- **Infrastructure (20%)**: CI/CD, deployment, monitoring, developer tooling
- **Research (10%)**: Exploring new technologies, proof of concepts, performance analysis

---

## Decision Matrix

### When Choosing Next Task

**High Priority If**:
- Blocks other work or team members
- Fixes critical bug or security issue
- Roadmap commitment (milestone deadline)
- Significant user impact or pain point
- Reduces technical debt materially
- Improves developer productivity significantly

**Medium Priority If**:
- Important but not blocking
- Clear value but no deadline pressure
- Moderate user impact
- Code quality improvement
- Nice-to-have feature with demand

**Low Priority If**:
- Nice-to-have without clear demand
- No immediate impact
- High effort with unclear return
- Purely aesthetic or cosmetic
- Experimental without clear goal

**Defer If**:
- Depends on external factors
- Requires major refactoring without clear benefit
- No clear requirements or acceptance criteria
- Low value vs effort investment
- Can be automated or solved differently
- Not aligned with current project goals

---

## Impact × Urgency Matrix

```
High Impact, High Urgency    │ High Impact, Low Urgency
P0 - DO NOW                   │ P1 - SCHEDULE THIS WEEK
• Critical bugs               │ • Important features
• Security issues             │ • Strategic improvements
• Production blockers         │ • Quality investments
──────────────────────────────┼──────────────────────────────
Low Impact, High Urgency      │ Low Impact, Low Urgency
P2 - QUICK WINS               │ P3 - BACKLOG
• Minor bugs with workarounds │ • Nice-to-have features
• Small improvements          │ • Low-value enhancements
• Quick documentation fixes   │ • Experimental ideas
```

**How to use**:
1. Assess impact: How much does this improve the project/user experience?
2. Assess urgency: How soon must this be done? What's the cost of delay?
3. Plot on matrix: Determines priority level (P0/P1/P2/P3)
4. Review weekly: Urgency and impact can change over time

---

## This Week's Breakdown

| Day | Primary Focus | Secondary Focus | Time Available |
|-----|---------------|-----------------|----------------|
| Mon | [Main task] | [Backup work] | [Hours] |
| Tue | [Main task] | [Backup work] | [Hours] |
| Wed | [Main task] | [Backup work] | [Hours] |
| Thu | [Main task] | [Backup work] | [Hours] |
| Fri | [Main task] | [Backup work] | [Hours] |

**Example**:

| Day | Primary Focus | Secondary Focus | Time Available |
|-----|---------------|-----------------|----------------|
| Mon | RBAC implementation | API docs research | 6h |
| Tue | RBAC testing | Auth integration tests | 6h |
| Wed | API docs generation | Test coverage | 5h |
| Thu | API docs UI | Performance review | 6h |
| Fri | Documentation | Weekly review + planning | 4h |

---

## Priority Management Guidelines

### Setting Priorities

**Daily**:
- Review top 3 priorities
- Adjust based on blockers/progress
- Plan work for today

**Weekly**:
- Review and update top 3 priorities
- Set weekly goals
- Plan this week's breakdown
- Review last week's outcomes
- Update focus areas if needed

**Monthly**:
- Set monthly goals
- Review progress on milestones
- Adjust focus area percentages
- Plan next month tentatively

### Tracking Progress

**Update frequency**:
- Top 3: Daily (progress percentage, blockers)
- Weekly goals: Daily (check off completed, update in-progress)
- Monthly goals: Weekly (assess trajectory)
- Focus areas: Monthly (adjust based on actual time spent)

**Progress indicators**:
- 0%: Not started
- 25%: Initial work done, design/planning complete
- 50%: Core implementation done, needs testing/polish
- 75%: Testing done, documentation in progress
- 100%: Complete, merged, documented

### Handling Blockers

**When blocked**:
1. Document blocker clearly
2. Identify what's needed to unblock
3. Move to secondary focus
4. Check blocker status daily
5. Escalate if blocker persists >3 days

**Types of blockers**:
- **External**: Waiting on third party, external approval
- **Dependency**: Waiting on other task/person
- **Technical**: Technical challenge needs research/help
- **Requirements**: Unclear requirements need clarification

---

## Cross-References

- **TODO List**: `/dev/todo/TODO.md` (active work items)
- **Backlog**: `/dev/todo/BACKLOG.md` (future work, deferred items)
- **Tasks**: `/dev/tasks/TASK_INDEX.md` (detailed task tracking)
- **Milestones**: `/dev/progress/MILESTONES.md` (major goals and achievements)
- **Daily Log**: `/dev/progress/DAILY_LOG.md` (daily progress updates)
- **Project Rules**: `/dev/rules/CLAUDE.md` (project context and standards)
