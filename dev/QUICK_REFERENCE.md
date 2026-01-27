# /dev Quick Reference

**One-page cheat sheet for daily use**

---

## 📁 Directory Structure

```
/dev/
├── README_DEV.md          # Main development hub (START HERE)
├── GETTING_STARTED.md     # Detailed onboarding guide
├── MANIFEST.md            # Complete file listing
├── QUICK_REFERENCE.md     # This file - one-page cheat sheet
├── rules/                 # How to work
│   ├── AGENTS.md         # Agent behavior (HOW/WHEN)
│   ├── CLAUDE.md         # Project knowledge (WHAT)
│   ├── CODE_STANDARDS.md # Code conventions
│   ├── DOCUMENTATION.md  # Documentation standards
│   └── WORKFLOW.md       # Processes
├── templates/             # Reusable templates (git-tracked)
│   ├── README.md         # Template philosophy
│   ├── progress.template/ # Progress tracking templates
│   │   ├── DAILY_LOG.md
│   │   ├── CHANGELOG.md
│   │   └── MILESTONES.md
│   ├── tasks.template/   # Task management templates
│   │   ├── TASK_INDEX.md
│   │   ├── SAMPLE_TASK.md
│   │   ├── active/.gitkeep
│   │   └── completed/.gitkeep
│   └── todo.template/    # Todo planning templates
│       ├── TODO.md
│       ├── BACKLOG.md
│       └── PRIORITIES.md
├── scripts/               # Utility scripts
│   └── migrate_dev_environment.sh
├── tasks/                 # Active work (gitignored personal state)
│   ├── TASK_INDEX.md
│   ├── active/           # Current tasks
│   └── completed/        # Done tasks
├── todo/                  # Planning (gitignored personal state)
│   ├── TODO.md
│   ├── BACKLOG.md
│   └── PRIORITIES.md
└── progress/              # History (gitignored personal state)
    ├── DAILY_LOG.md
    ├── CHANGELOG.md
    └── MILESTONES.md

.claude/                   # Executable components (outside dev/)
├── skills/                # 11 specialized skills
│   ├── dev-env-setup/
│   ├── environment-setup/
│   ├── graph-routing/
│   ├── s57-import/
│   └── ... (7 more skills)
├── commands/              # Slash commands
│   └── dev/              # 14 dev-specific commands
│       ├── session-start.md
│       ├── session-end.md
│       ├── new-todo.md
│       ├── new-task.md
│       └── ... (10 more)
└── agents/                # Specialized agents
    └── docs7-agent.md
```

---

## ⚡ Quick Commands

### Daily Workflow
```bash
# First-time setup (new developers)
/dev:setup
# or: bash dev/scripts/migrate_dev_environment.sh

# Morning: Check priorities
cat /dev/todo/PRIORITIES.md

# Start work: Create task (uses slash command)
/dev:new-task

# During: Log progress (uses slash command)
/dev:log-progress

# Evening: Update status
/dev:update-task

# Complete: Archive task
/dev:complete-task
```

### Search & Find
```bash
# Find task
grep "TASK-123" /dev/tasks/*.md

# Search logs
grep -n "keyword" /dev/progress/DAILY_LOG.md

# List available dev commands
ls .claude/commands/dev/

# Find skill
ls .claude/skills/
```

---

## 🎯 File Purposes

| File | Purpose | Update Frequency |
|------|---------|-----------------|
| `README_DEV.md` | Main development hub | Per major change |
| `templates/README.md` | Template philosophy | Per project change |
| `templates/*.template/` | Personal state templates | Per project change |
| `TODO.md` | Immediate action items | Daily |
| `PRIORITIES.md` | Ranked work list | Daily |
| `BACKLOG.md` | Future ideas | Weekly |
| `TASK_INDEX.md` | All tasks master list | Per task |
| `DAILY_LOG.md` | Daily journal | Multiple times daily |
| `CHANGELOG.md` | Significant changes | Per completion |
| `MILESTONES.md` | Major achievements | Per milestone |
| `scripts/migrate_dev_environment.sh` | Environment setup | As needed |

---

## 🔄 Core Workflows

### 0. First-Time Setup
```
Clone Project → Run /dev:setup → Verify Setup → Start Work
```

### 1. New Work Item
```
Idea → BACKLOG.md → TODO.md → PRIORITIES.md →
/dev:new-task → Update INDEX → Work → /dev:log-progress →
/dev:complete-task → Update CHANGELOG → Archive
```

### 2. Daily Routine
```
Morning: /dev:session-start (checks PRIORITIES.md)
During: Work + /dev:log-progress
Evening: /dev:session-end (updates TASKS_INDEX.md + DAILY_LOG.md)
```

### 3. Task Lifecycle
```
/dev:new-task (creates from template) → Active Work →
/dev:update-task (updates progress) → /dev:complete-task →
Update CHANGELOG → Move to completed/
```

---

## 📝 ID Formats

| Type | Format | Example |
|------|--------|---------|
| Task | TASK-XXX | TASK-001 |
| TODO | TODO-XXX | TODO-042 |
| Backlog | BACK-XXX | BACK-015 |
| Skill | [CAT]-XXX | DEV-003 |
| Milestone | MS-XXX | MS-001 |

---

## 🏷️ Status Labels

**Tasks:**
- 📋 Planned
- 🚀 Active  
- ⏸️ Blocked
- ✅ Completed
- ❌ Cancelled

**Priorities:**
- P0: Critical (now)
- P1: High (this week)
- P2: Medium (this month)
- P3: Low (when possible)

**Effort:**
- S: Small (< 1h)
- M: Medium (1-4h)
- L: Large (4h+)

---

## 📊 Log Templates

### Daily Log Entry
```markdown
### YYYY-MM-DD

#### Tasks
- TASK-XXX: [progress]

#### Progress  
- [accomplishment]

#### Blockers
[none / blocker]

#### Next
- [next step]
```

### Task Template
```markdown
## TASK-XXX: [Title]

**Created:** YYYY-MM-DD
**Status:** Active
**Priority:** P1

### Description
[What needs to be done]

### Acceptance Criteria
- [ ] Criteria 1
- [ ] Criteria 2

### Progress
[Log progress here]

### References
- Related issue: #XXX
- Parent task: TASK-YYY
```

### Changelog Entry
```markdown
## YYYY-MM-DD

### Added
- [Feature] (TASK-XXX)

### Fixed
- [Bug] (TASK-XXX)
```

---

## ✅ Daily Checklist

**First Time (One-time)**
- [ ] Run `/dev:setup` to initialize environment
- [ ] Verify setup completion
- [ ] Review templates in dev/templates/

**Morning (5 min)**
- [ ] Run `/dev:session-start` (checks PRIORITIES.md)
- [ ] Review active tasks
- [ ] Plan today's work

**During Work**
- [ ] Use `/dev:log-progress` as you work
- [ ] Update task files
- [ ] Document decisions

**Evening (5 min)**
- [ ] Run `/dev:session-end` (updates DAILY_LOG.md)
- [ ] Update task statuses
- [ ] Plan tomorrow

---

## 🎯 When to Use What

**Use TODO.md when:**
- Quick capture of ideas
- Planning immediate work
- Tracking action items

**Use BACKLOG.md when:**
- Long-term ideas
- Low priority items
- Need to defer something

**Use PRIORITIES.md when:**
- Deciding what's next
- Planning the week
- Checking focus

**Use TASKS_INDEX.md when:**
- Creating new task
- Checking task status
- Getting overview

**Use DAILY_LOG.md when:**
- Starting work
- Making progress
- Making decisions
- Ending session

**Use CHANGELOG.md when:**
- Completing significant work
- Making breaking changes
- Releasing features

**Use MILESTONES.md when:**
- Achieving big goals
- Planning quarters
- Celebrating wins

**Use slash commands when:**
- Starting/ending sessions: `/dev:session-start`, `/dev:session-end`
- Managing tasks: `/dev:new-task`, `/dev:complete-task`, `/dev:update-task`
- Managing todos: `/dev:new-todo`, `/dev:complete-todo`, `/dev:prioritize`
- Logging progress: `/dev:log-progress`
- Working with notebooks: `/dev:nb-convert`, `/dev:nb-sync`
- First-time setup: `/dev:setup`

---

## 🚫 Common Mistakes

| Mistake | Fix |
|---------|-----|
| Not logging daily | Set reminder, make it habit |
| Too much detail | Keep entries concise |
| Files scattered | Always save to /dev |
| Forgetting to update INDEX | Check before creating task |
| Not celebrating wins | Update MILESTONES.md |
| Everything is P0 | Prioritize ruthlessly |
| Vague task descriptions | Be specific and actionable |
| Starting without task file | Always create task first |

---

## 💡 Pro Tips

1. **Log as you work**, not at end of day
2. **Start with top priority** each morning
3. **Document decisions** immediately
4. **Update indexes** when creating files
5. **Break large tasks** into smaller ones
6. **Celebrate completions** in MILESTONES
7. **Review weekly** to stay on track
8. **Keep TODO short** (< 20 items)
9. **Use search** to find things
10. **Customize** for your needs

---

## 🔗 Quick Links

**Most Used:**
- `README_DEV.md` - Main development hub
- `todo/PRIORITIES.md` - What's next
- `progress/DAILY_LOG.md` - Daily work
- `tasks/TASK_INDEX.md` - All tasks

**Templates:**
- `templates/README.md` - Template philosophy
- `templates/tasks.template/SAMPLE_TASK.md`
- `templates/progress.template/` - Progress templates
- `templates/todo.template/` - Todo templates

**Skills & Commands:**
- `.claude/skills/` - 11 specialized skills
- `.claude/commands/dev/` - 14 dev slash commands
- `/dev:setup` - Initialize environment

**Guides:**
- `GETTING_STARTED.md` - Onboarding guide
- `rules/WORKFLOW.md` - Development processes

---

## 📞 Help

**Stuck?** Check:
1. `GETTING_STARTED.md` - Detailed guide
2. `rules/WORKFLOW.md` - Process details
3. `rules/AGENTS.md` - Best practices

**Questions?** Document in:
- `progress/DAILY_LOG.md` notes section

---

## 🎯 Remember

- **Single source of truth**: Everything in /dev
- **Linear history**: Clear progression
- **Log as you go**: Don't wait
- **Start simple**: Add complexity as needed
- **Make it yours**: Customize for your workflow

---

**Keep this handy for daily reference!**

---

## Keyboard Shortcuts (if using vim)

```
# Jump to today's log entry
/## `date +%Y-%m-%d`

# Jump to task
/TASK-XXX

# Add date stamp
:r !date +\%Y-\%m-\%d
```

---

## Shell Aliases to Add

```bash
# Add to ~/.bashrc or ~/.zshrc
alias devcd='cd /path/to/project/dev'
alias devlog='vim /path/to/project/dev/progress/DAILY_LOG.md'
alias devtodo='vim /path/to/project/dev/todo/TODO.md'
alias devpri='cat /path/to/project/dev/todo/PRIORITIES.md'
alias devtasks='cat /path/to/project/dev/tasks/TASK_INDEX.md'
alias devsetup='bash /path/to/project/dev/scripts/migrate_dev_environment.sh'
```

---

**Print this page and keep it visible!**
