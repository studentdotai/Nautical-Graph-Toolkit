# /dev Template Manifest

**Complete list of all files included in this template**

---

## 📦 What's Included

This template contains **31 files** organized in **7 directories** (plus templates subdirectories) to help you manage agentic development work.

---

## 📋 File Listing

### Root Level (4 files)

1. **README.md** - Main overview and guide
   - Purpose: Central documentation hub
   - Use: Start here to understand the system

2. **GETTING_STARTED.md** - Comprehensive getting started guide
   - Purpose: Step-by-step introduction
   - Use: Learn how to use the system

3. **QUICK_REFERENCE.md** - One-page cheat sheet
   - Purpose: Daily reference guide
   - Use: Quick lookups, print and keep visible

4. **MANIFEST.md** - This file
   - Purpose: Complete file listing
   - Use: Understand what's included

---

### rules/ Directory (3 files)

5. **rules/AGENTS.md**
   - Purpose: Agent behavioral guidelines (HOW/WHEN to operate)
   - Content: Operational principles, workflows, error handling, communication
   - Use: Reference for agent behavior patterns and task execution

6. **rules/CLAUDE.md**
   - Purpose: Project-specific knowledge reference (WHAT the project is)
   - Content: Architecture, dependencies, domain knowledge, configuration
   - Use: Reference for project understanding and technical details

7. **rules/CODE_STANDARDS.md**
   - Purpose: Coding standards and conventions
   - Content: Documentation standards, code style, testing, security
   - Use: Maintain code quality and consistency

8. **rules/WORKFLOW.md**
   - Purpose: Development workflow processes
   - Content: Planning, execution, tracking, completion phases
   - Use: Follow standard workflows for tasks

---

### templates/ Directory (12 items)

**templates/README.md**
   - Purpose: Template philosophy and usage guide
   - Content: How templates work, customization, integration
   - Use: Understand template system

**templates/progress.template/**
   - DAILY_LOG.md - Template for daily work journal (~150 lines)
   - CHANGELOG.md - Template for session changes (~130 lines)
   - MILESTONES.md - Template for major achievements (~100 lines)

**templates/tasks.template/**
   - TASK_INDEX.md - Template for master task list (~180 lines)
   - SAMPLE_TASK.md - Complete task file template (~120 lines)
   - active/.gitkeep - Preserve directory in git
   - completed/.gitkeep - Preserve directory in git

**templates/todo.template/**
   - TODO.md - Template for active work items (~170 lines)
   - BACKLOG.md - Template for future work (~100 lines)
   - PRIORITIES.md - Template for priority framework (~120 lines)

**Note**: Templates are tracked in git (shared). Personal state files created from templates are gitignored (private to each developer).

---

### templates/ Directory (11 items)

**templates/README.md**
   - Purpose: Template philosophy and usage guide
   - Content: How templates work, customization, integration
   - Use: Understand template system

**templates/progress.template/**
   - DAILY_LOG.md - Template for daily work journal (~150 lines)
   - CHANGELOG.md - Template for session changes (~130 lines)
   - MILESTONES.md - Template for major achievements (~100 lines)

**templates/tasks.template/**
   - TASK_INDEX.md - Template for master task list (~180 lines)
   - SAMPLE_TASK.md - Complete task file template (~120 lines)
   - active/.gitkeep - Preserve directory in git
   - completed/.gitkeep - Preserve directory in git

**templates/todo.template/**
   - TODO.md - Template for active work items (~170 lines)
   - BACKLOG.md - Template for future work (~100 lines)
   - PRIORITIES.md - Template for priority framework (~120 lines)

**Note**: Templates are tracked in git (shared). Personal state files created from templates are gitignored (private to each developer).

---

### scripts/ Directory (1 file)

**scripts/migrate_dev_environment.sh**
   - Purpose: Environment setup automation
   - Content: Bash script for initializing dev environment
   - Use: Run `/dev:setup` or execute directly

---

### tasks/ Directory (4 items)

11. **tasks/TASK_INDEX.md**
    - Purpose: Master list of all tasks
    - Content: Active, blocked, planned, completed tasks tracking
    - Use: Central task management hub

12. **tasks/SAMPLE_TASK.md** (in templates/)
    - Purpose: Template for creating task files
    - Content: Comprehensive task structure with all fields
    - Use: Reference for creating new tasks (or use `/dev:new-task`)

13. **tasks/active/** (directory)
    - Purpose: Store currently active task files
    - Format: `TASK-XXX-[name].md`
    - Use: Work in progress

14. **tasks/completed/** (directory)
    - Purpose: Archive completed task files
    - Format: Same as active, moved after completion
    - Use: Historical record

---

### todo/ Directory (3 files)

15. **todo/TODO.md**
    - Purpose: Immediate action items and work queue
    - Content: Prioritized TODO items, quick wins, weekly focus
    - Use: Daily planning and task tracking

16. **todo/BACKLOG.md**
    - Purpose: Future ideas and deferred items
    - Content: Ideas by category, someday/maybe, deferred TODOs
    - Use: Long-term planning, idea capture

17. **todo/PRIORITIES.md**
    - Purpose: Ranked list of work priorities
    - Content: What to do next, weekly/monthly goals, focus areas
    - Use: Daily decision-making, planning

---

### progress/ Directory (3 files)

18. **progress/DAILY_LOG.md**
    - Purpose: Daily work journal
    - Content: Daily entries with tasks, progress, decisions, blockers
    - Use: Track daily work, maintain history

19. **progress/CHANGELOG.md**
    - Purpose: Significant changes history
    - Content: Chronological log of notable changes
    - Use: Track major changes, maintain version history

20. **progress/MILESTONES.md**
    - Purpose: Major achievements tracking
    - Content: Completed, in-progress, and planned milestones
    - Use: Track big wins, celebrate achievements

---

## 📊 Statistics

- **Total Files**: 27+ (dev/) + 30+ (.claude/)
- **Total Directories**: 8 main (dev/) + 3 (.claude/) + template subdirectories
- **Documentation Files**: 24 (rules, core docs, skills)
- **Template Files**: 11 (progress.template, tasks.template, todo.template + README)
- **Index Files**: 1 (TASK_INDEX)
- **Skills**: 11 specialized skills in `.claude/skills/`
- **Slash Commands**: 15 commands in `.claude/commands/` (14 dev + 1 add-to-changelog)
- **Empty Directories Ready for Use**: 4 (tasks/active, tasks/completed, template subdirs)

---

## 🎯 File Categories

### Core Documentation (4 files)
- README.md
- GETTING_STARTED.md
- QUICK_REFERENCE.md
- MANIFEST.md

### Rules & Standards (4 files)
- AGENTS.md
- CLAUDE.md
- CODE_STANDARDS.md
- WORKFLOW.md

### Planning & Tracking (6 files)
- TODO.md
- BACKLOG.md
- PRIORITIES.md
- TASKS_INDEX.md
- DAILY_LOG.md
- CHANGELOG.md
- MILESTONES.md

### Templates (10 files)
- templates/progress.template/DAILY_LOG.md
- templates/progress.template/CHANGELOG.md
- templates/progress.template/MILESTONES.md
- templates/tasks.template/TASK_INDEX.md
- templates/tasks.template/SAMPLE_TASK.md
- templates/todo.template/TODO.md
- templates/todo.template/BACKLOG.md
- templates/todo.template/PRIORITIES.md
- templates/README.md
- scripts/migrate_dev_environment.sh

### Skills & Commands (outside dev/)
- `.claude/skills/` - 11 specialized skills
- `.claude/commands/` - 15 slash commands (14 dev + 1 add-to-changelog)
- `.claude/agents/` - Specialized agents

### Indexes (1 file)
- TASK_INDEX.md (TODO.md serves as TODO index)

---

## 📏 File Sizes (Approximate)

| File | Size | Lines | Purpose |
|------|------|-------|---------|
| README_DEV.md | Medium | ~300 | Overview |
| GETTING_STARTED.md | Very Large | ~600+ | Guide |
| QUICK_REFERENCE.md | Medium | ~400 | Cheat sheet |
| MANIFEST.md | Very Large | ~500+ | This file |
| AGENTS.md | Large | ~250 | Agent guidelines (HOW/WHEN) |
| CLAUDE.md | Very Large | ~330+ | Project knowledge (WHAT) |
| CODE_STANDARDS.md | Large | ~400+ | Code standards |
| DOCUMENTATION.md | Large | ~300 | Documentation standards |
| NOTEBOOK_STANDARDS.md | Very Large | ~1100+ | Notebook conventions |
| WORKFLOW.md | Large | ~430+ | Workflows |
| TASK_INDEX.md | Large | ~200+ | Tasks registry |
| SAMPLE_TASK.md | Medium | ~120 | Task template |
| TODO.md | Large | ~400 | TODO list |
| BACKLOG.md | Large | ~400 | Backlog |
| PRIORITIES.md | Large | ~450 | Priorities |
| DAILY_LOG.md | Large | ~350 | Daily journal |
| CHANGELOG.md | Large | ~300 | Change log |
| MILESTONES.md | Very Large | ~500 | Milestones |

**Total (dev/)**: ~6,000+ lines of comprehensive documentation and templates
**Total (.claude/)**: ~5,000+ lines of skills, commands, and agents

---

## 🔄 Customization Guide

### Files You Can Modify Freely
- All content files (add your own data)
- Template files in `dev/templates/` (adjust to your needs)
- README (customize for your project)

### Files to Keep Structure
- Index files (maintain format for consistency)
- Template files (keep sections, customize content)

### Files That Evolve
- Daily logs (grows continuously)
- Task files (created as needed via `/dev:new-task`)
- Skills (added to `.claude/skills/` following skill template)

---

## 🎯 Getting Started Checklist

Use this checklist when setting up your dev environment:

- [ ] Clone the project
- [ ] Run `/dev:setup` to initialize your personal environment
- [ ] Read README_DEV.md
- [ ] Read GETTING_STARTED.md
- [ ] Create your first TODO in TODO.md
- [ ] Create your first task using `/dev:new-task`
- [ ] Add first entry to DAILY_LOG.md
- [ ] Update TASK_INDEX.md with your task
- [ ] Start working!

---

## 📖 Recommended Reading Order

1. **README_DEV.md** - Get the overview
2. **QUICK_REFERENCE.md** - Learn the basics
3. **GETTING_STARTED.md** - Detailed walkthrough
4. **WORKFLOW.md** - Understand processes
5. **AGENTS.md** - Learn best practices
6. Start using the system!

---

## 💡 Key Features

This template provides:

✅ **Single Source of Truth** - Everything in one place
✅ **Linear History** - Clear progression of work
✅ **Comprehensive Templates** - Ready-to-use documentation
✅ **Flexible Structure** - Adapt to your needs
✅ **Best Practices** - Built-in guidance
✅ **Agent-Friendly** - Designed for AI collaboration
✅ **Scalable** - Works for small and large projects
✅ **Well-Documented** - Extensive guides and examples

---

## 🔗 File Relationships

```
README_DEV.md ─────┐
                   ├─→ GETTING_STARTED.md ─→ QUICK_REFERENCE.md
                   │
rules/ ────────────┼─→ AGENTS.md
                   │   CLAUDE.md
                   │   CODE_STANDARDS.md
                   │   DOCUMENTATION.md
                   │   NOTEBOOK_STANDARDS.md
                   │   WORKFLOW.md
                   │
templates/ ─────────┼─→ README.md
                   │   progress.template/
                   │   tasks.template/
                   │   todo.template/
                   │
scripts/ ───────────┼─→ migrate_dev_environment.sh
                   │
tasks/ ─────────────┼─→ TASK_INDEX.md
                   │   active/[tasks].md
                   │   completed/[tasks].md
                   │
todo/ ──────────────┼─→ TODO.md
                   │   BACKLOG.md
                   │   PRIORITIES.md
                   │
progress/ ──────────┼─→ DAILY_LOG.md
                   │   CHANGELOG.md
                   └─→ MILESTONES.md

.claude/ (external) ┐
                   ├─→ skills/[13 skills]
                   ├─→ commands/dev/[15 commands]
                   └─→ agents/[docs7-agent]
```

---

## 🎉 You're Ready!

You now have a complete, production-ready template for managing agentic development work.

**Next Steps:**
1. Copy to your project
2. Start with TODO.md
3. Create your first task
4. Begin logging in DAILY_LOG.md
5. Make it yours!

---

**Template Version**: 1.0  
**Created**: 2024  
**License**: Free to use and modify  
**Support**: See GETTING_STARTED.md for help

---

**Happy Organizing! 🚀**
