# /dev - Agentic Development Hub

**Single Source of Truth for All Development Work**

This directory contains all rules, skills, tasks, todos, and progress tracking for agentic development projects. Everything is centralized here to avoid scattered files and maintain linear records.

## 📁 Directory Structure

```
/dev/
├── README_DEV.md               # This file - overview and guide
├── GETTING_STARTED.md          # Comprehensive onboarding guide
├── MANIFEST.md                 # Complete file listing
├── QUICK_REFERENCE.md          # One-page cheat sheet
├── rules/                      # Development rules and guidelines
│   ├── AGENTS.md              # Agent behavioral guidelines (HOW/WHEN)
│   ├── CLAUDE.md              # Project knowledge reference (WHAT)
│   ├── CODE_STANDARDS.md      # Coding standards and conventions
│   ├── DOCUMENTATION.md       # Documentation standards
│   ├── NOTEBOOK_STANDARDS.md  # Notebook conventions
│   └── WORKFLOW.md            # Development workflow rules
├── templates/                  # Reusable templates (git-tracked)
│   ├── README.md              # Template philosophy
│   ├── progress.template/     # Progress tracking templates
│   ├── tasks.template/        # Task management templates
│   └── todo.template/         # Todo planning templates
├── scripts/                    # Utility scripts
│   └── migrate_dev_environment.sh
├── tasks/                      # Active and completed tasks (gitignored)
│   ├── TASK_INDEX.md          # Master task list and status
│   ├── active/                # Currently active tasks
│   └── completed/             # Completed tasks (archived)
├── todo/                       # Todo items and backlog (gitignored)
│   ├── TODO.md                # Main todo list
│   ├── BACKLOG.md             # Future items and ideas
│   └── PRIORITIES.md          # Prioritized work items
└── progress/                   # Progress tracking and logs (gitignored)
    ├── CHANGELOG.md           # Chronological change log
    ├── DAILY_LOG.md           # Daily progress entries
    └── MILESTONES.md          # Major milestones achieved

.claude/                        # Executable components (outside dev/)
├── skills/                     # 11 specialized skills
│   ├── dev-env-setup/
│   ├── environment-setup/
│   ├── graph-routing/
│   ├── s57-import/
│   └── ... (7 more skills)
├── commands/                    # Slash commands
│   └── dev/                   # 14 dev-specific commands
└── agents/                     # Specialized agents
    └── docs7-agent.md
```

## 🚀 First-Time Setup

**New developers** must initialize their personal dev environment before using this system:

```bash
# Option 1: Using Claude Code skill (recommended)
/dev:setup

# Option 2: Using migration script
bash dev/scripts/migrate_dev_environment.sh
```

This creates your personal state files (progress/, tasks/, todo/) from templates and configures .gitignore. Your work stays private while project rules and templates remain shared.

**See**: `/dev/GETTING_STARTED.md` for detailed onboarding guide, `/dev/templates/README.md` for template philosophy, and `.claude/skills/` for specialized skills.

---

## 🎯 Quick Start (After Setup)

1. **Before Starting Work**: Check `todo/TODO.md` and `tasks/TASKS_INDEX.md`
2. **During Work**: Log progress in `progress/DAILY_LOG.md`
3. **After Completion**: Update `progress/CHANGELOG.md` and move task to `tasks/completed/`
4. **Need Help?**: Check `rules/AGENTS.md` (operations), `rules/CLAUDE.md` (project knowledge), or `.claude/skills/` (specialized procedures)

## 📋 Usage Guidelines

### Creating New Tasks
- Use `/dev:new-task` slash command (creates from template)
- Or manually create file in `tasks/active/[TASK-ID].md` from `templates/tasks.template/SAMPLE_TASK.md`
- Add to `tasks/TASK_INDEX.md` with unique ID
- Reference related skills from `.claude/skills/`

### Recording Progress
- Daily updates go in `progress/DAILY_LOG.md` (append to top)
- Significant changes go in `progress/CHANGELOG.md`
- Milestone achievements go in `progress/MILESTONES.md`

### Managing Skills
- Skills are located in `.claude/skills/[skill-name]/SKILL.md`
- 11 specialized skills available (environment-setup, graph-routing, s57-import, etc.)
- Access via slash commands (e.g., `/graph-routing`, `/s57-import`)
- Skills are git-tracked and shared across the project

### Following Rules
- Check `rules/AGENTS.md` for agent behavioral guidelines
- Check `rules/CLAUDE.md` for project-specific knowledge
- Follow `rules/CODE_STANDARDS.md` for code quality
- Adhere to `rules/WORKFLOW.md` for process

## 🔄 Workflow

1. **Plan** → Add items to `todo/TODO.md`
2. **Prioritize** → Update `todo/PRIORITIES.md`
3. **Execute** → Create task in `tasks/active/`
4. **Track** → Log in `progress/DAILY_LOG.md`
5. **Complete** → Update `progress/CHANGELOG.md`, move to `tasks/completed/`
6. **Review** → Check `progress/MILESTONES.md` for achievements

## 🎨 Benefits

- ✅ **Single Source of Truth**: Everything in one place
- ✅ **Linear History**: Clear progression from planning to completion
- ✅ **No Scattered Files**: Organized hierarchy prevents chaos
- ✅ **Audit Trail**: Complete record of decisions and changes
- ✅ **Reusable Knowledge**: Skills and rules are easily referenced
- ✅ **Progress Visibility**: Clear view of what's done and what's next

## 📊 Status at a Glance

Check these files for quick overview:
- **What's Next?** → `todo/PRIORITIES.md`
- **What's Active?** → `tasks/TASK_INDEX.md`
- **What Changed Today?** → `progress/DAILY_LOG.md`
- **What Have We Achieved?** → `progress/MILESTONES.md`
- **Available Skills?** → `.claude/skills/` (or use slash commands like `/graph-routing`)

---

**Last Updated**: [Current Date]
**Maintained By**: [Your Name/Team]
