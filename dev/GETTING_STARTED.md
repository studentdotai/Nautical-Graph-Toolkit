# Getting Started with /dev

**Your guide to using the /dev directory structure for agentic development**

## 🎯 Quick Start (5 Minutes)

### 1. First Time Setup

**New developers** - Initialize your personal dev environment:

```bash
# Option 1: Using Claude Code skill (recommended)
/dev:setup

# Option 2: Using migration script
bash dev/scripts/migrate_dev_environment.sh
```

**What this does**:
- Creates personal state files from templates (progress/, tasks/, todo/)
- Sets up .gitignore to exclude your personal work from git
- Initializes session tracking
- Verifies setup completed successfully

**Setup modes**:
- **Fresh Install**: New developer onboarding (auto-detected)
- **Preserve Mode**: Keep existing work, update templates only
- **Backup & Reset**: Archive current state, start fresh

**After setup**, you'll have:
- ✓ Structured files for tracking work (DAILY_LOG, TODO, TASK_INDEX)
- ✓ Templates with examples showing expected format
- ✓ Personal state excluded from git (your work stays private)
- ✓ Session tracking initialized

See `/dev/templates/README.md` for template details.

### 2. Your First Task
```bash
# 1. Add a TODO
# Edit todo/TODO.md and add your first item

# 2. Create a Task (using slash command)
/dev:new-task

# 3. Update Index
# Add task to tasks/TASK_INDEX.md

# 4. Start Logging
# Add today's entry to progress/DAILY_LOG.md

# 5. Begin Work!
```

### 3. Daily Routine
```bash
# Morning: Check priorities
cat todo/PRIORITIES.md

# During Work: Log progress
# Update progress/DAILY_LOG.md

# End of Day: Wrap up
# Update task status, log completion
```

---

## 📁 Directory Overview

### **rules/** - Your Operating Manual
- `AGENTS.md` - Agent behavioral guidelines (HOW/WHEN)
- `CLAUDE.md` - Project knowledge reference (WHAT)
- `CODE_STANDARDS.md` - Coding conventions
- `WORKFLOW.md` - Development processes

**Use when**: Setting up processes, onboarding, making decisions

### **skills/** - Your Knowledge Base
- `.claude/skills/` - 11 specialized skills (accessible via slash commands)
- Skills cover: GDAL setup, PostGIS, S-57 import, graph routing, testing, and more

**Use when**: Learning new techniques, solving common problems
**Access via**: Slash commands like `/graph-routing`, `/s57-import`, `/environment-setup`

### **tasks/** - Your Active Work
- `TASKS_INDEX.md` - All tasks in one place
- `active/` - Current work
- `completed/` - Finished tasks

**Use when**: Planning, executing, tracking work

### **todo/** - Your Planning Hub
- `TODO.md` - Immediate action items
- `BACKLOG.md` - Future ideas
- `PRIORITIES.md` - What's most important

**Use when**: Planning, prioritizing, managing backlog

### **progress/** - Your History
- `DAILY_LOG.md` - Daily journal
- `CHANGELOG.md` - Significant changes
- `MILESTONES.md` - Major achievements

**Use when**: Tracking progress, reflecting, reporting

---

## 🔄 Typical Workflows

### Workflow 1: Starting New Work

1. **Check priorities**
   ```bash
   # See what's most important
   cat todo/PRIORITIES.md
   ```

2. **Create task from TODO**
   ```bash
   # Use slash command (recommended)
   /dev:new-task

   # Or manually copy template
   cp dev/templates/tasks.template/SAMPLE_TASK.md dev/tasks/active/TASK-001-feature-name.md

   # Fill in details
   # Update TASK_INDEX.md
   ```

3. **Begin work and log**
   ```bash
   # Add entry to DAILY_LOG.md at the top
   # Start working
   ```

### Workflow 2: During Active Work

1. **Update task file regularly**
   - Add progress notes
   - Document decisions
   - Note blockers

2. **Log in DAILY_LOG.md**
   - Update throughout the day
   - Don't wait until evening

3. **Reference skills as needed**
   ```bash
   # List available skills
   ls .claude/skills/

   # Use skill via slash command
   /graph-routing    # For maritime routing workflows
   /s57-import       # For S-57 ENC conversions
   /environment-setup  # For GDAL/PostGIS setup
   ```

### Workflow 3: Completing Work

1. **Verify completion**
   - Check all acceptance criteria
   - Update task file with summary

2. **Update all tracking**
   - `TASKS_INDEX.md` → Status: Completed
   - `CHANGELOG.md` → Add entry
   - `DAILY_LOG.md` → Log completion

3. **Archive task**
   ```bash
   # Move to completed
   mv tasks/active/TASK-001-name.md tasks/completed/
   ```

4. **Check if milestone achieved**
   - Update `MILESTONES.md` if applicable

---

## 💡 Best Practices

### Daily Habits

**Morning (5 min)**
- Check `PRIORITIES.md`
- Review active tasks in `TASKS_INDEX.md`
- Plan your day

**During Work**
- Log as you go in `DAILY_LOG.md`
- Update task files with progress
- Document decisions immediately

**Evening (5 min)**
- Complete DAILY_LOG entry
- Update task statuses
- Plan tomorrow's priorities

### Weekly Habits

**End of Week (15 min)**
- Write weekly summary in `DAILY_LOG.md`
- Update `TASKS_INDEX.md` statistics
- Review `PRIORITIES.md`
- Plan next week

**Start of Week (10 min)**
- Review last week's log
- Set weekly goals
- Update priorities

### Monthly Habits

**End of Month (30 min)**
- Review `CHANGELOG.md`
- Update `MILESTONES.md`
- Deep review of progress
- Clean up TODO/BACKLOG
- Plan next month

---

## 🎯 Common Scenarios

### Scenario: "I have a new idea"
1. Add to `todo/BACKLOG.md`
2. Include enough context for future you
3. Estimate value and effort
4. Review during monthly planning

### Scenario: "I need to start a new task"
1. Check if in `TODO.md`, if not add it
2. Use `/dev:new-task` slash command (creates from template)
3. Fill in all sections
4. Add to `TASK_INDEX.md`
5. Log start in `DAILY_LOG.md`

### Scenario: "I'm blocked on something"
1. Document blocker in task file
2. Log blocker in `DAILY_LOG.md`
3. Update task status to "Blocked"
4. Identify who/what can unblock
5. Work on something else

### Scenario: "I completed a major milestone"
1. Complete all related tasks
2. Update `CHANGELOG.md`
3. Add entry to `MILESTONES.md`
4. Celebrate! 🎉
5. Share with team

### Scenario: "I need to find old work"
```bash
# Search daily logs
grep -n "keyword" progress/DAILY_LOG.md

# Find task
grep -r "TASK-123" tasks/

# Check changelog
grep "feature-name" progress/CHANGELOG.md
```

---

## 🚀 Power User Tips

### Tip 1: Use Shell Aliases
```bash
# Add to ~/.bashrc or ~/.zshrc
alias devlog='cd /path/to/dev && vim progress/DAILY_LOG.md'
alias devtodo='cd /path/to/dev && vim todo/TODO.md'
alias devpri='cd /path/to/dev && cat todo/PRIORITIES.md'
```

### Tip 2: Create Scripts
```bash
# Create new task script
#!/bin/bash
TASK_ID=$1
TASK_NAME=$2
cp dev/templates/tasks.template/SAMPLE_TASK.md "dev/tasks/active/TASK-${TASK_ID}-${TASK_NAME}.md"
echo "Created TASK-${TASK_ID}-${TASK_NAME}.md"
```

### Tip 3: Use Git
```bash
# Track your /dev directory with git
cd /path/to/dev
git init
git add .
git commit -m "Initialize dev structure"
```

### Tip 4: Automate Daily Log
```bash
# Add to cron or use a script
#!/bin/bash
DATE=$(date +%Y-%m-%d)
echo -e "\n### $DATE\n\n#### Tasks\n\n#### Progress\n" >> progress/DAILY_LOG.md
```

---

## 📊 Measuring Success

### Track These Metrics

**Completion Rate**
- Tasks completed per week
- TODO to Task conversion rate

**Time Management**
- Estimated vs actual effort
- Time to complete tasks

**Quality**
- Blockers encountered
- Decisions documented
- Lessons learned captured

**Consistency**
- Daily log entries
- Weekly reviews completed
- Monthly reviews completed

---

## 🔧 Customization

### Adapt to Your Needs

This template is a starting point. Feel free to:
- Add new categories to TODO
- Create custom task templates
- Add more skill categories
- Customize progress tracking
- Add your own sections

**Just maintain these principles:**
- Single source of truth
- Linear history
- Clear organization
- Regular updates

---

## ❓ FAQ

**Q: Do I need to use all files?**
A: Start with basics (TODO, TASKS, DAILY_LOG). Add others as needed.

**Q: How much detail in DAILY_LOG?**
A: Enough to remember context later. 2-5 minutes per entry.

**Q: What if I forget to log?**
A: Add entry when you remember. Better late than never.

**Q: Can I use this for team projects?**
A: Yes! Share /dev directory, assign owners, coordinate updates.

**Q: What if a task takes weeks?**
A: Break into smaller tasks. Update progress regularly.

**Q: Should I track ALL work here?**
A: Track significant work. Don't track every tiny thing.

**Q: How to handle interruptions?**
A: Log them in DAILY_LOG. Track as separate small tasks if needed.

---

## 🆘 Troubleshooting

### Problem: "I'm overwhelmed by templates"
**Solution**: Start minimal. Use TODO.md, TASKS_INDEX.md, DAILY_LOG.md only.

### Problem: "My logs are getting too long"
**Solution**: Archive old entries. Create monthly files.

### Problem: "I forget to update"
**Solution**: Set reminders. Make it part of your routine.

### Problem: "Too much overhead"
**Solution**: Simplify. Use shorter templates. Log less detail.

### Problem: "Files are scattered again"
**Solution**: Review workflow. Always save to /dev. Use search.

---

## 🎓 Learning Path

### Week 1: Basics
- Set up directory
- Use TODO.md
- Create first task
- Start DAILY_LOG.md

### Week 2: Workflows
- Follow WORKFLOW.md
- Update TASKS_INDEX.md
- Use PRIORITIES.md
- Complete first task

### Week 3: Depth
- Create first skill
- Update CHANGELOG.md
- Add to BACKLOG.md
- Weekly review

### Week 4: Mastery
- Monthly review
- Milestone tracking
- Optimize for your style
- Share with others

---

## 🔗 Additional Resources

### Templates Location
All templates are in the `/dev/templates/` directory:
- Task: `templates/tasks.template/SAMPLE_TASK.md`
- Progress: `templates/progress.template/` (DAILY_LOG.md, CHANGELOG.md, MILESTONES.md)
- Todo: `templates/todo.template/` (TODO.md, BACKLOG.md, PRIORITIES.md)
- Skills: `.claude/skills/[skill-name]/SKILL.md` (11 specialized skills)

### Key Files to Bookmark
1. `README.md` - Overview
2. `todo/PRIORITIES.md` - What's next
3. `progress/DAILY_LOG.md` - Daily journal
4. `tasks/TASKS_INDEX.md` - All tasks

### External Resources
- Keep a Changelog: https://keepachangelog.com/
- Getting Things Done: GTD methodology
- SMART Goals: Goal-setting framework

---

## ✨ Success Stories

**Benefit 1: No More Scattered Files**
- Everything in one place
- Easy to find
- Clear structure

**Benefit 2: Clear History**
- Linear progression
- Audit trail
- Lessons preserved

**Benefit 3: Better Planning**
- Priorities clear
- Dependencies visible
- Progress tracked

**Benefit 4: Reduced Context Switching**
- All context in one place
- Easy to resume work
- Faster onboarding

---

## 🎉 You're Ready!

You now have:
- ✅ Complete /dev structure
- ✅ Understanding of workflows
- ✅ Templates for everything
- ✅ Best practices guide

**Next Steps:**
1. Copy /dev to your project
2. Create your first TODO
3. Start your DAILY_LOG
4. Begin your first task

**Remember**: Start simple, add complexity as needed. The goal is to help you work better, not add overhead.

---

**Happy Coding! 🚀**

For questions or improvements, update `progress/DAILY_LOG.md` with your thoughts and iterate on the system.
