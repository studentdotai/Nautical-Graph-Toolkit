# Development Templates

Templates for initializing personal developer state files. These provide structure and examples for tracking your work.

## Purpose

This directory contains template files that are copied to your personal dev state directories during setup. Templates show the expected format and structure for:
- Progress tracking (daily logs, changelogs, milestones)
- Task management (task index, individual task files)
- Work planning (TODOs, backlog, priorities)

## Philosophy

**Templates provide structure, not constraints**:
- Show format with real examples (not just empty structure)
- Include inline instructions and guidelines
- Demonstrate best practices from actual project work
- Allow customization to fit your workflow

**Separation of concerns**:
- **Templates** (this directory): Tracked in git, shared across developers
- **Personal state** (dev/progress/, dev/tasks/, dev/todo/): Gitignored, unique to each developer
- **Project rules** (dev/rules/): Tracked in git, project standards for all

## Template Files

### progress.template/ (3 files)

**DAILY_LOG.md** (~150 lines)
- Structure: Date headers, work sections, decisions, results
- Sample entry: Complete example showing expected detail level
- Guidelines: When to update, what to include, retention policy
- Purpose: Daily work journal with structured format

**CHANGELOG.md** (~130 lines)
- Format: Keep a Changelog standard with categories
- Sample entry: Development session with notable changes
- Guidelines: What to log vs what goes in root /CHANGELOG.md
- Purpose: Track significant changes per session

**MILESTONES.md** (~100 lines)
- Structure: Completed, in-progress, and planned milestones
- Sample milestone: Full example with metrics and impact
- Guidelines: When to create, how to track, archival
- Purpose: Major project achievements and goals

### tasks.template/ (4 items)

**TASK_INDEX.md** (~180 lines)
- Structure: Statistics, active/blocked/completed tables, lifecycle
- Sample entries: Task details with metadata
- Guidelines: Task management workflow, archival process
- Purpose: Master registry of all development tasks

**SAMPLE_TASK.md** (~120 lines)
- Complete template: All sections with examples
- Metadata: Priority, status, effort, dependencies
- Sections: Description, acceptance criteria, plan, progress, outcomes
- Purpose: Template for creating new task files

**active/.gitkeep** (empty)
- Purpose: Preserve directory structure in git

**completed/.gitkeep** (empty)
- Purpose: Preserve directory structure in git

### todo.template/ (3 files)

**TODO.md** (~170 lines)
- Structure: P0-P3 priority levels, recently completed
- Sample entries: TODOs with estimates and rationale
- Guidelines: List management, promotion to tasks
- Purpose: Active work items ready to be worked on

**BACKLOG.md** (~100 lines)
- Structure: Categorized future work (features, performance, testing, docs)
- Sample entries: Items with value/effort assessment
- Guidelines: Backlog management, promotion criteria
- Purpose: Future ideas and deferred work

**PRIORITIES.md** (~120 lines)
- Structure: Current focus, top 3, weekly goals, decision matrix
- Sample planning: Week breakdown, impact×urgency matrix
- Guidelines: Priority setting, progress tracking
- Purpose: Daily decision-making guide for what to work on

## Usage

### First-Time Setup

**Automatic** (via skill):
```bash
# Trigger the dev-env-setup skill
/dev:setup
```

**Manual** (via script):
```bash
# Run migration script
bash dev/scripts/migrate_dev_environment.sh
```

**What happens**:
1. Templates copied to personal state locations
2. Dates updated to current date
3. .gitignore configured
4. Session tracking initialized

### Fresh Install Mode

Creates new personal state from templates:
```
dev/templates/progress.template/* → dev/progress/*
dev/templates/tasks.template/*    → dev/tasks/*
dev/templates/todo.template/*     → dev/todo/*
```

Result: Clean start with structured files ready for your work.

### Preserve Mode

Keeps your existing work, no file operations:
- Your 20+ log entries preserved
- Your active/completed tasks preserved
- Your TODO items preserved
- Templates remain available for reference

Result: No changes to your data.

### Backup & Reset Mode

Archives current state, then fresh install:
1. Creates `dev_backup_YYYYMMDD_HHMMSS/`
2. Copies all personal state to backup
3. Removes old personal state
4. Copies fresh templates

Result: Clean slate with safety net.

## Customizing Templates

### When to Update Templates

**Update templates when**:
- Project evolves and format needs change
- Better examples or structures emerge
- Guidelines need clarification
- New sections prove useful

**Don't update for**:
- Personal preferences (customize your files, not templates)
- Project-specific content (that goes in dev/rules/)
- Temporary experiments (test in your files first)

### How to Update Templates

1. **Test changes in your personal files first**:
   ```bash
   # Make changes to dev/progress/DAILY_LOG.md
   # Use new format for 1-2 weeks
   # Verify it works well
   ```

2. **Update template to match**:
   ```bash
   # Edit template file
   vim dev/templates/progress.template/DAILY_LOG.md

   # Sanitize: Remove personal data, generalize examples
   # Keep structure and guidelines from your working file
   ```

3. **Commit to git**:
   ```bash
   git add dev/templates/
   git commit -m "Update templates: improve daily log format"
   ```

4. **Share with team**:
   - Templates are tracked in git
   - Other developers get updates on pull
   - New developers get latest templates on setup

### Sanitizing Content

When updating templates from personal files:

**Remove**:
- Personal information (names, dates, specific details)
- Project-specific implementation details
- Sensitive or confidential information
- Actual performance metrics (use generic examples)

**Keep**:
- Structure and section headers
- Format and style conventions
- Guidelines and best practices
- 1-2 sanitized examples showing format

**Add**:
- Inline instructions (how to use each section)
- Usage notes (when to update, what to include)
- Cross-references to related files
- Guidelines for retention and archival

## Template Quality Standards

### Structure

- Clear hierarchical organization
- Consistent markdown formatting
- Logical section progression
- Comprehensive but not overwhelming

### Examples

- Real but sanitized (based on actual usage)
- Show expected level of detail
- Cover common scenarios
- Include edge cases

### Guidelines

- Actionable and specific
- Based on project experience
- Explain rationale, not just rules
- Reference related documentation

### Maintenance

- Review quarterly
- Update when patterns change
- Keep synchronized with dev/rules/
- Test with new developers

## Integration with Project

### Tracked in Git

Templates are part of project repository:
- `dev/templates/` directory committed to git
- Updates pulled by all developers
- Version controlled like other documentation
- Part of project standards

### Used by Setup System

Multiple tools use these templates:
- `/dev:setup` Claude skill
- `dev/scripts/migrate_dev_environment.sh` script
- CI/CD environment setup
- New developer onboarding

### Relationship to Project Rules

**Templates** (this directory):
- Personal state file formats
- Work tracking structures
- How to document your work

**Project rules** (dev/rules/):
- Project knowledge and architecture
- Code standards and conventions
- Development workflows
- Agent collaboration patterns

Templates = HOW to track your work
Rules = WHAT standards to follow

## Troubleshooting

### Templates not found during setup

**Cause**: Missing or deleted template files

**Solution**:
```bash
# Verify templates exist
ls -la dev/templates/

# Pull from git if missing
git checkout dev/templates/

# Verify structure
ls -la dev/templates/*/
```

### Template dates not updated

**Cause**: Setup script sed command failed

**Solution**:
```bash
# Manually update YYYY-MM-DD to current date
find dev/progress dev/tasks dev/todo -name "*.md" -exec sed -i "s/YYYY-MM-DD/$(date +%Y-%m-%d)/g" {} \;
```

### Personal files don't match template format

**Normal**: Templates are starting points, customization is expected

**Recommendations**:
- Keep core structure (headers, sections)
- Adapt guidelines to your workflow
- Document your conventions
- Consider updating templates if improvements are widely useful

## Examples

### Scenario 1: New Developer

1. Clones project
2. Runs `/dev:setup` or migration script
3. Templates copied to personal state
4. Gets structured files with examples
5. Starts logging work immediately
6. References templates for format

### Scenario 2: Template Evolution

1. Developer improves daily log format
2. Uses new format for 2 weeks
3. Verifies improvements
4. Updates DAILY_LOG.md template
5. Sanitizes personal data
6. Commits updated template
7. Other developers benefit from improvement

### Scenario 3: Team Onboarding

1. New team member joins
2. Reviews dev/templates/README.md (this file)
3. Understands template philosophy
4. Runs setup with Fresh Install mode
5. Gets structured workspace
6. Templates guide initial workflow
7. Customizes based on needs

## Cross-References

- **Dev Hub Overview**: `/dev/README_DEV.md` (complete development system)
- **Getting Started**: `/dev/GETTING_STARTED.md` (onboarding guide)
- **Project Rules**: `/dev/rules/CLAUDE.md` (project standards)
- **Setup Skill**: `/.claude/skills/dev-env-setup/SKILL.md` (automated setup)
- **Migration Script**: `/dev/scripts/migrate_dev_environment.sh` (manual setup)

---

**Last Updated**: 2026-01-22
**Maintained By**: Development Team
**Review Frequency**: Quarterly or as needed
