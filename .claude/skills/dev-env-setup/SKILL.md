# Dev Environment Setup

Initialize /dev environment for new developers. Creates personal state files from templates, sets up .gitignore, and verifies setup. Supports 3 modes (Fresh/Preserve/Backup). Activates on keywords 'setup dev environment', 'initialize dev', 'reset dev environment'.

---
name: dev-env-setup
description: Initialize /dev environment for new developers. Creates personal state files from templates, sets up .gitignore, and verifies setup. Supports 3 modes (Fresh/Preserve/Backup). Activates on keywords 'setup dev environment', 'initialize dev', 'reset dev environment'.
allowed-tools: [Bash, Read, Write, Edit, AskUserQuestion]
---

# Dev Environment Setup Controller

Interactive controller for initializing developer environment with personal state files. This skill detects existing setups, guides through mode selection, creates file structure from templates, and verifies configuration.

⚠️ **Active Controller**: This skill executes file operations and system configuration. It will detect your current state, ask for mode selection, and execute setup with verification.

## Quick Start

### First-time setup (new developer)
```bash
# Automatically detects fresh install and creates all files
/dev:setup
```

### Update templates only (preserve your data)
```bash
# Detected existing setup, offers mode selection
/dev:setup
# Choose: "Preserve Mode" to keep your data
```

### Clean slate with backup
```bash
# Creates backup then fresh install
/dev:setup
# Choose: "Backup & Reset" to save current state
```

## Understanding Setup Modes

### Mode 1: Fresh Install (New Developers)

**When to use**: First-time setup on new machine or fresh clone

**What it does**:
- Copies all 10 template files → personal state locations
- Initializes session tracking
- Updates .gitignore if needed
- Creates directory structure

**Use cases**:
- New developer onboarding
- Fresh project clone
- CI/CD environment setup
- Clean development environment

**Files created**:
```
dev/
├── progress/
│   ├── DAILY_LOG.md (from template)
│   ├── CHANGELOG.md (from template)
│   └── MILESTONES.md (from template)
├── tasks/
│   ├── TASK_INDEX.md (from template)
│   ├── active/ (empty)
│   └── completed/ (empty)
└── todo/
    ├── TODO.md (from template)
    ├── BACKLOG.md (from template)
    └── PRIORITIES.md (from template)
```

### Mode 2: Preserve Mode (Existing Developers)

**When to use**: You have existing work and want to keep it

**What it does**:
- Keeps ALL your personal state files unchanged
- Only updates templates/ directory (for future use)
- Verifies .gitignore is correct
- No data loss

**Use cases**:
- Updating template files after project changes
- Verifying gitignore configuration
- Running setup script after updates
- Testing without risk

**What's preserved**:
- All daily log entries
- All tasks (active and completed)
- All TODO items and backlog
- All progress tracking
- Your existing work remains 100% intact

### Mode 3: Backup & Reset (Clean Slate)

**When to use**: Want fresh start but keep backup of current work

**What it does**:
- Creates timestamped backup: `dev_backup_YYYYMMDD_HHMMSS/`
- Copies ALL personal state to backup
- Removes old personal state
- Copies fresh templates to personal locations
- Shows backup location

**Use cases**:
- Starting new project phase
- Resetting after major changes
- Cleaning up old entries
- Archiving previous work

**Backup structure**:
```
dev_backup_20260122_143052/
├── progress/
│   ├── DAILY_LOG.md (your 20+ entries)
│   ├── CHANGELOG.md (your changes)
│   └── MILESTONES.md (your milestones)
├── tasks/
│   ├── TASK_INDEX.md (your tasks)
│   ├── active/ (your active tasks)
│   └── completed/ (your completed tasks)
├── todo/
│   ├── TODO.md (your 9 TODOs)
│   ├── BACKLOG.md (your backlog)
│   └── PRIORITIES.md (your priorities)
└── BACKUP_INFO.txt (metadata)
```

## Mode Comparison Table

| Feature | Fresh Install | Preserve | Backup & Reset |
|---------|:-------------:|:--------:|:--------------:|
| For new developers | ✓ | | |
| Keeps existing work | | ✓ | |
| Creates backup | | | ✓ |
| Updates templates | ✓ | ✓ | ✓ |
| Modifies personal state | ✓ | | ✓ |
| Risk of data loss | None | None | None (backed up) |
| Speed | Fast (~5s) | Very fast (~2s) | Medium (~10s) |
| Use when | Fresh clone | Have data | Want clean slate |

## Execution Procedure

When you invoke this skill, it follows a 7-phase process:

### Phase 1: Detection
1. Check for daily log existence and size (>100 bytes)
2. Check for session tracking file (.claude/dev.local.md)
3. Count active task files (*.md in dev/tasks/active/)
4. Check git history (git log dev/)
5. **Classify**: FRESH (0-1 signals) or EXISTING (2+ signals)
6. **Gather stats** if existing:
   - Count daily log entries
   - Count active/completed tasks
   - Count TODO items
   - Check session count

### Phase 2: Mode Selection
1. **If FRESH**: Auto-select Fresh Install mode
2. **If EXISTING**: Display detection results and prompt:
   ```
   Existing dev environment detected:
   - Daily log: 20 entries (1220 lines)
   - Active tasks: 2 tasks
   - Completed tasks: 2 tasks
   - TODO items: 9 active
   - Session count: 12

   Choose setup mode:
   1. Fresh Install (⚠️ WARNING: Overwrites your data)
   2. Preserve Mode (✓ RECOMMENDED: Keep your data, update templates)
   3. Backup & Reset (Creates backup, then fresh install)
   4. Cancel
   ```
3. Validate user choice

### Phase 3: Backup (if Backup & Reset mode)
1. Create backup directory: `dev_backup_YYYYMMDD_HHMMSS/`
2. Copy personal state:
   - `dev/progress/` → `dev_backup_*/progress/`
   - `dev/tasks/` → `dev_backup_*/tasks/`
   - `dev/todo/` → `dev_backup_*/todo/`
3. Create BACKUP_INFO.txt with metadata
4. Verify backup (file count, total size)
5. Display backup location to user

### Phase 4: Setup Execution

**Fresh Install or Backup & Reset**:
1. Remove old personal state (if reset)
2. Copy templates → personal locations:
   ```bash
   cp -r dev/templates/progress.template/* dev/progress/
   cp -r dev/templates/tasks.template/* dev/tasks/
   cp -r dev/templates/todo.template/* dev/todo/
   ```
3. Update template dates to current date
4. Initialize session tracking (.claude/dev.local.md):
   ```yaml
   ---
   setup_date: YYYY-MM-DD
   session_count: 0
   last_session: null
   ---
   ```

**Preserve Mode**:
1. Display: "Keeping your data unchanged. Templates already in place."
2. Skip file operations
3. Proceed to verification

### Phase 5: .gitignore Update
1. Check if section exists: `grep "# Developer Personal State" .gitignore`
2. **If missing**: Append new section after line 63:
   ```
   # ============================================
   # Developer Personal State (Dev Environment)
   # ============================================
   # Personal progress tracking
   dev/progress/
   # Personal task tracking
   dev/tasks/active/
   dev/tasks/completed/
   dev/tasks/TASK_INDEX.md
   # Personal todo lists
   dev/todo/TODO.md
   dev/todo/BACKLOG.md
   dev/todo/PRIORITIES.md
   # Backup directories
   dev_backup_*/
   # Keep templates and rules tracked
   !dev/templates/
   !dev/rules/
   !dev/*.md
   ```
3. **If exists**: Display "✓ .gitignore already configured"

### Phase 6: Verification
Run 7 automated checks:
1. **Files exist**: dev/progress/DAILY_LOG.md, dev/tasks/TASK_INDEX.md, dev/todo/TODO.md
2. **Personal state ignored**: `git check-ignore dev/progress/DAILY_LOG.md` outputs path
3. **Project rules tracked**: `git check-ignore dev/rules/CLAUDE.md` outputs nothing
4. **Templates tracked**: `git check-ignore dev/templates/progress.template/DAILY_LOG.md` outputs nothing
5. **Session tracking initialized**: .claude/dev.local.md exists
6. **Directory structure**: active/, completed/ directories exist
7. **Backup created** (if Backup & Reset): dev_backup_* directory exists

Calculate score: (passed / 7) × 100

### Phase 7: Completion Summary
Display results:
```
✅ Dev Environment Setup Complete

Mode: [Fresh Install / Preserve / Backup & Reset]
Verification: 7/7 checks passed (100%)

Files created/verified:
✓ dev/progress/DAILY_LOG.md
✓ dev/progress/CHANGELOG.md
✓ dev/progress/MILESTONES.md
✓ dev/tasks/TASK_INDEX.md
✓ dev/tasks/active/ (empty)
✓ dev/tasks/completed/ (empty)
✓ dev/todo/TODO.md
✓ dev/todo/BACKLOG.md
✓ dev/todo/PRIORITIES.md
✓ .gitignore configured
✓ Session tracking initialized

[If backup created]
Backup location: dev_backup_20260122_143052/
(15 files, 2.1 MB)

Next steps:
1. Start logging work: dev/progress/DAILY_LOG.md
2. Add your first TODO: dev/todo/TODO.md
3. Check priorities: dev/todo/PRIORITIES.md
4. Available commands: /dev:session-start, /dev:new-todo, /dev:new-task
```

## Detection Logic Details

### Signal Detection

**Signal 1: Daily Log** (25 points)
```bash
if [ -f "dev/progress/DAILY_LOG.md" ] && [ $(wc -c < "dev/progress/DAILY_LOG.md") -gt 100 ]; then
  echo "yes"
else
  echo "no"
fi
```

**Signal 2: Session Tracking** (25 points)
```bash
[ -f ".claude/dev.local.md" ] && echo "yes" || echo "no"
```

**Signal 3: Active Tasks** (25 points)
```bash
active_task_count=$(find dev/tasks/active -name "*.md" -type f 2>/dev/null | wc -l)
[ "$active_task_count" -gt 0 ] && echo "yes" || echo "no"
```

**Signal 4: Git History** (25 points)
```bash
git_commits=$(git log --oneline dev/ 2>/dev/null | head -1 | wc -l)
[ "$git_commits" -gt 0 ] && echo "yes" || echo "no"
```

**Classification**:
- **FRESH**: 0-1 signals present (0-25 points)
- **EXISTING**: 2+ signals present (50-100 points)

### Stats Gathering (for EXISTING)

```bash
# Count daily log entries (## YYYY-MM-DD pattern)
entry_count=$(grep -c "^## [0-9]" dev/progress/DAILY_LOG.md)

# Count active tasks
active_tasks=$(find dev/tasks/active -name "TASK-*.md" -type f | wc -l)

# Count completed tasks
completed_tasks=$(find dev/tasks/completed -name "TASK-*.md" -type f | wc -l)

# Count TODO items (- [ ] **TODO-XXX** pattern)
todo_count=$(grep -c "- \[ \] \*\*TODO-" dev/todo/TODO.md)

# Get session count from .claude/dev.local.md
session_count=$(grep "^session_count:" .claude/dev.local.md | awk '{print $2}')
```

## Examples

### Example 1: New Developer (Fresh Install)

**User says**: "setup dev environment"

**Detection**:
- Daily log: not found
- Session tracking: not found
- Active tasks: 0
- Git history: no commits
- **Result**: FRESH (0 signals)

**Execution**:
1. Auto-select Fresh Install mode
2. Create all directories
3. Copy 10 template files
4. Initialize session tracking
5. Update .gitignore
6. Verify: 7/7 passed

**Output**:
```
✅ Dev Environment Setup Complete (Fresh Install)

Created 10 files from templates:
✓ 3 progress tracking files
✓ 2 task management files + 2 directories
✓ 3 TODO planning files
✓ .gitignore updated
✓ Session tracking initialized

Verification: 7/7 checks passed (100%)

You're ready to start! Try these commands:
- /dev:session-start - Begin work session
- /dev:new-todo - Add your first TODO item
```

**Duration**: ~5 seconds

---

### Example 2: Existing Developer (Preserve Mode)

**User says**: "run dev setup"

**Detection**:
- Daily log: 1220 lines, 20 entries ✓
- Session tracking: exists, session_count=12 ✓
- Active tasks: 2 files ✓
- Git history: 5 commits ✓
- **Result**: EXISTING (4/4 signals)

**Prompt**:
```
Existing dev environment detected:
- Daily log: 20 entries (1220 lines)
- Active tasks: 2 tasks
- Completed tasks: 2 tasks
- TODO items: 9 active
- Session count: 12

Choose setup mode:
1. Fresh Install (⚠️ WARNING: Overwrites your data)
2. Preserve Mode (✓ RECOMMENDED: Keep your data)
3. Backup & Reset (Creates backup first)
4. Cancel
```

**User selects**: 2 (Preserve Mode)

**Execution**:
1. Skip file operations (preserve all data)
2. Verify templates exist
3. Check .gitignore (already configured)
4. Run verification checks
5. All checks passed

**Output**:
```
✅ Dev Environment Setup Complete (Preserve Mode)

Your data preserved:
✓ 20 daily log entries kept
✓ 2 active tasks kept
✓ 2 completed tasks kept
✓ 9 TODO items kept
✓ All progress tracking kept

Verification: 7/7 checks passed (100%)

No changes made to your work. Templates updated for future use.
```

**Duration**: ~2 seconds

---

### Example 3: Clean Slate (Backup & Reset)

**User says**: "reset dev environment with backup"

**Detection**: EXISTING (same as Example 2)

**User selects**: 3 (Backup & Reset)

**Execution**:
1. Create backup: `dev_backup_20260122_143052/`
2. Copy all personal state to backup
3. Verify backup (15 files, 2.1 MB)
4. Remove old personal state
5. Copy fresh templates
6. Initialize session tracking
7. Verify setup

**Output**:
```
✅ Dev Environment Setup Complete (Backup & Reset)

Backup created:
Location: dev_backup_20260122_143052/
Files: 15 (progress: 3, tasks: 4, todo: 3, other: 5)
Size: 2.1 MB
Contents:
  - 20 daily log entries
  - 4 tasks (2 active, 2 completed)
  - 9 TODO items
  - 12 session records

Fresh environment created:
✓ 10 template files copied
✓ Session tracking reset
✓ .gitignore updated
✓ All directories created

Verification: 7/7 checks passed (100%)

To restore backup: mv dev_backup_20260122_143052/* dev/
```

**Duration**: ~10 seconds

---

### Example 4: CI/CD (Non-Interactive)

**Environment**: GitHub Actions workflow

**Command**:
```bash
# In CI/CD, templates must exist in repo
# This runs silently without prompts
```

**Execution**:
1. Auto-detect FRESH (CI environment)
2. Run Fresh Install mode automatically
3. Skip AskUserQuestion (non-interactive)
4. Verify setup
5. Exit with code 0 (success) or 1 (failure)

**Output** (log):
```
[INFO] Dev environment setup: FRESH detected
[INFO] Running Fresh Install mode
[INFO] Created 10 files from templates
[INFO] Initialized session tracking
[INFO] Verification: 7/7 passed
[INFO] Setup complete: SUCCESS
```

## Common Issues & Solutions

### Issue 1: "Templates directory not found"

**Symptom**: Error during setup: `dev/templates/progress.template/ not found`

**Cause**: Template files not created yet or deleted

**Solution**:
```bash
# Verify templates exist
ls -la dev/templates/

# If missing, they should be in git
git status dev/templates/

# Pull latest changes
git pull origin main

# If still missing, templates need to be created first
# (This skill expects templates to exist)
```

**Prevention**: Templates are tracked in git and should always exist

---

### Issue 2: "Backup directory already exists"

**Symptom**: Backup fails with "dev_backup_YYYYMMDD_HHMMSS/ exists"

**Cause**: Running Backup & Reset twice in same second (unlikely) or old backup not cleaned

**Solution**:
```bash
# Rename existing backup
mv dev_backup_20260122_143052 dev_backup_20260122_143052_old

# Or remove if no longer needed
rm -rf dev_backup_20260122_143052

# Run setup again
```

---

### Issue 3: ".gitignore update failed"

**Symptom**: Verification fails on gitignore check

**Cause**: No write permissions or .gitignore locked

**Solution**:
```bash
# Check permissions
ls -la .gitignore

# Check if file is writable
[ -w ".gitignore" ] && echo "writable" || echo "not writable"

# If not writable, fix permissions
chmod u+w .gitignore

# Manual fix: Add section after line 63
# (See Phase 5 for content)
```

---

### Issue 4: "Personal state files modified in git"

**Symptom**: `git status` shows modified files in dev/progress/, dev/tasks/, dev/todo/

**Cause**: .gitignore not working correctly

**Solution**:
```bash
# Remove from git tracking (keep local files)
git rm --cached -r dev/progress/
git rm --cached -r dev/tasks/active/ dev/tasks/completed/ dev/tasks/TASK_INDEX.md
git rm --cached -r dev/todo/TODO.md dev/todo/BACKLOG.md dev/todo/PRIORITIES.md

# Verify .gitignore section exists
grep "# Developer Personal State" .gitignore

# If missing, add it (see Phase 5)

# Commit the removal
git commit -m "Remove personal dev state from tracking"

# Verify fixed
git status dev/
# Should show only: rules/, templates/, and *.md files in dev/ root
```

---

### Issue 5: "Session tracking file corrupted"

**Symptom**: Session commands fail or behave incorrectly

**Cause**: .claude/dev.local.md has invalid YAML

**Solution**:
```bash
# Check current content
cat .claude/dev.local.md

# Reset to fresh state
cat > .claude/dev.local.md << 'EOF'
---
setup_date: $(date +%Y-%m-%d)
session_count: 0
last_session: null
---
EOF

# Verify fixed
cat .claude/dev.local.md
```

---

### Issue 6: "Template files are outdated"

**Symptom**: Templates don't match current project patterns

**Cause**: Project evolved but templates not updated

**Solution**:
```bash
# Update templates from current best examples
# (Requires manual review and updates)

# 1. Review current working files
cat dev/progress/DAILY_LOG.md  # Your current format

# 2. Update template to match
# Edit: dev/templates/progress.template/DAILY_LOG.md

# 3. Commit updated templates
git add dev/templates/
git commit -m "Update templates to match current practices"

# 4. Run Preserve mode to update templates without losing data
/dev:setup
# Choose: Preserve Mode
```

## Verification Checklist

After setup, these checks should ALL pass:

### Files Exist
```bash
[ -f "dev/progress/DAILY_LOG.md" ] && echo "✓" || echo "✗"
[ -f "dev/progress/CHANGELOG.md" ] && echo "✓" || echo "✗"
[ -f "dev/progress/MILESTONES.md" ] && echo "✓" || echo "✗"
[ -f "dev/tasks/TASK_INDEX.md" ] && echo "✓" || echo "✗"
[ -d "dev/tasks/active" ] && echo "✓" || echo "✗"
[ -d "dev/tasks/completed" ] && echo "✓" || echo "✗"
[ -f "dev/todo/TODO.md" ] && echo "✓" || echo "✗"
[ -f "dev/todo/BACKLOG.md" ] && echo "✓" || echo "✗"
[ -f "dev/todo/PRIORITIES.md" ] && echo "✓" || echo "✗"
[ -f ".claude/dev.local.md" ] && echo "✓" || echo "✗"
```

### Git Ignore Working
```bash
# Should output path (ignored)
git check-ignore dev/progress/DAILY_LOG.md

# Should output nothing (tracked)
git check-ignore dev/rules/CLAUDE.md
git check-ignore dev/templates/progress.template/DAILY_LOG.md

# Should show only rules/ and templates/
git status dev/ --short
```

### Session Tracking Initialized
```bash
grep -q "setup_date:" .claude/dev.local.md && echo "✓" || echo "✗"
grep -q "session_count:" .claude/dev.local.md && echo "✓" || echo "✗"
```

### Expected Git Status
```bash
# Should show (example):
# ?? dev/scripts/               (if exists)
# M  dev/README_DEV.md          (if modified)
# ?? .claude/dev.local.md       (personal)

# Should NOT show:
# M  dev/progress/DAILY_LOG.md  (personal - ignored)
# M  dev/tasks/TASK_INDEX.md    (personal - ignored)
# M  dev/todo/TODO.md           (personal - ignored)
```

## Related Skills

- **dev:session-start** - Start new work session with TODO review
- **dev:session-end** - End session with progress summary
- **dev:new-todo** - Add new TODO item
- **dev:new-task** - Create task from TODO
- **dev:complete-task** - Mark task complete and archive
- **dev:update-task** - Update task progress
- **dev:log-progress** - Add entry to daily log
- **notebook-convert** - Sync notebook changes (uses dev/ for tracking)

## Cross-References

- **Project Rules**: `/dev/rules/CLAUDE.md` (project knowledge and architecture)
- **Agent Guidelines**: `/dev/rules/AGENTS.md` (agent behavior patterns)
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md` (coding conventions)
- **Workflow Guide**: `/dev/rules/WORKFLOW.md` (development processes)
- **Templates**: `/dev/templates/README.md` (template philosophy and usage)
- **Getting Started**: `/dev/GETTING_STARTED.md` (onboarding guide)
- **Dev Hub Overview**: `/dev/README_DEV.md` (complete dev system documentation)

## Notes

**This skill is designed for**:
- First-time developer setup (Fresh Install)
- Preserving existing work (Preserve Mode)
- Clean slate with safety net (Backup & Reset)
- CI/CD automated setup (Fresh Install, non-interactive)

**This skill does NOT**:
- Modify project rules or templates (only personal state)
- Change git configuration or commit anything
- Install dependencies or system packages
- Modify code or documentation files

**Safety features**:
- Non-destructive Preserve mode (recommended default)
- Automatic backup before reset
- Verification checks after setup
- Clear prompts before data changes
- Rollback instructions provided

**Performance**:
- Fresh Install: ~5 seconds
- Preserve Mode: ~2 seconds
- Backup & Reset: ~10 seconds
- Verification: ~1 second
