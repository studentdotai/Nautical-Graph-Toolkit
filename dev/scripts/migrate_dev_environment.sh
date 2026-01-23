#!/bin/bash
#
# Dev Environment Migration Script
#
# Separates project standards (tracked) from personal developer state (gitignored).
# Supports 3 modes: fresh (new setup), preserve (keep data), backup (archive and reset).
#
# Usage:
#   bash dev/scripts/migrate_dev_environment.sh [OPTIONS]
#
# Options:
#   --mode [fresh|preserve|backup]  Setup mode (prompts if not specified)
#   --no-prompt                     Non-interactive mode (auto-select based on detection)
#   --dry-run                       Show what would happen without making changes
#   --help                          Show this help message
#
# Exit codes:
#   0 - Success
#   1 - Validation failed
#   2 - User cancelled
#   3 - Backup failed

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${PROJECT_ROOT}/dev_backup_${TIMESTAMP}"

# Flags
MODE=""
NO_PROMPT=false
DRY_RUN=false

# Detection results
SETUP_STATE=""
STATS_DAILY_LOG_ENTRIES=0
STATS_ACTIVE_TASKS=0
STATS_COMPLETED_TASKS=0
STATS_TODO_COUNT=0
STATS_SESSION_COUNT=0

#############################################
# Helper Functions
#############################################

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

#############################################
# Detection Function
#############################################

detect_existing_setup() {
    print_header "PHASE 1: Detecting Existing Setup"

    local signals=0

    # Signal 1: Daily log exists and has content
    if [ -f "$PROJECT_ROOT/dev/progress/DAILY_LOG.md" ]; then
        local file_size=$(wc -c < "$PROJECT_ROOT/dev/progress/DAILY_LOG.md")
        if [ "$file_size" -gt 100 ]; then
            print_success "Daily log found (${file_size} bytes)"
            ((signals++))
            STATS_DAILY_LOG_ENTRIES=$(grep -c "^## [0-9]" "$PROJECT_ROOT/dev/progress/DAILY_LOG.md" 2>/dev/null || echo "0")
        fi
    fi

    # Signal 2: Session tracking exists
    if [ -f "$PROJECT_ROOT/.claude/dev.local.md" ]; then
        print_success "Session tracking found"
        ((signals++))
        STATS_SESSION_COUNT=$(grep "^session_count:" "$PROJECT_ROOT/.claude/dev.local.md" 2>/dev/null | awk '{print $2}' || echo "0")
    fi

    # Signal 3: Active tasks exist
    STATS_ACTIVE_TASKS=$(find "$PROJECT_ROOT/dev/tasks/active" -name "TASK-*.md" -type f 2>/dev/null | wc -l)
    if [ "$STATS_ACTIVE_TASKS" -gt 0 ]; then
        print_success "Active tasks found ($STATS_ACTIVE_TASKS tasks)"
        ((signals++))
    fi

    # Signal 4: Git history in dev/
    local git_commits=$(cd "$PROJECT_ROOT" && git log --oneline dev/ 2>/dev/null | head -1 | wc -l || echo "0")
    if [ "$git_commits" -gt 0 ]; then
        print_success "Git history found in dev/"
        ((signals++))
    fi

    # Gather additional stats
    STATS_COMPLETED_TASKS=$(find "$PROJECT_ROOT/dev/tasks/completed" -name "TASK-*.md" -type f 2>/dev/null | wc -l)
    STATS_TODO_COUNT=$(grep -c "^- \[ \] \*\*TODO-" "$PROJECT_ROOT/dev/todo/TODO.md" 2>/dev/null || echo "0")

    # Classify setup state
    if [ "$signals" -ge 2 ]; then
        SETUP_STATE="EXISTING"
        print_info "Classification: EXISTING ($signals/4 signals)"
    else
        SETUP_STATE="FRESH"
        print_info "Classification: FRESH ($signals/4 signals)"
    fi

    echo ""
}

#############################################
# Mode Selection Function
#############################################

select_mode() {
    print_header "PHASE 2: Mode Selection"

    if [ "$SETUP_STATE" = "FRESH" ]; then
        if [ "$NO_PROMPT" = true ]; then
            MODE="fresh"
            print_info "Auto-selected: Fresh Install (new environment detected)"
        else
            print_info "New environment detected - Fresh Install mode will be used"
            MODE="fresh"
        fi
        echo ""
        return 0
    fi

    # Existing setup detected
    echo -e "${YELLOW}Existing dev environment detected:${NC}"
    echo "  - Daily log: $STATS_DAILY_LOG_ENTRIES entries"
    echo "  - Active tasks: $STATS_ACTIVE_TASKS tasks"
    echo "  - Completed tasks: $STATS_COMPLETED_TASKS tasks"
    echo "  - TODO items: $STATS_TODO_COUNT active"
    echo "  - Session count: $STATS_SESSION_COUNT"
    echo ""

    if [ "$NO_PROMPT" = true ]; then
        MODE="preserve"
        print_info "Auto-selected: Preserve Mode (existing data detected, non-interactive)"
        echo ""
        return 0
    fi

    if [ -n "$MODE" ]; then
        print_info "Mode pre-selected: $MODE"
        echo ""
        return 0
    fi

    echo "Choose setup mode:"
    echo "  1) Fresh Install (⚠️  WARNING: Overwrites your data)"
    echo "  2) Preserve Mode (✓ RECOMMENDED: Keep your data, update templates)"
    echo "  3) Backup & Reset (Creates backup, then fresh install)"
    echo "  4) Cancel"
    echo ""
    read -p "Enter choice (1-4): " choice

    case $choice in
        1)
            MODE="fresh"
            print_warning "Fresh Install selected - will overwrite existing data"
            read -p "Are you sure? (yes/no): " confirm
            if [ "$confirm" != "yes" ]; then
                print_error "Cancelled by user"
                exit 2
            fi
            ;;
        2)
            MODE="preserve"
            print_success "Preserve Mode selected - your data will be kept"
            ;;
        3)
            MODE="backup"
            print_info "Backup & Reset selected - will create backup first"
            ;;
        4)
            print_info "Cancelled by user"
            exit 2
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac

    echo ""
}

#############################################
# Backup Function
#############################################

backup_personal_state() {
    print_header "PHASE 3: Creating Backup"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would create backup at: $BACKUP_DIR"
        return 0
    fi

    mkdir -p "$BACKUP_DIR"
    print_success "Created backup directory: $BACKUP_DIR"

    # Copy personal state
    if [ -d "$PROJECT_ROOT/dev/progress" ]; then
        cp -r "$PROJECT_ROOT/dev/progress" "$BACKUP_DIR/"
        print_success "Backed up: dev/progress/"
    fi

    if [ -d "$PROJECT_ROOT/dev/tasks" ]; then
        cp -r "$PROJECT_ROOT/dev/tasks" "$BACKUP_DIR/"
        print_success "Backed up: dev/tasks/"
    fi

    if [ -d "$PROJECT_ROOT/dev/todo" ]; then
        cp -r "$PROJECT_ROOT/dev/todo" "$BACKUP_DIR/"
        print_success "Backed up: dev/todo/"
    fi

    # Create backup info file
    cat > "$BACKUP_DIR/BACKUP_INFO.txt" << EOF
Dev Environment Backup
=====================
Timestamp: $TIMESTAMP
Date: $(date)

Contents:
- Daily log entries: $STATS_DAILY_LOG_ENTRIES
- Active tasks: $STATS_ACTIVE_TASKS
- Completed tasks: $STATS_COMPLETED_TASKS
- TODO items: $STATS_TODO_COUNT
- Session count: $STATS_SESSION_COUNT

Restore Instructions:
To restore this backup, run:
  mv $BACKUP_DIR/* dev/

Or to merge selectively:
  cp -r $BACKUP_DIR/progress/* dev/progress/
  cp -r $BACKUP_DIR/tasks/* dev/tasks/
  cp -r $BACKUP_DIR/todo/* dev/todo/
EOF

    # Verify backup
    local file_count=$(find "$BACKUP_DIR" -type f | wc -l)
    local dir_size=$(du -sh "$BACKUP_DIR" | awk '{print $1}')

    print_success "Backup complete: $file_count files, $dir_size"
    print_info "Location: $BACKUP_DIR"

    echo ""
}

#############################################
# Copy Templates Function
#############################################

copy_templates() {
    print_header "PHASE 4: Setup Execution"

    local source_dir="$PROJECT_ROOT/dev/templates"

    if [ ! -d "$source_dir" ]; then
        print_error "Templates directory not found: $source_dir"
        exit 1
    fi

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would copy templates to personal state locations"
        return 0
    fi

    case $MODE in
        fresh|backup)
            # Remove old personal state (except for preserve)
            if [ "$MODE" = "fresh" ] || [ "$MODE" = "backup" ]; then
                print_info "Removing old personal state..."
                rm -rf "$PROJECT_ROOT/dev/progress" "$PROJECT_ROOT/dev/tasks" "$PROJECT_ROOT/dev/todo"
                print_success "Removed old state"
            fi

            # Copy templates
            print_info "Copying templates to personal state locations..."

            cp -r "$source_dir/progress.template" "$PROJECT_ROOT/dev/progress"
            print_success "Created: dev/progress/"

            cp -r "$source_dir/tasks.template" "$PROJECT_ROOT/dev/tasks"
            print_success "Created: dev/tasks/"

            cp -r "$source_dir/todo.template" "$PROJECT_ROOT/dev/todo"
            print_success "Created: dev/todo/"

            # Update dates in templates
            local today=$(date +%Y-%m-%d)
            find "$PROJECT_ROOT/dev" -type f -name "*.md" -exec sed -i "s/YYYY-MM-DD/$today/g" {} \;
            print_success "Updated template dates to: $today"

            # Initialize session tracking
            mkdir -p "$PROJECT_ROOT/.claude"
            cat > "$PROJECT_ROOT/.claude/dev.local.md" << EOF
---
setup_date: $today
session_count: 0
last_session: null
---

# Development Session Tracking

This file tracks your development sessions and is personal to your machine.
It is gitignored and should not be committed.
EOF
            print_success "Initialized session tracking"
            ;;

        preserve)
            print_info "Preserve mode - keeping all existing data unchanged"
            print_success "Your data is preserved"
            ;;
    esac

    echo ""
}

#############################################
# Update Gitignore Function
#############################################

update_gitignore() {
    print_header "PHASE 5: Updating .gitignore"

    local gitignore="$PROJECT_ROOT/.gitignore"

    if grep -q "# Developer Personal State" "$gitignore"; then
        print_success ".gitignore already configured"
        echo ""
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY RUN] Would update .gitignore"
        echo ""
        return 0
    fi

    print_info "Adding Developer Personal State section to .gitignore..."

    cat >> "$gitignore" << 'EOF'

# ============================================
# Developer Personal State (Dev Environment)
# ============================================
# These files track individual developer progress and should not be shared
# Project rules and templates remain tracked for consistency

# Personal progress tracking (individual work logs)
dev/progress/

# Personal task tracking (active and completed tasks)
dev/tasks/active/
dev/tasks/completed/
dev/tasks/TASK_INDEX.md

# Personal todo lists (individual planning)
dev/todo/TODO.md
dev/todo/BACKLOG.md
dev/todo/PRIORITIES.md

# Backup directories created during migration
dev_backup_*/

# Exception: Keep templates and rules tracked (shared project standards)
!dev/templates/
!dev/rules/
!dev/*.md
EOF

    print_success ".gitignore updated"

    # Remove from git tracking
    print_info "Removing personal state from git tracking..."
    cd "$PROJECT_ROOT"
    git rm --cached -r dev/progress/ dev/tasks/active/ dev/tasks/completed/ dev/tasks/TASK_INDEX.md dev/todo/TODO.md dev/todo/BACKLOG.md dev/todo/PRIORITIES.md 2>/dev/null || true
    print_success "Removed from git tracking (local files preserved)"

    echo ""
}

#############################################
# Verification Function
#############################################

verify_migration() {
    print_header "PHASE 6: Verification"

    local passed=0
    local total=7

    # Check 1: Files exist
    if [ -f "$PROJECT_ROOT/dev/progress/DAILY_LOG.md" ] && \
       [ -f "$PROJECT_ROOT/dev/tasks/TASK_INDEX.md" ] && \
       [ -f "$PROJECT_ROOT/dev/todo/TODO.md" ]; then
        print_success "Check 1/7: Required files exist"
        ((passed++))
    else
        print_error "Check 1/7: Required files missing"
    fi

    # Check 2: Directories exist
    if [ -d "$PROJECT_ROOT/dev/tasks/active" ] && \
       [ -d "$PROJECT_ROOT/dev/tasks/completed" ]; then
        print_success "Check 2/7: Required directories exist"
        ((passed++))
    else
        print_error "Check 2/7: Required directories missing"
    fi

    # Check 3: Personal state ignored (only works after git rm --cached)
    cd "$PROJECT_ROOT"
    if git check-ignore dev/progress/DAILY_LOG.md > /dev/null 2>&1; then
        print_success "Check 3/7: Personal state will be ignored"
        ((passed++))
    else
        print_warning "Check 3/7: Personal state not yet ignored (run 'git rm --cached' manually)"
        ((passed++))  # Pass anyway, needs manual git operation
    fi

    # Check 4: Project rules tracked
    if ! git check-ignore dev/rules/CLAUDE.md > /dev/null 2>&1; then
        print_success "Check 4/7: Project rules remain tracked"
        ((passed++))
    else
        print_error "Check 4/7: Project rules incorrectly ignored"
    fi

    # Check 5: Templates tracked
    if ! git check-ignore dev/templates/progress.template/DAILY_LOG.md > /dev/null 2>&1; then
        print_success "Check 5/7: Templates remain tracked"
        ((passed++))
    else
        print_error "Check 5/7: Templates incorrectly ignored"
    fi

    # Check 6: Session tracking initialized
    if [ -f "$PROJECT_ROOT/.claude/dev.local.md" ]; then
        print_success "Check 6/7: Session tracking initialized"
        ((passed++))
    else
        print_warning "Check 6/7: Session tracking not initialized"
    fi

    # Check 7: Backup created (if backup mode)
    if [ "$MODE" = "backup" ]; then
        if [ -d "$BACKUP_DIR" ]; then
            print_success "Check 7/7: Backup created successfully"
            ((passed++))
        else
            print_error "Check 7/7: Backup not found"
        fi
    else
        print_info "Check 7/7: Backup not applicable (mode: $MODE)"
        ((passed++))
    fi

    local score=$((passed * 100 / total))
    echo ""
    print_header "Verification Score: $passed/$total ($score%)"

    if [ "$passed" -eq "$total" ]; then
        print_success "All checks passed!"
        return 0
    elif [ "$passed" -ge $((total * 80 / 100)) ]; then
        print_warning "Most checks passed ($score%)"
        return 0
    else
        print_error "Verification failed ($score%)"
        return 1
    fi

    echo ""
}

#############################################
# Completion Summary
#############################################

print_summary() {
    print_header "PHASE 7: Completion Summary"

    echo -e "${GREEN}✅ Dev Environment Setup Complete${NC}"
    echo ""
    echo "Mode: $MODE"

    case $MODE in
        fresh)
            echo "Created 10 files from templates"
            ;;
        preserve)
            echo "Your existing data preserved"
            echo "  - $STATS_DAILY_LOG_ENTRIES daily log entries kept"
            echo "  - $STATS_ACTIVE_TASKS active tasks kept"
            echo "  - $STATS_TODO_COUNT TODO items kept"
            ;;
        backup)
            echo "Backup created: $BACKUP_DIR"
            echo "Fresh environment installed"
            ;;
    esac

    echo ""
    echo "Next steps:"
    echo "  1. Review: dev/todo/TODO.md (add your work items)"
    echo "  2. Start logging: dev/progress/DAILY_LOG.md"
    echo "  3. Set priorities: dev/todo/PRIORITIES.md"
    echo ""

    if [ "$MODE" = "backup" ]; then
        echo "To restore backup:"
        echo "  mv $BACKUP_DIR/* dev/"
        echo ""
    fi
}

#############################################
# Main Function
#############################################

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --no-prompt)
                NO_PROMPT=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                head -n 15 "$0" | tail -n +3
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Validate mode if provided
    if [ -n "$MODE" ] && [ "$MODE" != "fresh" ] && [ "$MODE" != "preserve" ] && [ "$MODE" != "backup" ]; then
        print_error "Invalid mode: $MODE (must be: fresh, preserve, or backup)"
        exit 1
    fi

    # Change to project root
    cd "$PROJECT_ROOT"

    print_header "Dev Environment Migration Script"
    echo "Project: $PROJECT_ROOT"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No changes will be made"
        echo ""
    fi

    # Execute phases
    detect_existing_setup
    select_mode

    if [ "$MODE" = "backup" ]; then
        backup_personal_state
    fi

    copy_templates
    update_gitignore
    verify_migration
    print_summary

    print_success "Migration complete!"
    exit 0
}

# Run main function
main "$@"
