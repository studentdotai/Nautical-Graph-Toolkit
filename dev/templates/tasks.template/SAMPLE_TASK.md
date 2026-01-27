# TASK-XXX: [Task Title Here]

Brief one-sentence description of what this task accomplishes.

## Metadata

- **ID**: TASK-XXX
- **Priority**: P0/P1/P2/P3 (Critical/High/Medium/Low)
- **Status**: Planned / Active / Blocked / Completed / Cancelled
- **Created**: YYYY-MM-DD
- **Completed**: YYYY-MM-DD (or "in progress" or "not yet")
- **Owner**: [Your name or team name]
- **Estimated Effort**: S (<2h) / M (2-5h) / L (5-10h) / XL (>10h)
- **Actual Effort**: ~Xh (update as you work)
- **Progress**: X% (estimate of completion, e.g., 0%, 25%, 50%, 75%, 100%)
- **Dependencies**: TASK-YYY, TASK-ZZZ (or "None")
- **Related TODO**: TODO-ABC (which TODO item triggered this task)

## Description

Detailed description of what needs to be done and why. Include:
- Problem or requirement being addressed
- Context and background information
- Why this work is important
- Expected outcomes or deliverables

**Example**:
> This task implements user authentication using JWT tokens with role-based access control. Currently, the API has no authentication, making it vulnerable to unauthorized access. This work will secure all endpoints and enable multi-tenant functionality planned for v2.0.

**Scope**: Clearly define what is included and what is NOT included in this task.

**In Scope**:
- JWT token generation and validation
- User login/logout endpoints
- Password hashing with bcrypt
- Basic role-based access control (admin, user, guest)
- Session management

**Out of Scope**:
- OAuth integration (separate task)
- Password reset functionality (separate task)
- Two-factor authentication (future version)
- User profile management (separate task)

## Acceptance Criteria

Clear, testable criteria that define when this task is complete. Use checkboxes for tracking.

- [ ] Criterion 1: [Specific, measurable, testable requirement]
- [ ] Criterion 2: [What must work correctly]
- [ ] Criterion 3: [What quality standards must be met]
- [ ] Criterion 4: [What documentation must be created]
- [ ] Criterion 5: [What tests must pass]

**Example Acceptance Criteria**:
- [ ] Users can log in with username/password and receive JWT token
- [ ] All API endpoints require valid JWT token
- [ ] Token expiration handled correctly (401 response after expiry)
- [ ] RBAC implemented: admin can access all endpoints, users limited to own data
- [ ] Password hashing uses bcrypt with minimum 12 rounds
- [ ] Integration tests cover all authentication flows
- [ ] API documentation updated with auth requirements
- [ ] Security audit passes with no critical issues

## Implementation Plan

Step-by-step plan for completing the task. Update as you progress.

### Phase 1: [Phase Name, e.g., "Setup and Research"]
- [x] Step 1: [What you'll do first] (completed YYYY-MM-DD)
- [x] Step 2: [Next action item] (completed YYYY-MM-DD)
- [ ] Step 3: [Not yet done]
- [ ] Step 4: [Not yet done]

### Phase 2: [Phase Name, e.g., "Core Implementation"]
- [ ] Step 5: [Implementation detail]
- [ ] Step 6: [Another implementation step]
- [ ] Step 7: [Testing or validation]

### Phase 3: [Phase Name, e.g., "Testing and Documentation"]
- [ ] Step 8: [Testing activities]
- [ ] Step 9: [Documentation updates]
- [ ] Step 10: [Final verification]

**Example Implementation Plan**:

### Phase 1: Setup and Dependencies
- [x] Research JWT libraries (PyJWT vs python-jose) (completed 2026-01-10)
- [x] Install and configure PyJWT (completed 2026-01-10)
- [x] Create authentication module structure (completed 2026-01-11)
- [ ] Set up test fixtures for auth testing

### Phase 2: Core Authentication
- [ ] Implement JWT token generation function
- [ ] Implement JWT token validation middleware
- [ ] Create user login endpoint (POST /api/auth/login)
- [ ] Create user logout endpoint (POST /api/auth/logout)
- [ ] Implement password hashing with bcrypt
- [ ] Add token refresh mechanism

### Phase 3: RBAC Implementation
- [ ] Design role hierarchy (admin, user, guest)
- [ ] Implement permission decorators
- [ ] Apply permissions to all endpoints
- [ ] Create admin-only endpoints

### Phase 4: Testing and Documentation
- [ ] Write unit tests for token generation/validation
- [ ] Write integration tests for login/logout flow
- [ ] Write RBAC permission tests
- [ ] Update OpenAPI documentation
- [ ] Create security documentation
- [ ] Perform security review

## Progress Notes

Chronological notes tracking progress, decisions, and issues. Add new entries at the top.

### YYYY-MM-DD: [Brief update title]

**What was done**:
- [Accomplishment 1]
- [Accomplishment 2]
- [Accomplishment 3]

**Decisions made**:
- [Decision 1 with rationale]
- [Decision 2 with alternatives considered]

**Issues encountered**:
- [Issue 1 and how it was resolved]
- [Issue 2 and current status]

**Next steps**:
- [Next action item]
- [Upcoming work]

---

### Example Progress Note

### 2026-01-11: Core JWT implementation complete

**What was done**:
- Implemented JWT token generation with configurable expiration
- Created validation middleware with error handling
- Added token blacklist for logout functionality
- Set up Redis for token storage

**Decisions made**:
- Chose PyJWT over python-jose for better documentation and community support
- Token expiration set to 1 hour (balance between security and UX)
- Refresh tokens expire after 30 days
- Used Redis for blacklist (faster than database queries)

**Issues encountered**:
- Token validation failing on first request after login (RESOLVED: clock skew issue, added 60s leeway)
- Redis connection pooling causing intermittent failures (RESOLVED: increased pool size to 10)

**Next steps**:
- Start RBAC implementation Phase 3
- Write integration tests for login/logout flow
- Document authentication flow in OpenAPI spec

---

## Skills Used

List any skills, tools, or techniques used during this task. Helps with knowledge transfer and future planning.

- [Skill/Tool 1]: [How it was used]
- [Skill/Tool 2]: [What it helped accomplish]
- [Technique 1]: [Why it was chosen]

**Example**:
- JWT (JSON Web Tokens): Token-based authentication
- bcrypt: Secure password hashing
- Redis: Fast token blacklist storage
- PyJWT library: Token generation and validation
- Integration testing: End-to-end flow verification

## Outcomes

### Expected Outcomes (defined at task creation)

What you expect to achieve:
- [Expected outcome 1]
- [Expected outcome 2]
- [Expected outcome 3]

**Example**:
- Secure API endpoints with JWT authentication
- Role-based access control operational
- <100ms authentication overhead per request
- Zero critical security vulnerabilities

### Actual Outcomes (filled on completion)

What was actually achieved:
- [Actual outcome 1 with metrics]
- [Actual outcome 2 with measurements]
- [Actual outcome 3 with comparisons to expected]
- [Unexpected benefits or limitations]

**Example**:
- ✅ All API endpoints secured with JWT authentication
- ✅ RBAC implemented with 3 roles (admin, user, guest)
- ✅ Authentication overhead: 45ms average (better than 100ms target)
- ✅ Security audit passed with zero critical issues
- ⚠️ Token refresh requires additional database query (5ms overhead)
- ➕ Bonus: Added API key authentication for service accounts

## Lessons Learned

Insights gained during this task. What went well, what didn't, what would you do differently?

### What Went Well
- [Success 1]
- [Success 2]

### What Didn't Go Well
- [Challenge 1 and how it was handled]
- [Challenge 2 and lessons learned]

### What Would You Do Differently
- [Improvement 1 for next time]
- [Process change 2]

### Knowledge Gaps Identified
- [Skill or knowledge area to improve]
- [Tool or technique to learn more about]

**Example Lessons Learned**:

### What Went Well
- PyJWT library worked excellently, clear API and good error messages
- Redis integration simplified token blacklist implementation
- Integration tests caught edge cases early (clock skew, concurrent logins)

### What Didn't Go Well
- Initial token expiration set too low (5 minutes), caused UX issues in testing
- Underestimated RBAC complexity, took 2h longer than estimated
- Should have set up test fixtures earlier, spent time creating test data

### What Would You Do Differently
- Start with longer token expiration (1 hour) and adjust based on security requirements
- Break RBAC into separate task (too much scope for one task)
- Create comprehensive test fixtures before implementation starts
- Add more detailed security documentation from the beginning

### Knowledge Gaps Identified
- Need to learn more about token rotation strategies
- Should understand OAuth 2.0 flow better for future integration
- Need practice with security testing tools (OWASP ZAP, Burp Suite)

---

## Related Items

- **TODO**: TODO-XXX (which TODO item triggered this task)
- **Priority**: P0/P1/P2/P3 (from priority framework)
- **Milestone**: MS-XXX (if part of milestone)
- **Dependencies**: TASK-YYY, TASK-ZZZ (tasks that must complete first)
- **Blocks**: TASK-AAA (tasks waiting for this one)

**Example**:
- **TODO**: TODO-015 (Implement authentication)
- **Priority**: P1 (High)
- **Milestone**: MS-003 (Security improvements)
- **Dependencies**: TASK-004 (API docs must be complete first)
- **Blocks**: TASK-006 (Password reset), TASK-007 (Email verification)

## Cross-References

- **Project Knowledge**: `/dev/rules/CLAUDE.md` (project architecture and patterns)
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md` (coding conventions)
- **Task Index**: `/dev/tasks/TASK_INDEX.md` (master task registry)
- **TODO List**: `/dev/todo/TODO.md` (source of task requirements)
- **Daily Log**: `/dev/progress/DAILY_LOG.md` (detailed daily updates)
- **Workflow Guide**: `/dev/rules/WORKFLOW.md` (development processes)

---

## Task File Template Usage

**When to use this template**:
- Creating new task from high-priority TODO
- Promoting backlog item to active work
- Starting multi-step feature or bug fix

**How to use this template**:
1. Copy this file to `dev/tasks/active/TASK-XXX-your-task-name.md`
2. Replace XXX with next available task number (check TASK_INDEX.md)
3. Fill in all metadata fields
4. Write detailed description and scope
5. Define acceptance criteria (specific and testable)
6. Create implementation plan (break into phases)
7. Update as you work (add progress notes)
8. Mark complete when all criteria met
9. Document outcomes and lessons learned
10. Move to `dev/tasks/completed/` when done

**Tips for effective task files**:
- Be specific and measurable in acceptance criteria
- Break large tasks into smaller phases
- Update progress notes regularly (daily or after significant work)
- Document decisions and rationale
- Link to related files, docs, and other tasks
- Keep outcomes realistic and honest (not everything goes as planned)
- Extract lessons learned for future improvements
