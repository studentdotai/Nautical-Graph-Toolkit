---
name: context7-usage
description: Leveraging the docs7-agent for real-time, up-to-date library documentation. Use when working with GDAL, GeoPandas, SQLAlchemy, Pydantic, or other external libraries to get current API documentation.
allowed-tools: [Task]
---

# Using docs7-agent for Library Documentation

Leveraging the docs7-agent for real-time, up-to-date library documentation when working with GDAL, GeoPandas, SQLAlchemy, and other dependencies.

**Note**: The docs7-agent is defined at `.claude/agents/docs7-agent.md`. This skill explains when and how to use it.

## Purpose

Claude Code's training data can be outdated, especially for rapidly evolving libraries like GDAL. The docs7-agent provides real-time access to current library documentation, ensuring accurate API usage and avoiding deprecated patterns.

## Prerequisites

- Claude Code CLI with docs7-agent available
- Understanding of which libraries need up-to-date docs
- Familiarity with Task tool for launching agents

## When to Use docs7-agent

### Always Use For:

- **GDAL** (APIs change frequently between versions, pinned at 3.10.3)
- **GeoPandas** (frequent API updates, CRS handling changes)
- **SQLAlchemy** (v2.0+ major changes from v1.x)
- **GeoAlchemy2** (PostGIS integration specifics)
- **Pydantic** (v2.x validation patterns)
- **NetworkX** (graph algorithm parameters)
- **H3** (hexagonal grid operations)

### Optional For:

- **Pandas** (mostly stable, but new features)
- **Plotly** (visualization syntax)
- **BeautifulSoup4** (web scraping patterns)

### Proactive Usage

Claude Code should **automatically** launch docs7-agent when:
- Implementing new features using external libraries
- Debugging library-specific errors
- Working with API methods not seen in recent context
- User asks about library capabilities or syntax

## Procedure

### Step 1: Identify Need for Documentation

Claude recognizes when library documentation is needed:
- User explicitly requests it
- Working with unfamiliar library APIs
- Implementing new features with external libraries
- Debugging library-specific issues

### Step 2: Launch docs7-agent

Claude invokes the Task tool with docs7-agent:

```
Task tool:
  subagent_type: "docs7-agent"
  description: "Fetch GDAL VectorTranslate documentation"
  prompt: "I need documentation for GDAL's VectorTranslate function for converting S-57 files to GeoPackage format"
```

### Step 3: Agent Retrieves Documentation

The docs7-agent autonomously:
- Resolves library name to Context7 ID
- Fetches current documentation
- Returns relevant information focused on the query

### Step 4: Use Documentation in Implementation

Claude uses the retrieved documentation to provide accurate, current implementation:

```
Claude: "Based on the current GDAL 3.10.3 documentation from docs7-agent, here's how to convert S-57 to GeoPackage using VectorTranslate..."
```

## Examples

### Example 1: GDAL S-57 Conversion

**User Query:**
```
"How do I use GDAL to convert S-57 .000 files to GeoPackage?"
```

**Claude Actions:**
1. Launch docs7-agent with Task tool:
   - Prompt: "Get GDAL documentation for VectorTranslate function, specifically for S-57 to GeoPackage conversion"
2. docs7-agent retrieves current GDAL 3.10.3 documentation
3. Provides accurate gdal.VectorTranslate() usage with current parameters

### Example 2: GeoPandas CRS Handling

**User Query:**
```
"What's the correct way to handle CRS in GeoPandas when writing to PostGIS?"
```

**Claude Actions:**
1. Launch docs7-agent with Task tool:
   - Prompt: "Get GeoPandas documentation for to_postgis method and CRS handling"
2. docs7-agent retrieves current GeoPandas documentation
3. Provides current to_postgis() parameters and CRS handling best practices

### Example 3: SQLAlchemy 2.0 Patterns

**User Query:**
```
"Show me how to create a PostGIS connection with SQLAlchemy 2.0."
```

**Claude Actions:**
1. Launch docs7-agent with Task tool:
   - Prompt: "Get SQLAlchemy 2.0 documentation for creating database engines and PostGIS connections"
2. docs7-agent retrieves current SQLAlchemy 2.0 documentation
3. Provides v2.0-compatible create_engine() usage (avoiding deprecated v1.x patterns)

## Common Use Cases

### GDAL Operations

```
Queries requiring docs7-agent:
- "How to use gdal.OpenEx with S-57 files?"
- "What are the gdal.VectorTranslate options for GeoPackage?"
- "How to set GDAL config options programmatically?"
```

**Task tool invocation:**
```
subagent_type: "docs7-agent"
prompt: "Get GDAL documentation for [specific function/topic]"
```

### GeoPandas Spatial Operations

```
Queries requiring docs7-agent:
- "How to perform spatial join in GeoPandas?"
- "What's the syntax for gdf.to_file() with GeoPackage?"
- "How to handle multi-part geometries in GeoPandas?"
```

**Task tool invocation:**
```
subagent_type: "docs7-agent"
prompt: "Get GeoPandas documentation for [specific operation]"
```

### SQLAlchemy Database Operations

```
Queries requiring docs7-agent:
- "How to create SQLAlchemy engine for PostGIS?"
- "What's the correct GeoAlchemy2 import pattern?"
- "How to execute raw SQL with SQLAlchemy 2.0?"
```

**Task tool invocation:**
```
subagent_type: "docs7-agent"
prompt: "Get SQLAlchemy 2.0 documentation for [specific topic]"
```

## Library Reference Information

Common libraries used by docs7-agent (for reference only - agent handles resolution automatically):

| Library | Internal ID Pattern | Notes |
|---------|-------------|-------|
| GDAL | `/osgeo/gdal` | Version-specific: `/osgeo/gdal/v3.10.3` |
| GeoPandas | `/geopandas/geopandas` | Latest stable |
| SQLAlchemy | `/sqlalchemy/sqlalchemy` | v2.0+ patterns |
| Pydantic | `/pydantic/pydantic` | v2.x validation |
| NetworkX | `/networkx/networkx` | Graph algorithms |
| Pandas | `/pandas-dev/pandas` | DataFrames |
| H3 | `/uber/h3-py` | Hexagonal grids |

**Note:** You don't need to know these IDs - docs7-agent resolves library names automatically. This table is for reference only.

## Best Practices

1. **Be Specific with Prompts**: Provide clear, focused prompts to docs7-agent
   - Good: "Get GDAL documentation for VectorTranslate with S-57 format specifics"
   - Avoid: "Get GDAL documentation" (too general)

2. **Request Proactively**: Launch docs7-agent at start of implementation, not after errors

3. **Version Awareness**: Mention specific versions when critical (e.g., GDAL 3.10.3, SQLAlchemy 2.0)

4. **Combine with Examples**: docs7-agent docs + working examples = best results

5. **Trust the Agent**: docs7-agent handles library resolution and documentation retrieval autonomously

## Common Issues

### Issue: Generic Documentation Returned

**Symptom**: docs7-agent returns broad, unfocused documentation

**Solution**: Provide more specific prompts
```
Good: "Get GDAL VectorTranslate documentation for S-57 format conversion to GeoPackage"
Avoid: "Get GDAL documentation"
```

### Issue: Outdated Patterns Suggested

**Symptom**: Claude suggests deprecated API usage

**Cause**: docs7-agent wasn't launched

**Solution**:
- Explicitly launch docs7-agent for library-specific tasks
- Claude should proactively use docs7-agent when working with external libraries

### Issue: Wrong Library Version

**Symptom**: Documentation for wrong version (e.g., GDAL 3.9 instead of 3.10.3)

**Solution**: Specify version in prompt
```
"Get GDAL 3.10.3 documentation for VectorTranslate"
```

## How to Launch docs7-agent

Use the Task tool with the following parameters:

```python
Task(
    subagent_type="docs7-agent",
    description="Fetch [library] documentation",  # 3-5 word description
    prompt="Get [library name/version] documentation for [specific topic/function]"
)
```

**Example invocations:**

```python
# GDAL documentation
Task(
    subagent_type="docs7-agent",
    description="Fetch GDAL VectorTranslate docs",
    prompt="Get GDAL 3.10.3 documentation for VectorTranslate function, specifically for converting S-57 format to GeoPackage"
)

# GeoPandas documentation
Task(
    subagent_type="docs7-agent",
    description="Fetch GeoPandas CRS docs",
    prompt="Get GeoPandas documentation for CRS handling and to_postgis method"
)

# SQLAlchemy documentation
Task(
    subagent_type="docs7-agent",
    description="Fetch SQLAlchemy engine docs",
    prompt="Get SQLAlchemy 2.0 documentation for creating database engines with PostgreSQL and PostGIS"
)
```

## Related Skills

- **environment-setup**: Environment Setup (library installation)
- **gdal-s57-setup**: GDAL S-57 Setup (GDAL-specific configuration)
- **s57-conversion-patterns**: S-57 Conversion Patterns (applies GDAL knowledge)

## Cross-References

- **docs7-agent Definition**: `.claude/agents/docs7-agent.md` (agent implementation details)
- **Project Knowledge**: `/dev/rules/CLAUDE.md` (Documentation and API Accuracy section)
- **Agent Guidelines**: `/dev/rules/AGENTS.md` (docs7-agent Integration section)
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md`