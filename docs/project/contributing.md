# Contributing to Nautical Graph Toolkit

Thank you for your interest in contributing to the Nautical Graph Toolkit! This project is currently in active development (v{{ project_version }}), and we're preparing for community contributions starting with **v0.2.0**.

## 📋 Project Status

**Current Version**: v{{ project_version }} ({{ last_updated }})

**Contribution Timeline**:

- **v0.1.x** (current: v0.1.5): Foundation completion in progress - internal contributions only
- **v0.2.0**: Community contributions welcome - PyTorch integration & production readiness
- **v0.3.0+**: Full collaborative development - QGIS integration & advanced features

## 🤝 How to Contribute (Starting v0.2.0)

When contributions open with v0.2.0, here's the workflow:

### 1. Fork the Repository
```bash
# Create your fork on GitHub
# Then clone it locally
git clone https://github.com/YOUR_USERNAME/Nautical-Graph-Toolkit.git
cd Nautical-Graph-Toolkit
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/your-bug-fix
```

### 3. Develop & Test

Follow the [code standards](#code-standards) listed in the Development Resources section below.

### Run Tests
```bash
# Install development dependencies
uv pip install -r requirements.txt
mamba activate nautical

# Run all tests
pytest

# Run specific test file
pytest tests/core/test_weights.py

# Run with coverage report
pytest --cov=nautical_graph_toolkit --cov-report=html
```

### 4. Commit with Clear Messages

Use conventional commit format:
```bash
git commit -m "feat(core): add weighted routing for obstacles"
git commit -m "fix(utils): resolve S-57 attribute parsing bug"
git commit -m "docs: update API reference"
git commit -m "test: add comprehensive graph weighting tests"
```

**Prefix types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:

- Clear description of changes
- Motivation and context
- Testing performed
- Screenshots for UI changes (if applicable)

## 📚 Development Resources

### Environment Setup

See the [Installation Guide](../getting-started/install.md) for complete setup instructions.

### Code Standards

Follow PEP 8 and the project conventions:

- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Google-style on all public functions
- **Type hints**: Required for all public methods
- **Testing**: Unit tests in `tests/core/` and `tests/utils/`, integration tests in `tests/core__real_data/`

See the project's development guidelines in `/dev/rules/` directory.

### Architecture

The toolkit uses a layered architecture under `src/nautical_graph_toolkit/`:

- **Core**: S-57 conversion, graph generation (`BaseGraph`, `FineGraph`, `H3Graph`), three-tier weighting system (`WeightCalculator`, `Weights`, `WeightsOpen`), and multi-pass A* pathfinding (`Astar`, `AstarImproved`, `AstarMaritime`, `AstarMaritimeSmooth`)
- **Utils**: Database managers, geometry utilities (`Buffer`, `Bearing`), S-57 classification, and port utilities
- **Data**: S-57 reference data, port index, and configurations
- **Analysis**: Analytical tools and visualization
- **Scripts**: CLI launcher (`ngt.py`), workflow scripts, benchmarks, and S-57 import tools

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/test_weights.py

# Run with coverage
pytest --cov=nautical_graph_toolkit
```

Test data is included in `data/ENC_ROOT/`

## 🐛 Report Bugs

Found an issue? Please report it!

**Use the [GitHub Issues](https://github.com/studentdotai/Nautical-Graph-Toolkit/issues) tracker with**:

1. **Title**: Clear, concise problem description
2. **Description**:
    - What you expected to happen
    - What actually happened
    - Steps to reproduce
3. **Environment**:
    - Python version (`python --version`)
    - GDAL version (`gdalinfo --version`)
    - Operating system
    - Conda/Mamba versions
4. **Code sample**: If applicable, minimal reproducible example
5. **Logs**: Any error messages or stack traces

Example:
```
**Title**: S-57 converter fails on large ENC files

**Description**:
When converting ENC files larger than 500MB, the converter crashes with:
- Expected: Conversion completes successfully
- Actual: MemoryError after processing
- Steps: Convert enc_west.enc (750MB) using S57Base.convert_by_enc()

**Environment**:
- Python 3.11.7
- GDAL 3.10.3
- macOS 14.2
- GDAL command: gdalinfo shows PostGIS support available

**Error**:
```
MemoryError: Unable to allocate 2.5 GiB
```
```

## 💡 Suggest Features

**Have an idea?** [Open a GitHub Discussion](https://github.com/studentdotai/Nautical-Graph-Toolkit/discussions) or [Issue](https://github.com/studentdotai/Nautical-Graph-Toolkit/issues).

Include:

- Problem statement
- Proposed solution
- Motivation & use cases
- Potential implementation approach

## 🌊 Support Vector Nautical

This is a **solo-developer initiative** supported by active seafarers and maritime researchers.

**Support the project**:

- ⭐ Star the repository
- 💬 Provide feedback and suggestions
- 🔗 Share with maritime/geospatial communities
- 💰 [Donate on Open Collective](https://opencollective.com/vectornautical)

## 📞 Questions?

- **General questions**: [GitHub Discussions](https://github.com/studentdotai/Nautical-Graph-Toolkit/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/studentdotai/Nautical-Graph-Toolkit/issues)
- **Feature requests**: [GitHub Issues](https://github.com/studentdotai/Nautical-Graph-Toolkit/issues) with `enhancement` label

## 📋 Code Review Process

When v0.2.0 opens contributions:

1. **Automated checks**:
    - CI/CD pipeline runs tests
    - Code coverage must be ≥80%
    - Style checks pass (Ruff)

2. **Manual review**:
    - Architecture & design
    - Code quality & readability
    - Documentation & examples
    - Performance implications

3. **Approval & merge**:
    - At least 1 maintainer approval
    - All CI checks pass
    - No merge conflicts

## 🎯 Priority Areas (When Contributions Open)

**High priority** (good for first-time contributors):

- Documentation improvements and examples
- Test coverage expansion (current: 11 test files, target: 80% for v0.2.0)
- Bug fixes and edge case handling
- Notebook examples for new v0.1.5 features (weights, pathfinding variants, CLI)

**Medium priority**:

- Cross-backend performance optimization
- Configuration and workflow enhancements
- RTZ 1.2 route export validation and testing
- ML weight optimization pipeline (`GraphWeightOptimizer`)

**Lower priority** (requires discussion):

- PyTorch integration (planned for v0.2.0)
- New core features or architecture changes
- QGIS plugin development (planned for v0.3.0)

## 📜 License

By contributing, you agree that your contributions will be licensed under the **AGPL-3.0 license**.

---

## Current Contributors

**Lead Developer**: Vector Nautical (solo developer)

**Advisors & Validators**:

- Maritime officers & researchers
- Academic PhD researchers
- GIS/geospatial specialists

**Interested in collaborating?** [Reach out on Open Collective](https://opencollective.com/vectornautical).

---

**Thank you for being interested in the Nautical Graph Toolkit!** ⚓🌊

Check back with v0.2.0 for full community contribution opportunities.