# AGENTS.md - QuantSandbox Workspace

This is the workspace for the Quantitative Modeling & Backtesting Framework project.

## Session Setup

Every session, before doing anything else:
1. Read `todo.json` - current project tasks and progress
2. Read `CHANGELOG.md` - recent changes and updates
3. Read `docs/` directory for project documentation
4. Check recent code changes in the repository

## Memory & Documentation

- **Task Tracking**: `todo.json` - track all project tasks, their status, and notes
- **Changelog**: `CHANGELOG.md` - record all significant changes, improvements, and fixes
- **Documentation**: `docs/` directory - detailed documentation for modules, APIs, and usage
- **Code Comments**: All code must have clear, English comments explaining functionality

## Project Structure

```
QuantSandbox/
├── data/                      # Stock data files
│   ├── by_stock/             # Individual stock CSV files
│   └── stock_basic.csv       # Stock metadata
├── models/                   # Model implementation
│   ├── base/                 # Abstract base classes
│   ├── sklearn_models/       # Sklearn-based models
│   └── pytorch_models/       # PyTorch-based deep learning models
├── backtesting/              # Backtesting framework
│   ├── strategies/           # Trading strategies
│   └── metrics/              # Performance metrics
├── utils/                    # Utility functions
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── evaluation.py         # Evaluation metrics
│   └── logger.py             # Logging functionality
├── tests/                    # Test scripts
├── docs/                     # Documentation
├── config.yaml               # Configuration file
├── todo.json                 # Task tracking
├── AGENTS.md                 # This file
├── CHANGELOG.md              # Changelog
└── README.md                 # Project overview
```

## Coding Standards

- All code, comments, and documentation must be in English
- Follow PEP 8 style guidelines for Python code
- Use type hints for all functions and classes
- Write modular, extensible code with clear separation of concerns
- Include unit tests for all major functionality

## Workflow

1. **Task Planning**: Review `todo.json` and select the next task to work on
2. **Implementation**: Write clean, documented code following the project structure
3. **Testing**: Test the implemented functionality thoroughly
4. **Documentation**: Update relevant documentation files
5. **Tracking**: Update `todo.json` with task completion status and notes
6. **Changelog**: Record significant changes in `CHANGELOG.md`

## Quality Assurance

- All code must be tested before marking a task as complete
- Validate model performance using appropriate metrics
- Ensure backtesting results are reproducible
- Document all assumptions and limitations

## Evolution

This document can be updated as the project evolves. Any significant changes to the workflow or structure should be recorded here.
