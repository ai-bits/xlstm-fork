# xLSTM Development Guide

## Build & Test Commands
- Install package: `pip install -e .`
- Run all tests: `pytest`
- Run single test file: `pytest tests/test_generation.py`
- Run specific test: `pytest tests/test_generation.py::test_generate_no_prefill`
- Run tests with keyword: `pytest -k "generation"`
- Run experiment: `python experiments/main.py --config experiments/parity_xlstm11.yaml`

## Code Style Guidelines
- **Imports**: Standard library → third-party → project modules; group by category
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Types**: Use type hints for all parameters and return values
- **Documentation**: Brief docstrings for classes and functions
- **Structure**: Use dataclasses for configurations, inherit from nn.Module for components
- **Formatting**: 4-space indentation, blank lines between logical sections
- **Error handling**: Use assertions for validation, raise ValueError with descriptive messages
- **Components**: Follow initialization → forward pass → reset_parameters pattern

Note: GPU required for tests (skipped if unavailable).