## Project Overview

This project, `fwd-model-tools`, is a Python package for cosmological forward-modeling and sampling. It is built on top of JAX and JAXPM, and it provides tools for running N-body simulations, performing gravitational lensing calculations, and conducting Bayesian inference using MCMC methods. The library is designed to be highly performant and scalable, with support for distributed computing across multiple devices.

The core of the library is a set of JAX-native functions for defining and running simulations. These functions are designed to be composable and extensible, allowing users to build complex models from simple components. The library also provides a high-level interface for running MCMC simulations using either BlackJAX or NumPyro as the backend.

## Building and Running

### Installation

To install the package, run the following command:

```bash
pip install .
```

For development, you can install the package in editable mode with the development dependencies:

```bash
pip install -e .[dev]
```

### Running Tests

The project uses `pytest` for testing. To run the tests, use the following command:

```bash
pytest
```

### Running the Scripts

The `scripts` directory contains several example scripts that demonstrate how to use the library. For example, to run the simple sampling script, use the following command:

```bash
python scripts/run_simple_sampling.py --output-dir output/simple --num-warmup 500 --num-samples 1000
```

## Development Conventions

### Code Style

The project uses `ruff` for linting and formatting. The configuration can be found in the `pyproject.toml` file.

### Testing

The project uses `pytest` for testing. Tests are located in the `tests` directory. All new code should be accompanied by corresponding tests.

### Contribution Guidelines

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request. Make sure to run the tests before submitting a pull request.
