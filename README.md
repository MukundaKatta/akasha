# akasha — Asteroid Detection. Near-Earth object detection and trajectory prediction from telescope data

**Live:** <https://mukundakatta.github.io/akasha/>

Asteroid Detection. Near-Earth object detection and trajectory prediction from telescope data.

## Why akasha

akasha exists to make this workflow practical. Asteroid detection. near-earth object detection and trajectory prediction from telescope data. It favours a small, inspectable surface over sprawling configuration.

## Features

- CLI command `akasha`
- Included test suite
- Worked examples included

## Tech Stack

- **Runtime:** Python
- **Frameworks:** Click
- **AI/ML:** NumPy
- **Tooling:** pytest, Pydantic, Rich

## How It Works

The codebase is organised into `examples/`, `src/`, `tests/`. The primary entry points are `src/akasha/cli.py`, `src/akasha/__init__.py`. `src/akasha/cli.py` exposes functions like `cli`, `simulate`.

## Getting Started

```bash
pip install -e .
akasha --help
```

## Usage

```bash
akasha --help
```

## Project Structure

```
akasha/
├── .env.example
├── CLAUDE.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── config.example.yaml
├── examples/
├── index.html
├── pyproject.toml
├── requirements.txt
├── src/
├── tests/
```