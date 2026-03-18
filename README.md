# akasha

**Akasha — Asteroid Detection. Near-Earth object detection and trajectory prediction from telescope data.**

![Build](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-proprietary-red)

## Install
```bash
pip install -e ".[dev]"
```

## Quick Start
```python
from src.core import Akasha
 instance = Akasha()
r = instance.detect(input="test")
```

## CLI
```bash
python -m src status
python -m src run --input "data"
```

## API
| Method | Description |
|--------|-------------|
| `detect()` | Detect |
| `scan()` | Scan |
| `monitor()` | Monitor |
| `alert()` | Alert |
| `get_report()` | Get report |
| `configure()` | Configure |
| `get_stats()` | Get stats |
| `reset()` | Reset |

## Test
```bash
pytest tests/ -v
```

## License
(c) 2026 Officethree Technologies. All Rights Reserved.
