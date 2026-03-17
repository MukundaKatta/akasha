# AKASHA - Asteroid Detection and Trajectory Prediction

AKASHA is a Python toolkit for detecting asteroids in telescope image sequences,
computing orbital elements from observations, propagating trajectories, and
assessing impact risk using standard scales (Palermo and Torino).

## Features

- **Detection Pipeline** -- Identify moving objects in sequential telescope frames,
  classify near-Earth objects by orbit type (Aten, Apollo, Amor, Atira), and link
  detections across frames into consistent tracks.
- **Orbit Determination** -- Compute Keplerian orbital elements from three or more
  observations using Gauss's method, then refine with least-squares differential
  correction.
- **Trajectory Propagation** -- Predict future positions by solving Kepler's equation
  via Newton-Raphson iteration, with optional perturbation support.
- **Impact Risk Assessment** -- Calculate Minimum Orbit Intersection Distance (MOID)
  and rate hazard on both the Palermo Technical Scale and the Torino Scale.
- **Synthetic Observation Generator** -- Create realistic test data with configurable
  noise, cadence, and asteroid populations.
- **Rich CLI** -- Interactive command-line interface built with Click and Rich for
  scanning, tracking, orbit fitting, and risk reporting.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Generate synthetic observations
akasha simulate --num-asteroids 50 --num-frames 10 --output observations.json

# Detect moving objects
akasha scan --input observations.json --threshold 3.0

# Compute orbit
akasha orbit --input observations.json --asteroid-id AST-001

# Assess risk
akasha risk --input observations.json --asteroid-id AST-001

# Full report
akasha report --input observations.json --output report.txt
```

## Project Structure

```
src/akasha/
  cli.py              Command-line interface
  models.py           Pydantic data models
  simulator.py        Synthetic observation generator
  report.py           Report generation
  detection/
    scanner.py        Moving-object detection
    classifier.py     NEO orbit-type classification
    tracker.py        Multi-frame detection linking
  orbit/
    calculator.py     Orbital element determination
    propagator.py     Kepler-equation trajectory propagation
    risk.py           MOID and impact-risk scoring
```

## Author

Mukunda Katta
