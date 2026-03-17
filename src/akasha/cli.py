"""AKASHA command-line interface."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from akasha import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="akasha")
def cli() -> None:
    """AKASHA -- Asteroid Detection and Trajectory Prediction."""


# ------------------------------------------------------------------
# simulate
# ------------------------------------------------------------------


@cli.command()
@click.option("--num-asteroids", "-n", default=10, help="Number of asteroids to generate.")
@click.option("--num-frames", "-f", default=5, help="Observation frames per asteroid.")
@click.option("--cadence", "-c", default=30.0, help="Minutes between frames.")
@click.option("--noise", default=0.5, help="Astrometric noise (arcsec).")
@click.option("--seed", "-s", default=None, type=int, help="Random seed.")
@click.option("--output", "-o", default=None, type=click.Path(), help="Output JSON path.")
def simulate(
    num_asteroids: int,
    num_frames: int,
    cadence: float,
    noise: float,
    seed: int | None,
    output: str | None,
) -> None:
    """Generate synthetic asteroid observations."""
    from akasha.simulator import AsteroidSimulator

    console.print(f"[bold cyan]Generating {num_asteroids} synthetic asteroids...[/]")
    sim = AsteroidSimulator(seed=seed)
    population = sim.generate_population(
        num_asteroids=num_asteroids,
        num_frames=num_frames,
        cadence_minutes=cadence,
        noise_arcsec=noise,
    )

    data = sim.to_json(population)
    if output:
        Path(output).write_text(data)
        console.print(f"[green]Wrote {output}[/]")
    else:
        click.echo(data)

    console.print(f"[bold green]Generated {len(population)} asteroids with {num_frames} frames each.[/]")


# ------------------------------------------------------------------
# scan
# ------------------------------------------------------------------


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--threshold", "-t", default=3.0, help="Detection sigma threshold.")
def scan(input_path: str, threshold: float) -> None:
    """Detect moving objects in observation data."""
    from akasha.detection.classifier import NEOClassifier
    from akasha.models import Asteroid
    from akasha.orbit.calculator import OrbitCalculator

    data = json.loads(Path(input_path).read_text())
    asteroids = [Asteroid.model_validate(d) for d in data]

    calc = OrbitCalculator()
    classifier = NEOClassifier()

    console.print(f"[bold cyan]Scanning {len(asteroids)} objects...[/]")
    for ast in asteroids:
        if len(ast.observations) >= 3 and ast.orbital_elements is None:
            try:
                ast.orbital_elements = calc.determine_orbit(ast.observations)
            except Exception as exc:
                console.print(f"  [yellow]Orbit failed for {ast.asteroid_id}: {exc}[/]")
        classifier.classify_asteroid(ast)
        console.print(f"  {ast.asteroid_id}: [green]{ast.neo_type.value}[/]")

    console.print(f"[bold green]Scan complete. {len(asteroids)} objects classified.[/]")


# ------------------------------------------------------------------
# orbit
# ------------------------------------------------------------------


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--asteroid-id", "-a", required=True)
def orbit(input_path: str, asteroid_id: str) -> None:
    """Compute orbital elements for an asteroid."""
    from akasha.models import Asteroid
    from akasha.orbit.calculator import OrbitCalculator
    from akasha.report import ReportGenerator

    data = json.loads(Path(input_path).read_text())
    asteroids = {d["asteroid_id"]: Asteroid.model_validate(d) for d in data}

    if asteroid_id not in asteroids:
        console.print(f"[red]Asteroid {asteroid_id} not found.[/]")
        sys.exit(1)

    ast = asteroids[asteroid_id]
    calc = OrbitCalculator()
    elements = calc.determine_orbit(ast.observations)

    rpt = ReportGenerator()
    console.print(rpt.orbital_elements_table(elements, title=f"Orbit for {asteroid_id}"))


# ------------------------------------------------------------------
# risk
# ------------------------------------------------------------------


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--asteroid-id", "-a", required=True)
@click.option("--diameter", "-d", default=0.1, help="Estimated diameter (km).")
def risk(input_path: str, asteroid_id: str, diameter: float) -> None:
    """Assess impact risk for an asteroid."""
    from akasha.models import Asteroid
    from akasha.orbit.calculator import OrbitCalculator
    from akasha.orbit.risk import ImpactRiskAssessor
    from akasha.report import ReportGenerator

    data = json.loads(Path(input_path).read_text())
    asteroids = {d["asteroid_id"]: Asteroid.model_validate(d) for d in data}

    if asteroid_id not in asteroids:
        console.print(f"[red]Asteroid {asteroid_id} not found.[/]")
        sys.exit(1)

    ast = asteroids[asteroid_id]
    if ast.orbital_elements is None:
        calc = OrbitCalculator()
        ast.orbital_elements = calc.determine_orbit(ast.observations)

    assessor = ImpactRiskAssessor()
    assessment = assessor.assess(
        ast.orbital_elements,
        asteroid_id=asteroid_id,
        diameter_km=diameter,
    )

    rpt = ReportGenerator()
    console.print(rpt.risk_report(assessment))


# ------------------------------------------------------------------
# report
# ------------------------------------------------------------------


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default=None, type=click.Path())
def report(input_path: str, output: str | None) -> None:
    """Generate a full population report."""
    from akasha.detection.classifier import NEOClassifier
    from akasha.models import Asteroid
    from akasha.orbit.calculator import OrbitCalculator
    from akasha.orbit.risk import ImpactRiskAssessor
    from akasha.report import ReportGenerator

    data = json.loads(Path(input_path).read_text())
    asteroids = [Asteroid.model_validate(d) for d in data]

    calc = OrbitCalculator()
    classifier = NEOClassifier()
    assessor = ImpactRiskAssessor(moid_samples=360)

    console.print(f"[bold cyan]Processing {len(asteroids)} asteroids...[/]")
    risks = []
    for ast in asteroids:
        if len(ast.observations) >= 3 and ast.orbital_elements is None:
            try:
                ast.orbital_elements = calc.determine_orbit(ast.observations)
            except Exception:
                pass
        classifier.classify_asteroid(ast)
        if ast.orbital_elements:
            r = assessor.assess(
                ast.orbital_elements,
                asteroid_id=ast.asteroid_id,
                diameter_km=ast.diameter_km or 0.1,
            )
            risks.append(r)

    rpt = ReportGenerator()
    text = rpt.population_table(asteroids, risks)

    if output:
        Path(output).write_text(text)
        console.print(f"[green]Report written to {output}[/]")
    else:
        console.print(text)


if __name__ == "__main__":
    cli()
