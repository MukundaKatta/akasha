"""Report generation with Rich formatting."""

from __future__ import annotations

from io import StringIO
from typing import Optional

import numpy as np

from akasha.models import Asteroid, OrbitalElements, RiskAssessment

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _fmt_angle(rad: float) -> str:
    """Format an angle in radians as degrees with 4 decimal places."""
    return f"{np.degrees(rad):.4f} deg"


def _fmt_au(val: float) -> str:
    return f"{val:.6f} AU"


class ReportGenerator:
    """Generate formatted reports for asteroids and risk assessments.

    Parameters
    ----------
    file : optional
        Writable file object.  If None, prints to stdout via Rich console.
    """

    def __init__(self, file=None) -> None:
        self._file = file
        if HAS_RICH:
            self.console = Console(file=file)
        else:
            self.console = None

    # ------------------------------------------------------------------
    # Orbital elements table
    # ------------------------------------------------------------------

    def orbital_elements_table(self, elements: OrbitalElements, title: str = "Orbital Elements") -> str:
        """Return a formatted table of orbital elements."""
        rows = [
            ("Semi-major axis (a)", _fmt_au(elements.semi_major_axis)),
            ("Eccentricity (e)", f"{elements.eccentricity:.6f}"),
            ("Inclination (i)", _fmt_angle(elements.inclination)),
            ("Long. asc. node (Omega)", _fmt_angle(elements.longitude_of_ascending_node)),
            ("Arg. perihelion (omega)", _fmt_angle(elements.argument_of_perihelion)),
            ("Mean anomaly (M)", _fmt_angle(elements.mean_anomaly)),
            ("Perihelion (q)", _fmt_au(elements.perihelion)),
            ("Aphelion (Q)", _fmt_au(elements.aphelion)),
            ("Period (T)", f"{elements.period:.4f} yr"),
            ("Epoch", str(elements.epoch)),
        ]

        if HAS_RICH and self.console:
            table = Table(title=title, show_header=True)
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            for name, val in rows:
                table.add_row(name, val)
            buf = StringIO()
            Console(file=buf, width=80).print(table)
            return buf.getvalue()
        else:
            lines = [f"=== {title} ==="]
            for name, val in rows:
                lines.append(f"  {name:30s} {val}")
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Risk assessment report
    # ------------------------------------------------------------------

    def risk_report(self, assessment: RiskAssessment) -> str:
        """Return a formatted risk assessment report."""
        rows = [
            ("Asteroid ID", assessment.asteroid_id),
            ("MOID", _fmt_au(assessment.moid_au)),
            ("Impact probability", f"{assessment.impact_probability:.2e}"),
            ("Kinetic energy", f"{assessment.kinetic_energy_mt:.2e} MT"),
            ("Palermo scale", f"{assessment.palermo_scale:.2f}"),
            ("Torino scale", f"{assessment.torino_scale.value} ({assessment.torino_scale.name})"),
            ("Time to closest approach", f"{assessment.time_to_closest_approach_years:.1f} yr"),
        ]

        if assessment.notes:
            rows.append(("Notes", assessment.notes))

        if HAS_RICH and self.console:
            table = Table(title="Impact Risk Assessment", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="red" if assessment.torino_scale.value >= 3 else "green")
            for name, val in rows:
                table.add_row(name, val)
            buf = StringIO()
            Console(file=buf, width=80).print(table)
            return buf.getvalue()
        else:
            lines = ["=== Impact Risk Assessment ==="]
            for name, val in rows:
                lines.append(f"  {name:30s} {val}")
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Full asteroid summary
    # ------------------------------------------------------------------

    def asteroid_summary(self, asteroid: Asteroid, risk: Optional[RiskAssessment] = None) -> str:
        """Full summary for a single asteroid."""
        parts: list[str] = []

        header = f"Asteroid: {asteroid.asteroid_id}"
        if asteroid.name:
            header += f" ({asteroid.name})"
        parts.append(header)
        parts.append(f"  NEO type: {asteroid.neo_type.value}")
        parts.append(f"  Observations: {len(asteroid.observations)}")
        if asteroid.diameter_km is not None:
            parts.append(f"  Estimated diameter: {asteroid.diameter_km:.3f} km")
        if asteroid.absolute_magnitude is not None:
            parts.append(f"  Absolute magnitude H: {asteroid.absolute_magnitude:.2f}")

        if asteroid.orbital_elements:
            parts.append("")
            parts.append(self.orbital_elements_table(asteroid.orbital_elements))

        if risk:
            parts.append("")
            parts.append(self.risk_report(risk))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Population summary table
    # ------------------------------------------------------------------

    def population_table(
        self,
        asteroids: list[Asteroid],
        risks: Optional[list[RiskAssessment]] = None,
    ) -> str:
        """Tabular summary of a population of asteroids."""
        risk_map = {}
        if risks:
            risk_map = {r.asteroid_id: r for r in risks}

        if HAS_RICH and self.console:
            table = Table(title="Asteroid Population Summary", show_header=True)
            table.add_column("ID", style="cyan")
            table.add_column("Type")
            table.add_column("a (AU)", justify="right")
            table.add_column("e", justify="right")
            table.add_column("i (deg)", justify="right")
            table.add_column("MOID (AU)", justify="right")
            table.add_column("Torino", justify="center")
            table.add_column("Obs", justify="right")

            for ast in asteroids:
                oe = ast.orbital_elements
                r = risk_map.get(ast.asteroid_id)
                table.add_row(
                    ast.asteroid_id,
                    ast.neo_type.value,
                    f"{oe.semi_major_axis:.3f}" if oe else "-",
                    f"{oe.eccentricity:.4f}" if oe else "-",
                    f"{np.degrees(oe.inclination):.2f}" if oe else "-",
                    f"{r.moid_au:.4f}" if r else "-",
                    str(r.torino_scale.value) if r else "-",
                    str(len(ast.observations)),
                )
            buf = StringIO()
            Console(file=buf, width=120).print(table)
            return buf.getvalue()
        else:
            lines = ["=== Asteroid Population Summary ==="]
            hdr = f"  {'ID':10s} {'Type':8s} {'a(AU)':>8s} {'e':>8s} {'i(deg)':>8s} {'MOID':>10s} {'Torino':>6s} {'Obs':>4s}"
            lines.append(hdr)
            lines.append("  " + "-" * len(hdr.strip()))
            for ast in asteroids:
                oe = ast.orbital_elements
                r = risk_map.get(ast.asteroid_id)
                a_str = f"{oe.semi_major_axis:8.3f}" if oe else "       -"
                e_str = f"{oe.eccentricity:8.4f}" if oe else "       -"
                i_str = f"{np.degrees(oe.inclination):8.2f}" if oe else "       -"
                moid_str = f"{r.moid_au:10.4f}" if r else "         -"
                torino_str = f"{r.torino_scale.value:6d}" if r else "     -"
                lines.append(
                    f"  {ast.asteroid_id:10s} "
                    f"{ast.neo_type.value:8s} "
                    f"{a_str} {e_str} {i_str} {moid_str} {torino_str} "
                    f"{len(ast.observations):4d}"
                )
            return "\n".join(lines)

    def print(self, text: str) -> None:
        """Print text through the console or file."""
        if HAS_RICH and self.console:
            self.console.print(text)
        elif self._file:
            self._file.write(text + "\n")
        else:
            print(text)
