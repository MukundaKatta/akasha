"""Near-Earth Object orbit-type classifier.

Classifies asteroids into the four standard NEO sub-populations using
their orbital elements relative to Earth's orbit:

  Atira   -- orbit entirely inside Earth's (a < 1 AU, Q < q_E)
  Aten    -- Earth-crossing with a < 1 AU  (a < 1 AU, Q >= q_E)
  Apollo  -- Earth-crossing with a >= 1 AU (a >= 1 AU, q <= Q_E)
  Amor    -- Mars-crossing, approaching Earth (q > Q_E and q < 1.3 AU)

Reference values:
  Earth perihelion  q_E ~ 0.983 AU
  Earth aphelion    Q_E ~ 1.017 AU
"""

from __future__ import annotations

from akasha.models import Asteroid, NEOType, OrbitalElements


# Earth orbital boundaries (AU)
EARTH_PERIHELION = 0.983
EARTH_APHELION = 1.017
AMOR_Q_LIMIT = 1.3  # AU


class NEOClassifier:
    """Classify near-Earth objects by orbit type.

    Parameters
    ----------
    earth_perihelion : float
        Earth's perihelion distance in AU (default 0.983).
    earth_aphelion : float
        Earth's aphelion distance in AU (default 1.017).
    amor_q_limit : float
        Maximum perihelion distance for Amor classification (default 1.3 AU).
    """

    def __init__(
        self,
        earth_perihelion: float = EARTH_PERIHELION,
        earth_aphelion: float = EARTH_APHELION,
        amor_q_limit: float = AMOR_Q_LIMIT,
    ) -> None:
        self.q_e = earth_perihelion
        self.Q_e = earth_aphelion
        self.amor_q = amor_q_limit

    def classify(self, elements: OrbitalElements) -> NEOType:
        """Determine the NEO sub-type from orbital elements.

        Parameters
        ----------
        elements : OrbitalElements
            Keplerian orbital elements for the object.

        Returns
        -------
        NEOType
            One of Atira, Aten, Apollo, Amor, or Unknown.
        """
        a = elements.semi_major_axis
        q = elements.perihelion   # a * (1 - e)
        Q = elements.aphelion     # a * (1 + e)

        if a < 1.0:
            if Q < self.q_e:
                return NEOType.ATIRA
            elif Q >= self.q_e:
                return NEOType.ATEN
        else:  # a >= 1.0
            if q <= self.Q_e:
                return NEOType.APOLLO
            elif q > self.Q_e and q < self.amor_q:
                return NEOType.AMOR

        return NEOType.UNKNOWN

    def classify_asteroid(self, asteroid: Asteroid) -> Asteroid:
        """Classify an Asteroid in-place, returning it for chaining."""
        if asteroid.orbital_elements is None:
            asteroid.neo_type = NEOType.UNKNOWN
        else:
            asteroid.neo_type = self.classify(asteroid.orbital_elements)
        return asteroid

    def classify_batch(self, asteroids: list[Asteroid]) -> list[Asteroid]:
        """Classify a list of asteroids, returning the same list."""
        for ast in asteroids:
            self.classify_asteroid(ast)
        return asteroids
