"""Tests for Akasha."""
from src.core import Akasha
def test_init(): assert Akasha().get_stats()["ops"] == 0
def test_op(): c = Akasha(); c.detect(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Akasha(); [c.detect() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Akasha(); c.detect(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Akasha(); r = c.detect(); assert r["service"] == "akasha"
