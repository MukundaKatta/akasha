"""CLI for akasha."""
import sys, json, argparse
from .core import Akasha

def main():
    parser = argparse.ArgumentParser(description="Akasha — Asteroid Detection. Near-Earth object detection and trajectory prediction from telescope data.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Akasha()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.detect(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"akasha v0.1.0 — Akasha — Asteroid Detection. Near-Earth object detection and trajectory prediction from telescope data.")

if __name__ == "__main__":
    main()
