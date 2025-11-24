#!/usr/bin/env python3
"""
Launch the Windows FTEQW Quake engine from WSL/Python.

Usage examples (from repo root, inside your `rl` conda env):

    python code/quake/scripts/launch_fteqw.py
    python code/quake/scripts/launch_fteqw.py --map e1m1
    python code/quake/scripts/launch_fteqw.py --windowed
    python code/quake/scripts/launch_fteqw.py --extra "+skill 2 +noclip"

This script assumes the following layout:

    repo_root/
      code/quake/fteqw/windows/
        fteqw64.exe
        id1/pak0.pak
        id1/pak1.pak
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_repo_root() -> Path:
    """
    Resolve the repo root by walking up from this file until we see README.md.
    Adjust if your repo uses a different marker.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "README.md").exists():
            return parent
    # Fallback: use parent of this file's parent (code/quake/scripts -> repo_root)
    return Path(__file__).resolve().parents[3]


def build_fteqw_paths(repo_root: Path) -> dict:
    """
    Return important paths: engine exe and id1 directory.
    """
    fte_base = repo_root / "code" / "quake" / "fteqw" / "windows"
    exe_path = fte_base / "fteqw64.exe"
    id1_dir = fte_base / "id1"
    pak0 = id1_dir / "pak0.pak"
    pak1 = id1_dir / "pak1.pak"

    return {
        "fte_base": fte_base,
        "exe_path": exe_path,
        "id1_dir": id1_dir,
        "pak0": pak0,
        "pak1": pak1,
    }


def verify_install(paths: dict) -> None:
    """
    Make sure the engine and assets are present.
    """
    exe_path = paths["exe_path"]
    pak0 = paths["pak0"]
    pak1 = paths["pak1"]

    missing = []
    if not exe_path.exists():
        missing.append(f"Missing engine executable: {exe_path}")
    if not pak0.exists():
        missing.append(f"Missing pak0.pak: {pak0}")
    if not pak1.exists():
        missing.append(f"Missing pak1.pak: {pak1}")

    if missing:
        for m in missing:
            print(m, file=sys.stderr)
        raise SystemExit("FTEQW install appears incomplete. Fix the issues above.")


def launch_fteqw(exe_path: Path, args: list[str]) -> None:
    """
    Launch the Windows fteqw64.exe from WSL using subprocess.
    """
    # Convert to string; in WSL this will look like /mnt/c/Users/...
    exe_str = str(exe_path)

    cmd = [exe_str] + args
    print("Launching FTEQW with command:")
    print("  ", " ".join(cmd))

    try:
        # Start the process and return; the engine runs separately.
        subprocess.Popen(cmd)
    except FileNotFoundError:
        print(f"Could not execute: {exe_str}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error launching FTEQW: {e}", file=sys.stderr)
        raise


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch FTEQW Quake engine from WSL."
    )
    parser.add_argument(
        "--map",
        type=str,
        default=None,
        help="Map name to load (e.g., e1m1). If omitted, engine shows main menu.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Force windowed mode (equivalent to '+set vid_fullscreen 0').",
    )
    parser.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra command-line arguments to pass to FTEQW (e.g. '+skill 2 +noclip').",
    )
    return parser.parse_args()


def build_engine_args(args: argparse.Namespace) -> list[str]:
    """
    Build the argument list for FTEQW based on CLI options.
    """
    engine_args: list[str] = []

    # Force using this directory as basedir (optional, but explicit)
    # FTE normally auto-detects, but this can reduce confusion.
    # Example:
    # engine_args.extend(["-basedir", "."])

    # Windowed mode via cvar.
    if args.windowed:
        engine_args.extend(["+set", "vid_fullscreen", "0"])

    # Map to load.
    if args.map:
        engine_args.extend(["+map", args.map])

    # Extra raw args (single string, like '+skill 2 +noclip').
    if args.extra:
        # Split on whitespace so it becomes separate arguments.
        engine_args.extend(args.extra.split())

    return engine_args


def main() -> None:
    repo_root = find_repo_root()
    paths = build_fteqw_paths(repo_root)
    verify_install(paths)

    cli_args = parse_cli()
    engine_args = build_engine_args(cli_args)

    launch_fteqw(paths["exe_path"], engine_args)


if __name__ == "__main__":
    main()
