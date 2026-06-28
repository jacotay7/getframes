# SPDX-License-Identifier: MIT
"""Shared helpers for the getframes examples.

Every example accepts the same flags so they behave consistently:

    python examples/01_basic_dark_frame.py            # just print numbers
    python examples/01_basic_dark_frame.py --plot     # also open a figure window
    python examples/01_basic_dark_frame.py --save out.png   # render to a file

Plotting needs matplotlib, which is an optional dependency:

    pip install -e ".[examples]"

This module is intentionally tiny and self-contained; it only handles argument
parsing and a shared matplotlib look so the examples themselves stay focused on
the getframes API.
"""

from __future__ import annotations

import argparse
from typing import Any


def build_parser(description: str | None) -> argparse.ArgumentParser:
    """Return an argument parser with the flags shared by every example."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--plot", action="store_true", help="Display the figure in an interactive window."
    )
    parser.add_argument(
        "--save", metavar="PATH", default=None, help="Save the figure to PATH (e.g. figure.png)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (default: 0).")
    return parser


def get_pyplot(args: argparse.Namespace) -> Any | None:
    """Return a styled ``pyplot`` module if plotting was requested, else ``None``.

    When only saving (``--save`` without ``--plot``) a non-interactive backend is
    selected so the example also runs headless, e.g. in CI or over SSH.
    """
    if not (args.plot or args.save):
        return None
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit(
            "Plotting needs matplotlib. Install it with: pip install -e '.[examples]'"
        ) from exc
    if args.save and not args.plot:
        matplotlib.use("Agg")  # render to a file without needing a display
    import matplotlib.pyplot as plt

    _apply_style(plt)
    return plt


def _apply_style(plt: Any) -> None:
    """Apply a clean, consistent look shared by all example figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "image.cmap": "magma",
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


# A small, colour-blind-friendly palette used across the examples.
PALETTE = {
    "blue": "#4C72B0",
    "red": "#C44E52",
    "green": "#55A868",
    "orange": "#DD8452",
    "purple": "#8172B3",
    "grey": "#8C8C8C",
}


def finish(plt: Any, fig: Any, args: argparse.Namespace) -> None:
    """Lay out, then save and/or show ``fig`` according to the CLI flags."""
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save)
        print(f"\nSaved figure to {args.save}")
    if args.plot:
        plt.show()
