#!/usr/bin/env python3
"""Patch tbp.monty source files to lazy-import heavy packages.

Replaces top-level imports of wandb, pandas with lazy imports so that
monty_inference.py can run without these packages installed.

Note: torch-geometric CANNOT be patched away — model.pt contains pickled
torch_geometric.data.Data objects that require the module at unpickle time.
Install it with: pip3 install --no-deps torch-geometric

Usage:
    python3 scripts/patch_lazy_imports.py [--tbp-monty-src PATH]

This modifies files in-place. Run once after cloning tbp.monty.
Safe to run multiple times (idempotent — skips already-patched files).
"""
import argparse
import re
from pathlib import Path

PATCHES = [
    # --- object_model.py: torch_geometric ---
    {
        "file": "frameworks/models/object_model.py",
        "old": (
            "import torch_geometric\n"
            "import torch_geometric.transforms as T\n"
            "from scipy.spatial import KDTree\n"
            "from sklearn.neighbors import kneighbors_graph\n"
            "from torch_geometric.data import Data"
        ),
        "new": (
            "from scipy.spatial import KDTree\n"
            "try:\n"
            "    import torch_geometric\n"
            "    import torch_geometric.transforms as T\n"
            "    from sklearn.neighbors import kneighbors_graph\n"
            "    from torch_geometric.data import Data\n"
            "except ImportError:\n"
            "    torch_geometric = None  # Not needed for inference\n"
            "    T = None\n"
            "    kneighbors_graph = None\n"
            "    Data = None"
        ),
    },
    # --- graph_matching_loggers.py: wandb, pandas ---
    {
        "file": "frameworks/loggers/graph_matching_loggers.py",
        "old": (
            "import pandas as pd\n"
            "import wandb\n"
            "from sklearn.preprocessing import LabelEncoder"
        ),
        "new": (
            "try:\n"
            "    import pandas as pd\n"
            "except ImportError:\n"
            "    pd = None\n"
            "try:\n"
            "    import wandb\n"
            "except ImportError:\n"
            "    wandb = None\n"
            "try:\n"
            "    from sklearn.preprocessing import LabelEncoder\n"
            "except ImportError:\n"
            "    LabelEncoder = None"
        ),
    },
    # --- object_model_utils.py: torch_geometric version compat ---
    # In newer torch-geometric, Data.keys is a method, not a property.
    {
        "file": "frameworks/utils/object_model_utils.py",
        "old": "    for key in list(torch_graph.keys):",
        "new": (
            "    _keys = torch_graph.keys() "
            "if callable(torch_graph.keys) else torch_graph.keys\n"
            "    for key in list(_keys):"
        ),
    },
    # --- wandb_handlers.py: wandb, pandas ---
    {
        "file": "frameworks/loggers/wandb_handlers.py",
        "old": (
            "import pandas as pd\n"
            "import wandb"
        ),
        "new": (
            "try:\n"
            "    import pandas as pd\n"
            "except ImportError:\n"
            "    pd = None\n"
            "try:\n"
            "    import wandb\n"
            "except ImportError:\n"
            "    wandb = None"
        ),
    },
]


def apply_patches(src_root: Path):
    base = src_root / "tbp" / "monty"
    for patch in PATCHES:
        fpath = base / patch["file"]
        if not fpath.exists():
            print(f"SKIP (not found): {fpath}")
            continue

        content = fpath.read_text()

        if patch["old"] not in content:
            # Check if already patched
            if "except ImportError:" in content and any(
                keyword in patch["new"] for keyword in ["torch_geometric = None", "wandb = None", "pd = None"]
                if keyword in content
            ):
                print(f"SKIP (already patched): {fpath.name}")
            else:
                print(f"SKIP (pattern not found): {fpath.name}")
            continue

        content = content.replace(patch["old"], patch["new"])
        fpath.write_text(content)
        print(f"PATCHED: {fpath.name}")


def main():
    parser = argparse.ArgumentParser(description="Patch tbp.monty for minimal deps")
    parser.add_argument(
        "--tbp-monty-src",
        type=str,
        default=None,
        help="Path to tbp.monty/src/ (default: auto-detect from script location)",
    )
    args = parser.parse_args()

    if args.tbp_monty_src:
        src_root = Path(args.tbp_monty_src)
    else:
        # Auto-detect: scripts/ is next to tbp.monty/
        script_dir = Path(__file__).resolve().parent
        src_root = script_dir.parent / "tbp.monty" / "src"

    if not (src_root / "tbp" / "monty").exists():
        print(f"ERROR: tbp.monty source not found at {src_root}")
        print("Use --tbp-monty-src to specify the path to tbp.monty/src/")
        raise SystemExit(1)

    print(f"Patching source at: {src_root}")
    apply_patches(src_root)
    print("Done.")


if __name__ == "__main__":
    main()
