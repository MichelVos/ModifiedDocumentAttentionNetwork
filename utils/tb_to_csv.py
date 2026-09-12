#!/usr/bin/env python3
"""Export TensorBoard event scalar data from multiple runs into a single CSV file.

Usage examples:

# Basic (long format -- default):
python utils/tb_to_csv.py runs/dir1 runs/dir2 --output all_scalars.csv

# Include subdirectories recursively and labels:
python utils/tb_to_csv.py runs/* --recursive --labels runA runB runC --output all_scalars.csv

# Wide format (requires pandas):
python utils/tb_to_csv.py runs/* --recursive --output all_scalars_wide.csv --wide

# Manifest example (JSON array of objects with path and description):
# [{"path": "runs/dir1", "description": "exp-A"}, {"path": "runs/dir2", "description": "exp-B"}]
# Use: python utils/tb_to_csv.py --manifest manifest.json --output all_scalars.csv

# Tag-specific wide export (one column per run, grouped by step):
# python utils/tb_to_csv.py runs/* --recursive --tag "loss" --wide --output loss_by_run.csv
# Note: wide outputs group on `step` only (no wall_time) and aggregate duplicate step values by mean.

Output long format columns: run, description, tag, step, wall_time, value, file
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import sys
from typing import Iterable, List, Tuple

try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception as e:  # pragma: no cover - helpful error for user
    sys.exit("Error: could not import tensorboard's event_accumulator. Install with `pip install tensorboard`.")


def find_event_files(paths: Iterable[str], recursive: bool) -> List[str]:
    matches = []
    for p in paths:
        if os.path.isfile(p) and os.path.basename(p).startswith("events.out.tfevents"):
            matches.append(p)
            continue
        if os.path.isdir(p):
            pattern = "**/events.out.tfevents*" if recursive else "events.out.tfevents*"
            found = glob.glob(os.path.join(p, pattern), recursive=recursive)
            matches.extend(found)
            continue
        # treat glob patterns
        found = glob.glob(p, recursive=recursive)
        for f in found:
            if os.path.isfile(f) and os.path.basename(f).startswith("events.out.tfevents"):
                matches.append(f)
            elif os.path.isdir(f):
                pattern = "**/events.out.tfevents*" if recursive else "events.out.tfevents*"
                matches.extend(glob.glob(os.path.join(f, pattern), recursive=recursive))
    return sorted(set(matches))


def read_manifest(manifest_path: str) -> List[dict]:
    """Read a manifest file (JSON or CSV) and return list of dicts with keys 'path', 'description', 'label' (optional)."""
    if manifest_path.lower().endswith(".json"):
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Manifest JSON must be an array of objects with 'path' and optional 'description' fields.")
        out = []
        for item in data:
            if not isinstance(item, dict) or "path" not in item:
                raise ValueError("Each manifest entry must be an object with at least a 'path' key.")
            out.append({"path": item["path"], "description": item.get("description"), "label": item.get("label")})
        return out
    # try CSV
    if manifest_path.lower().endswith(".csv"):
        import csv
        out = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if "path" not in row:
                    raise ValueError("CSV manifest must include a 'path' column.")
                out.append({"path": row.get("path"), "description": row.get("description"), "label": row.get("label")})
        return out
    raise ValueError("Unsupported manifest format. Use .json or .csv")


def expand_manifest_entries(entries: List[dict], recursive: bool) -> List[tuple]:
    """Expand manifest entries to (event_file_path, label, description) tuples.

    If an entry path expands to multiple event files, each gets the same label/description.
    """
    out = []
    for e in entries:
        p = e["path"]
        label = e.get("label")
        description = e.get("description")
        found = find_event_files([p], recursive=recursive)
        if not found:
            # keep entry even if no event files found; caller may want to see message
            continue
        for ef in found:
            out.append((ef, label, description))
    return out


def read_scalars_from_eventfile(path: str) -> List[Tuple[str, int, float, float]]:
    """Return list of tuples (tag, step, wall_time, value) from an event file."""
    ea = event_accumulator.EventAccumulator(path, size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: 0,
        event_accumulator.IMAGES: 0,
        event_accumulator.AUDIO: 0,
        event_accumulator.HISTOGRAMS: 0,
        event_accumulator.SCALARS: 0,
        event_accumulator.TENSORS: 0,
    })
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    results = []
    for tag in tags:
        try:
            scalars = ea.Scalars(tag)
        except Exception:
            continue
        for s in scalars:
            # s is a ScalarEvent with wall_time, step, value
            results.append((tag, int(s.step), float(s.wall_time), float(s.value)))
    return results


def collect_runs(event_files: List[str], labels: List[str] | None = None, descriptions: List[str] | None = None) -> List[dict]:
    """Collect scalar rows from event files.

    Each row includes: run, label, description, tag, step, wall_time, value, file
    """
    rows = []
    if labels and len(labels) != len(event_files):
        raise ValueError("If --labels is used it must have same length as number of discovered event files.")
    if descriptions and len(descriptions) != len(event_files):
        raise ValueError("If descriptions are provided they must match the number of discovered event files.")
    for idx, ef in enumerate(event_files):
        label_val = labels[idx] if labels else None
        run_name = label_val if label_val else (os.path.basename(os.path.dirname(ef)) or os.path.basename(ef))
        desc = descriptions[idx] if descriptions else None
        scalars = read_scalars_from_eventfile(ef)
        for tag, step, wall_time, value in scalars:
            rows.append({
                "run": run_name,
                "label": label_val or "",
                "description": desc or "",
                "tag": tag,
                "step": step,
                "wall_time": wall_time,
                "value": value,
                "file": ef,
            })
    return rows


def write_long_csv(rows: List[dict], outpath: str) -> None:
    header = ["run", "description", "tag", "step", "wall_time", "value", "file"]
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def write_wide_csv(rows: List[dict], outpath: str) -> None:
    try:
        import pandas as pd
    except Exception:
        sys.exit("Error: wide output requires pandas. Install with `pip install pandas` and try again.")
    df = pd.DataFrame(rows)
    # create column name as run/tag
    df["run_tag"] = df["run"] + "/" + df["tag"]
    # Group by `step` only (omit wall_time) and aggregate duplicates by mean so there is one line per step
    pivot = df.pivot_table(index="step", columns="run_tag", values="value", aggfunc="mean")
    pivot.reset_index(inplace=True)
    pivot.to_csv(outpath, index=False)


def write_tag_columns_csv(rows: List[dict], outpath: str, tag: str, header_order: List[str] | None = None) -> None:
    """Write a CSV for a single tag where each column is one run (header uses label/description/run)."""
    try:
        import pandas as pd
    except Exception:
        sys.exit("Error: tag-specific wide output requires pandas. Install with `pip install pandas` and try again.")
    df = pd.DataFrame(rows)
    df_tag = df[df["tag"] == tag]
    if df_tag.empty:
        sys.exit(f"Error: no rows found for tag '{tag}'")
    # Decide header field preference: label > description > run
    if df_tag["label"].dropna().astype(bool).any():
        header_field = "label"
        # if header_order provided, respect it (deduplicate to avoid pandas error)
        if header_order:
            # preserve order but ensure uniqueness
            seen = set()
            unique_order = [h for h in header_order if (h not in seen and not seen.add(h))]
            try:
                df_tag[header_field] = pd.Categorical(df_tag[header_field], categories=unique_order, ordered=True)
            except ValueError as e:
                logging.warning("Could not apply ordered header categories (%s); falling back to default ordering: %s", e, unique_order)
                # continue without forcing categorical ordering
    elif df_tag["description"].dropna().astype(bool).any():
        header_field = "description"
    else:
        header_field = "run"
    # Group by `step` only (omit wall_time) and aggregate duplicates by mean so there is one line per step
    pivot = df_tag.pivot_table(index="step", columns=header_field, values="value", aggfunc="mean")
    pivot.reset_index(inplace=True)
    pivot.to_csv(outpath, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export TensorBoard scalars from multiple runs to CSV")
    p.add_argument("paths", nargs="*", help="One or more event directories, event file paths, or glob patterns")
    p.add_argument("--manifest", "-m", help="Path to a JSON or CSV manifest file containing an array of {path, description, label}")
    p.add_argument("--recursive", action="store_true", help="Search directories recursively for event files")
    p.add_argument("--labels", nargs="*", help="Optional labels corresponding to discovered event files (order matches files)")
    p.add_argument("--tag", help="Optional single scalar tag to filter; when used with --wide this produces one column per run (label as header)")
    p.add_argument("--output", "-o", default="tb_scalars.csv", help="Output CSV path")
    p.add_argument("--wide", action="store_true", help="Output wide format (requires pandas)")
    p.add_argument("--quiet", action="store_true", help="Suppress info logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)
    if args.manifest:
        logging.info("Reading manifest from %s", args.manifest)
        entries = read_manifest(args.manifest)
        expanded = expand_manifest_entries(entries, recursive=args.recursive)
        if not expanded:
            logging.error("Manifest provided but no event files found for entries.")
            sys.exit(2)
        event_files = [t[0] for t in expanded]
        labels = [t[1] for t in expanded]
        descriptions = [t[2] for t in expanded]
        logging.info("Found %d event files from manifest entries", len(event_files))
    else:
        logging.info("Discovering event files in: %s", args.paths)
        event_files = find_event_files(args.paths, recursive=args.recursive)
        if not event_files:
            logging.error("No TensorBoard event files found in given paths.")
            sys.exit(2)
        logging.info("Found %d event files", len(event_files))
        labels = args.labels
        descriptions = None
    rows = collect_runs(event_files, labels=labels, descriptions=descriptions)
    # If a single tag was requested, filter or create tag-specific wide file
    if args.tag:
        logging.info("Filtering rows for tag: %s", args.tag)
        if args.wide:
            logging.info("Writing tag-wide CSV to %s", args.output)
            # if labels were provided (or manifest), preserve their order as header order
            header_order = None
            if labels:
                header_order = labels
            write_tag_columns_csv(rows, args.output, args.tag, header_order=header_order)
        else:
            # just write the long CSV filtered to the tag
            filtered = [r for r in rows if r.get("tag") == args.tag]
            if not filtered:
                logging.error("No data for tag %s", args.tag)
                sys.exit(2)
            logging.info("Writing tag-filtered long CSV to %s", args.output)
            write_long_csv(filtered, args.output)
    else:
        if args.wide:
            logging.info("Writing wide CSV to %s", args.output)
            write_wide_csv(rows, args.output)
        else:
            logging.info("Writing long CSV to %s", args.output)
            write_long_csv(rows, args.output)
    logging.info("Done. Wrote %d scalar rows.", len(rows))


if __name__ == "__main__":
    main()

# Example: python3 tb_to_csv.py --manifest pathsMIM.json --tag "IAM-valid_cer" --wide --output validation_loss_by_MIM.csv