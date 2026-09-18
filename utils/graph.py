import argparse
import csv
import glob
import logging
import os
import sys
from typing import Iterable, List, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
try:
    from tensorboard.backend.event_processing import event_accumulator
except Exception as e:  # pragma: no cover - helpful error for user
    sys.exit("Error: could not import tensorboard's event_accumulator. Install with `pip install tensorboard`.")

manifest_path = f"{Path.home()}/dev/python/DAN/utils/IAM_Contrastive_15.json"
#manifest_path = f"{Path.home()}/dev/python/DAN/utils/IAM_MIM.json"
#manifest_path = f"{Path.home()}/dev/python/DAN/utils/IAM_MIM_CONTR.json"
tag = "IAM-valid_cer"
dataset = "IAM"
#manifest_path = f"{Path.home()}/dev/python/DAN/utils/READ_MIM.json"
manifest_path = f"{Path.home()}/dev/python/DAN/utils/READ_Contrastive_15.json"
tag = "READ_2016-valid_cer"
dataset = "READ_2016"
manifest_path = f"{Path.home()}/dev/python/DAN/utils/RIMES_Contrastive_15.json"
#manifest_path = f"{Path.home()}/dev/python/DAN/utils/RIMES_MIM.json"
tag = "RIMES-valid_cer"
dataset = "RIMES"

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
            out.append({"path": item["path"], "description": item.get("description"), "label": item.get("label"), "seed1": item.get("seed1")})
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

def collect_runs(event_files: List[str], labels: List[str] | None = None, descriptions: List[str] | None = None, seeds1: List[str] | None = None) -> List[dict]:
    """Collect scalar rows from event files.

    Each row includes: run, label, description, tag, step, wall_time, value, file
    """
    rows = []
    if labels and len(labels) != len(event_files):
        raise ValueError("If --labels is used it must have same length as number of discovered event files.")
    if descriptions and len(descriptions) != len(event_files):
        raise ValueError("If descriptions are provided they must match the number of discovered event files.")
    if seeds1 and len(seeds1) != len(event_files):
        raise ValueError("If seeds1 are provided they must match the number of discovered event files.")
    for idx, ef in enumerate(event_files):
        label_val = labels[idx] if labels else None
        run_name = label_val if label_val else (os.path.basename(os.path.dirname(ef)) or os.path.basename(ef))
        desc = descriptions[idx] if descriptions else None
        seed1 = seeds1[idx] if seeds1 else None
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
                "seed1": seed1 or "",
            })
    return rows


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

def expand_manifest_entries(entries: List[dict], recursive: bool) -> List[tuple]:
    """Expand manifest entries to (event_file_path, label, description) tuples.

    If an entry path expands to multiple event files, each gets the same label/description.
    """
    out = []
    for e in entries:
        p = e["path"]
        # check if path p exists
        if not os.path.exists(p):
            logging.warning("Manifest entry path does not exist: %s", p)
        label = e.get("label")
        description = e.get("description")
        seed1 = e.get("seed1")
        found = find_event_files([p], recursive=recursive)
        if not found:
            # keep entry even if no event files found; caller may want to see message
            continue
        for ef in found:
            out.append((ef, label, description, seed1))
    return out

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Reading manifest from %s", manifest_path)
    entries = read_manifest(manifest_path)
    expanded = expand_manifest_entries(entries, recursive=False)
    if not expanded:
        logging.error("Manifest provided but no event files found for entries.")
        sys.exit(2)
    event_files = [t[0] for t in expanded]
    labels = [t[1] for t in expanded]
    descriptions = [t[2] for t in expanded]
    seeds1 = [t[3] for t in expanded]
    logging.info("Found %d event files from manifest entries", len(event_files))
    rows = collect_runs(event_files, labels=labels, descriptions=descriptions, seeds1=seeds1)
    filtered = [r for r in rows if r.get("tag") == tag]
    if not filtered:
        logging.error("No data for tag %s", tag)
        sys.exit(2)
    # Now Create new data structure with epoch, and value grouped by seed1 and calculate for each epoch the mean and standard deviation.

    data_by_seed1 = {}
    labels = {}
    for r in filtered:
        seed1 = r.get("seed1", "")
        label = r.get("label", "")
        #if seed1 != "0":
        #    seed1 = "1"
        if seed1 not in data_by_seed1:
            data_by_seed1[seed1] = {}
            labels[seed1] = label
        if r["step"] not in data_by_seed1[seed1]:
            data_by_seed1[seed1][r["step"]] = []
        data_by_seed1[seed1][r["step"]].append(r["value"])

    logging.info("Data grouped by seed1 and epoch:")
    for seed1, epochs in data_by_seed1.items():
        logging.info("Seed1: %s", seed1)
        for epoch, values in epochs.items():
            logging.info("  Epoch %d: %d values", epoch, len(values))
    for seed in data_by_seed1:
        for epoch in data_by_seed1[seed]:
            values = data_by_seed1[seed][epoch]
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            stddev = variance ** 0.5
            data_by_seed1[seed][epoch] = {"mean": mean, "stddev": stddev}
            logging.info("Seed1: %s, Epoch %d: mean=%.4f, stddev=%.4f", seed, epoch, mean, stddev)
    # Now I want a graph per seed1 with epoch on x-axis and mean value on y-axis, and I want the stddeviation area to be grey area around the mean line.
    plt.figure()

    #labels = {
    #    '0': "Random (mean ± stddev, n = 3)",
    #    '1': "MIM (mean ± stddev, n = 3)",
    #    '2': "Seed 2 (mean ± stddev, n = 3)",
    #    '3': "Seed 3 (mean ± stddev, n = 3)",
    #}

    for seed1 in data_by_seed1.keys(): #['0', '1']:
        epochs = data_by_seed1[seed1]

        x = sorted(epochs.keys())
        y = [epochs[e]["mean"] for e in x]
        s = [epochs[e]["stddev"] for e in x]

        y_lower = [m - s for m, s in zip(y, s)]
        y_upper = [m + s for m, s in zip(y, s)]

        plt.plot(x, y, label=labels[seed1], linewidth=2)
        #plt.fill_between(x, y_lower, y_upper, alpha=0.3)
        plt.fill_between(
            x,
            y_lower,
            y_upper,
            alpha=0.3,
            linewidth=0
        )
    plt.ylim(0.05, 0.11)
    plt.xlim(20, 75)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    plt.xlabel("Epoch")
    plt.ylabel(f"{dataset} validation CER")
    plt.title(f"SSL SimCLR, MIM, vs Random ({dataset})")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()
