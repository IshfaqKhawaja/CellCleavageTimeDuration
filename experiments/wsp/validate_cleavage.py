"""Direct-weight intervention benchmark for cleavage event timing.

WSP edits detector cell-class biases directly. Raw detector logits are cached
so candidates can be decoded without rerunning the detector head. Held-out
embryos report onset timing against phase annotations.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


EVENTS = {f"t{i}": i for i in range(2, 9)}
EVENTS["t9+"] = 9


def read_phases(path):
    phases = []
    with path.open(newline="") as handle:
        for name, start, end in csv.reader(handle):
            if name in EVENTS:
                phases.append((name, int(start), int(end)))
    return sorted(phases, key=lambda row: row[1])


def read_times(path):
    times = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            times[int(row["frame_index"])] = float(row["time"])
    return times


def labelled_videos(images_root, annotations_root, times_root):
    videos = []
    for directory in sorted(p for p in images_root.iterdir() if p.is_dir()):
        phase_path = annotations_root / f"{directory.name}_phases.csv"
        time_path = times_root / f"{directory.name}_timeElapsed.csv"
        if phase_path.exists() and time_path.exists():
            phases = read_phases(phase_path)
            if phases:
                videos.append((directory, phases, read_times(time_path)))
    return videos


def resolve_device(device="auto"):
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise SystemExit("MPS requested but unavailable.")
    return torch.device(device)


def load_detector(weights, device="cpu"):
    # This legacy YOLO checkout imports seaborn only for optional plotting.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    os.environ.setdefault("YOLOV5_CONFIG_DIR", "/private/tmp/cleavage-yolo-config")
    sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
    checkpoint = torch.load(weights, map_location="cpu")
    model = checkpoint.get("ema") or checkpoint["model"]
    for module in model.modules():
        if isinstance(module, torch.nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    model.float().eval().to(device)
    return model


def set_cell_bias(model, offset, baseline_biases):
    """Directly edit raw class-1 biases; accept scalar or scale-anchor deltas."""
    head = model.model[-1]
    if np.isscalar(offset):
        deltas = np.full((head.nl, head.na), float(offset), dtype=np.float32)
    else:
        deltas = np.asarray(offset, dtype=np.float32).reshape(head.nl, head.na)
    with torch.no_grad():
        for scale, (conv, baseline) in enumerate(zip(head.m, baseline_biases)):
            conv.bias.copy_(baseline)
            shaped = conv.bias.view(head.na, head.no)
            shaped[:, 5 + 1].add_(torch.as_tensor(deltas[scale], device=conv.bias.device))


def image_tensor(path, size=416):
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    rgb = canvas[:, :, ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0


def frame_number(path):
    match = re.search(r"_RUN(\d+)\.jpe?g$", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def build_raw_cache(model, videos_to_process, batch_size=8):
    """Run backbone and detection convolutions once; keep raw logits on host RAM."""
    device = next(model.parameters()).device
    cache = []
    for directory, phases, _ in videos_to_process:
        last_labelled_frame = max(end for _, _, end in phases)
        image_paths = []
        for path in directory.glob("*.jpeg"):
            number = frame_number(path)
            if number is not None and number <= last_labelled_frame:
                image_paths.append((number, path))
        image_paths.sort()
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            frame_numbers = []
            tensors = []
            for number, path in batch_paths:
                tensor = image_tensor(path)
                if tensor is not None:
                    frame_numbers.append(number)
                    tensors.append(tensor)
            if not tensors:
                continue
            with torch.no_grad():
                raw_outputs = model(torch.stack(tensors).to(device))[1]
            cache.append((directory.name, start // batch_size, frame_numbers,
                          [raw.detach().to("cpu") for raw in raw_outputs]))
    return cache


def decode_raw_head(raw_outputs, head, offset):
    """Decode cached raw head logits after changing only cell-class logits."""
    deltas = np.asarray(offset, dtype=np.float32).reshape(head.nl, head.na)
    # Cached logits live on CPU to cap accelerator memory use. Decode there too;
    # only the one-time backbone/head pass uses the selected accelerator.
    device = raw_outputs[0].device
    predictions = []
    with torch.no_grad():
        for scale, raw in enumerate(raw_outputs):
            y = raw.to(device).clone()
            y[..., 6] += torch.as_tensor(deltas[scale], device=device).view(1, head.na, 1, 1)
            y = y.sigmoid()
            grid = head.grid[scale].to(device)
            stride = head.stride[scale].to(device)
            anchor_grid = head.anchor_grid[scale].to(device)
            if head.inplace:
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
                y[..., 2:4] = (y[..., 2:4] * 2.0) ** 2 * anchor_grid
            else:
                xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
                wh = (y[..., 2:4] * 2.0) ** 2 * anchor_grid.view(1, head.na, 1, 1, 2)
                y = torch.cat((xy, wh, y[..., 4:]), -1)
            predictions.append(y.view(y.shape[0], -1, head.no))
    return torch.cat(predictions, 1)


def detect_cell_scores(model, videos, videos_to_process, offsets, batch_size=8,
                       conf_thres=0.03, raw_cache=None, batch_group_size=1):
    from utils.general import non_max_suppression

    head = model.model[-1]
    if raw_cache is None:
        raw_cache = build_raw_cache(model, videos_to_process, batch_size)
    counts_by_offset = {offset: {} for offset in offsets}
    selected_video_names = {directory.name for directory, _, _ in videos_to_process}
    for name in selected_video_names:
        for offset in offsets:
            counts_by_offset[offset][name] = {}
    for video_name, batch_index, frame_numbers, raw_outputs in raw_cache:
        if video_name not in selected_video_names:
            continue
        for offset in offsets:
            candidate = np.asarray(offset, dtype=np.float32)
            if candidate.ndim == 0:
                batch_offset = np.full(9, float(candidate), dtype=np.float32)
            elif candidate.size == 9:
                batch_offset = candidate
            else:
                group_index = batch_index // batch_group_size
                gene_start = group_index * 9
                batch_offset = candidate[gene_start:gene_start + 9]
                # A held-out video may extend beyond validation coverage.
                if batch_offset.size < 9:
                    batch_offset = np.zeros(9, dtype=np.float32)
            pred = decode_raw_head(raw_outputs, head, batch_offset)
            pred[..., 0:4] *= 416
            # torchvision NMS is not consistently implemented on MPS.
            detections = non_max_suppression(pred.float().cpu(), conf_thres, 0.45, max_det=300)
            for number, det in zip(frame_numbers, detections):
                if det is None or not len(det):
                    counts_by_offset[offset][video_name][number] = []
                else:
                    cells = det[:, 5].long() == 1
                    counts_by_offset[offset][video_name][number] = det[cells, 4].tolist()
    return counts_by_offset


def project_style_onsets(cell_scores):
    """Mirror count_boxes' descending confidence thresholds and first hits."""
    thresholds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.65, 0.7, 0.8]
    cell_threshold = thresholds.pop()
    found = {"t0": None, "t1": None}
    for frame in sorted(cell_scores):
        cell_count = sum(conf >= cell_threshold for conf in cell_scores[frame])
        key = f"t{cell_count}"
        if len(thresholds) > 0 and key not in found:
            found[key] = frame
            cell_threshold = thresholds.pop()
    return found


def score(counts_by_video, videos):
    event_errors = []
    per_video = {}
    for directory, phases, times in videos:
        predicted_frames = project_style_onsets(counts_by_video.get(directory.name, {}))
        errors = []
        missed = []
        for name, start, _ in phases:
            prediction_key = "t9" if name == "t9+" else name
            predicted_frame = predicted_frames.get(prediction_key)
            if predicted_frame is not None and predicted_frame in times and start in times:
                error = abs(times[predicted_frame] - times[start])
                errors.append({"event": name, "error_hours": error})
                event_errors.append(error)
            else:
                missed.append(name)
        per_video[directory.name] = {"matched": errors, "missed": missed}
    mae = float(np.mean(event_errors)) if event_errors else None
    total = sum(len(phases) for _, phases, _ in videos)
    return {"mae_hours_matched": mae, "matched_events": len(event_errors),
            "total_events": total, "missed_events": total - len(event_errors),
            "coverage": len(event_errors) / total if total else 0.0, "per_video": per_video}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/weights/best.pt")
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--times", required=True)
    parser.add_argument("--output", default="experiments/wsp/pilot_results.json")
    parser.add_argument("--biases", type=float, nargs="+", default=[-1.0, 0.0])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    torch.set_num_threads(2)
    cv2.setLogLevel(0)
    os.environ.setdefault("YOLOV5_CONFIG_DIR", "/private/tmp/cleavage-yolo-config")
    images_root = Path(args.images)
    annotations_root = Path(args.annotations)
    times_root = Path(args.times)
    videos = labelled_videos(images_root, annotations_root, times_root)
    if len(videos) < 2:
        raise SystemExit("Need at least two labelled embryo videos.")

    groups = sorted({video[0].name.split("-")[0] for video in videos})
    random.Random(23).shuffle(groups)
    validation_groups = set(groups[:max(1, len(groups) // 2)])
    validation = [video for video in videos if video[0].name.split("-")[0] in validation_groups]
    test = [video for video in videos if video[0].name.split("-")[0] not in validation_groups]
    if not test:
        raise SystemExit("Need at least two distinct embryo groups for a holdout split.")
    device = resolve_device(args.device)
    model = load_detector(args.weights, device)
    results = {"checkpoint": args.weights, "class_names": model.names,
               "validation_videos": [v[0].name for v in validation],
               "test_videos": [v[0].name for v in test], "bias_trials": {}}

    validation_counts = detect_cell_scores(model, videos, validation, args.biases)
    for bias in args.biases:
        val_score = score(validation_counts[bias], validation)
        results["bias_trials"][str(bias)] = {"validation": val_score}

    selected = min(args.biases, key=lambda bias: results["bias_trials"][str(bias)]["validation"]["mae_hours_matched"]
                   if results["bias_trials"][str(bias)]["validation"]["mae_hours_matched"] is not None else float("inf"))
    results["selected_bias"] = selected
    test_offsets = [0.0] if selected == 0.0 else [0.0, selected]
    test_counts = detect_cell_scores(model, videos, test, test_offsets)
    results["baseline_test"] = score(test_counts[0.0], test)
    results["selected_test"] = score(test_counts[selected], test)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"validation_videos": results["validation_videos"],
                      "test_videos": results["test_videos"], "selected_bias": selected,
                      "baseline_test": results["baseline_test"],
                      "selected_test": results["selected_test"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
