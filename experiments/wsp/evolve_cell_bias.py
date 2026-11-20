"""Genetic search over batch-position-specific cell-class bias edits."""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from validate_cleavage import (
    build_raw_cache,
    detect_cell_scores,
    frame_number,
    labelled_videos,
    load_detector,
    resolve_device,
    score,
)


def canonical(values):
    return tuple(round(float(value), 4) for value in values)


def fitness(metrics, miss_penalty_hours):
    if metrics["mae_hours_matched"] is None:
        return float("inf")
    return metrics["mae_hours_matched"] + miss_penalty_hours * (1.0 - metrics["coverage"])


def grouped_split(videos, seed=23):
    groups = sorted({video[0].name.split("-")[0] for video in videos})
    random.Random(seed).shuffle(groups)
    validation_groups = set(groups[:max(1, len(groups) // 2)])
    validation = [video for video in videos if video[0].name.split("-")[0] in validation_groups]
    test = [video for video in videos if video[0].name.split("-")[0] not in validation_groups]
    if not test:
        raise SystemExit("Need at least two distinct embryo groups for holdout split.")
    return validation, test


def validation_batch_count(videos, batch_size):
    max_batches = 0
    for directory, phases, _ in videos:
        last_labelled_frame = max(end for _, _, end in phases)
        frame_count = sum(
            1 for path in directory.glob("*.jpeg")
            if (number := frame_number(path)) is not None and number <= last_labelled_frame
        )
        max_batches = max(max_batches, (frame_count + batch_size - 1) // batch_size)
    return max_batches


def evolve(model, all_videos, validation, population_size, generations, seed,
           bias_bound, mutation_sigma, mutation_rate, miss_penalty_hours,
           batch_size, batch_group_size, screen_fraction, finalists,
           raw_cache):
    rng = np.random.default_rng(seed)
    batch_count = validation_batch_count(validation, batch_size)
    if batch_count == 0:
        raise SystemExit("No labelled validation frames found.")
    group_count = (batch_count + batch_group_size - 1) // batch_group_size
    gene_count = group_count * 9
    zero = canonical(np.zeros(gene_count))
    direct_scalar = canonical(np.full(gene_count, -1.0))
    population = [zero, direct_scalar]
    while len(population) < population_size:
        population.append(canonical(rng.uniform(-0.6, 0.6, gene_count)))

    screen_videos = list(validation)
    random.Random(seed + 1).shuffle(screen_videos)
    screen_size = max(1, int(np.ceil(len(screen_videos) * screen_fraction)))
    screen_videos = screen_videos[:screen_size]
    screen_records = {}
    history = []

    for generation in range(generations):
        population = list(dict.fromkeys(population))
        while len(population) < population_size:
            population.append(canonical(rng.uniform(-bias_bound, bias_bound, gene_count)))
        new_candidates = [candidate for candidate in population if candidate not in screen_records]
        if new_candidates:
            counts = detect_cell_scores(
                model, all_videos, screen_videos, new_candidates,
                batch_size=batch_size, raw_cache=raw_cache,
                batch_group_size=batch_group_size,
            )
            for candidate in new_candidates:
                metrics = score(counts[candidate], screen_videos)
                screen_records[candidate] = {
                    "fitness": fitness(metrics, miss_penalty_hours), "metrics": metrics
                }
        ranked = sorted(population, key=lambda candidate: screen_records[candidate]["fitness"])
        generation_best = ranked[0]
        history.append({"generation": generation,
                        "screen_fitness": screen_records[generation_best]["fitness"],
                        "screen_metrics": screen_records[generation_best]["metrics"],
                        "best_candidate": generation_best,
                        "unique_candidates_scored": len(screen_records)})
        print(f"GA generation {generation + 1}/{generations}: "
              f"screen fitness {screen_records[generation_best]['fitness']:.4f}; "
              f"{len(screen_records)} unique candidates", flush=True)

        if generation + 1 == generations:
            break

        parent_count = max(2, (population_size + 1) // 2)
        parents = ranked[:parent_count]
        next_population = [ranked[0]]  # elitism
        while len(next_population) < population_size:
            left, right = rng.choice(len(parents), size=2, replace=True)
            p1 = np.asarray(parents[left], dtype=np.float32)
            p2 = np.asarray(parents[right], dtype=np.float32)
            mask = rng.random(gene_count) < 0.5
            child = np.where(mask, p1, p2)
            mutate = rng.random(gene_count) < mutation_rate
            child[mutate] += rng.normal(0.0, mutation_sigma, int(mutate.sum()))
            next_population.append(canonical(np.clip(child, -bias_bound, bias_bound)))
        population = next_population

    screen_ranked = sorted(screen_records, key=lambda candidate: screen_records[candidate]["fitness"])
    full_candidates = list(dict.fromkeys([zero] + screen_ranked[:finalists]))
    full_counts = detect_cell_scores(
        model, all_videos, validation, full_candidates,
        batch_size=batch_size, raw_cache=raw_cache,
        batch_group_size=batch_group_size,
    )
    full_records = {}
    for candidate in full_candidates:
        metrics = score(full_counts[candidate], validation)
        full_records[candidate] = {
            "fitness": fitness(metrics, miss_penalty_hours), "metrics": metrics
        }
    best_candidate = min(full_candidates, key=lambda candidate: full_records[candidate]["fitness"])
    return best_candidate, full_records[best_candidate], history, full_records, group_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="runs/weights/best.pt")
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--times", required=True)
    parser.add_argument("--output", default="experiments/wsp/ga_results.json")
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-group-size", type=int, default=4,
                        help="Share each bias vector across this many adjacent frame batches.")
    parser.add_argument("--screen-fraction", type=float, default=0.5)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--bias-bound", type=float, default=2.0)
    parser.add_argument("--mutation-sigma", type=float, default=0.35)
    parser.add_argument("--mutation-rate", type=float, default=0.02)
    parser.add_argument("--miss-penalty-hours", type=float, default=8.0)
    args = parser.parse_args()

    if (args.population < 4 or args.generations < 1 or args.batch_size < 1
            or args.batch_group_size < 1 or not 0 < args.screen_fraction <= 1
            or args.finalists < 1):
        raise SystemExit("Check population, generations, batch sizes, screen fraction, and finalists.")
    torch.set_num_threads(2)
    cv2.setLogLevel(0)
    images_root = Path(args.images)
    videos = labelled_videos(images_root, Path(args.annotations), Path(args.times))
    if len(videos) < 2:
        raise SystemExit("Need at least two labelled embryo videos.")
    validation, test = grouped_split(videos)
    device = resolve_device(args.device)
    model = load_detector(args.weights, device)
    print(f"Caching validation raw logits on {device}...", flush=True)
    validation_cache = build_raw_cache(model, validation, args.batch_size)
    print(f"Cached {len(validation_cache)} validation batches; starting GA.", flush=True)

    best, best_record, history, full_records, group_count = evolve(
        model, videos, validation, args.population, args.generations, args.seed,
        args.bias_bound, args.mutation_sigma, args.mutation_rate,
        args.miss_penalty_hours, args.batch_size, args.batch_group_size,
        args.screen_fraction, args.finalists, validation_cache,
    )
    baseline = canonical(np.zeros(group_count * 9))
    test_candidates = [baseline] if best == baseline else [baseline, best]
    test_cache = build_raw_cache(model, test, args.batch_size)
    test_counts = detect_cell_scores(model, videos, test, test_candidates,
                                     batch_size=args.batch_size, raw_cache=test_cache,
                                     batch_group_size=args.batch_group_size)
    baseline_test = score(test_counts[baseline], test)
    selected_test = score(test_counts[best], test)

    results = {
        "checkpoint": args.weights,
        "class_names": model.names,
        "method": "GA over direct cell-class bias edits, shared by grouped 8-frame positions across videos; no gradient updates",
        "genes": f"{group_count} groups x 9 bias deltas (3 anchors x 3 detection scales)",
        "batch_size_frames": args.batch_size,
        "batch_group_size": args.batch_group_size,
        "batch_parameterization": "Each video resets batch position to zero. One vector is learned per group of adjacent positions on validation videos and reused in held-out videos. Groups beyond validation coverage use zero bias.",
        "device": str(device),
        "validation_groups": [v[0].name for v in validation],
        "test_groups": [v[0].name for v in test],
        "search": {
            "population": args.population,
            "generations": args.generations,
            "screen_fraction": args.screen_fraction,
            "finalists_full_validation": args.finalists,
            "seed": args.seed,
            "bias_bound": args.bias_bound,
            "mutation_sigma": args.mutation_sigma,
            "mutation_rate": args.mutation_rate,
            "miss_penalty_hours": args.miss_penalty_hours,
        },
        "history": history,
        "selected_biases_by_batch_group_scale_anchor": np.asarray(best).reshape(group_count, 3, 3).tolist(),
        "validation_fitness": best_record["fitness"],
        "validation_metrics": best_record["metrics"],
        "full_validation_finalists": {
            str(index): {"fitness": record["fitness"], "metrics": record["metrics"]}
            for index, record in enumerate(full_records.values())
        },
        "baseline_test": baseline_test,
        "selected_test": selected_test,
        "test_delta_mae_hours": selected_test["mae_hours_matched"] - baseline_test["mae_hours_matched"],
    }

    direct_path = Path("experiments/wsp/pilot_results.json")
    if direct_path.exists():
        direct = json.loads(direct_path.read_text())
        results["direct_wsp_reference"] = {
            "selected_bias": direct.get("selected_bias"),
            "baseline_test": direct.get("baseline_test"),
            "selected_test": direct.get("selected_test"),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"best_biases_by_batch_group": results["selected_biases_by_batch_group_scale_anchor"],
                      "validation_fitness": results["validation_fitness"],
                      "baseline_test": baseline_test,
                      "selected_test": selected_test,
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
