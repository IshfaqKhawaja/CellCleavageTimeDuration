# Weight-space programming pilot

This experiment edits the YOLO detector's cell-class bias directly. It does
not use backpropagation. The original checkpoint stays unchanged. The script
chooses one bias offset on validation embryos and reports onset timing on
separate held-out embryos.

## Data

Download the Nantes human embryo time-lapse dataset from
<https://zenodo.org/records/7912264>. The record lists a CC BY-NC-SA 4.0
license. Keep downloaded images outside this repository and follow that
license when sharing any derived data.

The pilot needs the central-plane images plus the two small metadata archives:

- `embryo_dataset/` images
- `embryo_dataset_annotations/` phase CSV files
- `embryo_dataset_time_elapsed/` frame-time CSV files

## Run

```sh
python experiments/wsp/validate_cleavage.py \
  --images /path/to/embryo_dataset \
  --annotations /path/to/embryo_dataset_annotations \
  --times /path/to/embryo_dataset_time_elapsed \
  --output experiments/wsp/results.json
```

The script mirrors the project's descending confidence thresholds for cell
counts. It reports matched-event mean absolute timing error and event coverage.
The pilot evaluates cleavage stages `t2` through `t9+`; it omits other phases.

## Pilot result

This run used **41 videos** from about **0.7 GB** of central-plane images. It
split 20/21 videos by embryo group. Validation chose a zero bias offset:
**4.80 h** matched-event MAE versus **4.80 h** for the −1 offset. On holdout,
baseline and selected result were identical: **4.38 h** MAE, with **120/133**
events matched. This WSP edit did not improve the model.

The result is worse than the reported 2.64-frame timing error in the
[RB-NODE study](https://pubmed.ncbi.nlm.nih.gov/42593624/). Its acquisition
interval varies from 10–20 minutes per frame, making that roughly 26–53
minutes. The pilot uses a different split and metric, so this is a failure
signal rather than a head-to-head benchmark.

## Batch-wise GA

`evolve_cell_bias.py` searches 9 direct bias deltas per group of adjacent
8-frame batches. Default group size is 4, reducing the pilot from as many as
504 genes to about 126. Each group vector is shared across videos, then reused
at the corresponding positions in held-out videos. Raw detector logits are
cached once; candidates change cell-class logits and rerun decoding/NMS only.
The GA scores half the validation videos during search, then scores its top
finalists on all validation videos. Test labels only score the locked result.

```sh
python experiments/wsp/evolve_cell_bias.py \
  --images data/nantes_pilot \
  --annotations data/annotations \
  --times data/time_elapsed \
  --population 6 --generations 5 --device auto \
  --output experiments/wsp/ga_results.json
```

Candidate fitness is memoized across generations. `--device auto` prefers
CUDA, then Apple MPS, then CPU; NMS runs on CPU for compatibility. This search
can still overfit the small validation set. Treat results as a pilot; increase
the dataset and repeat with grouped folds before making a performance claim.

The completed pilot GA used MPS, 6 candidates, 5 generations, and 14 groups
(126 genes). It selected zero edits. Held-out performance matched baseline:
**4.38 h** MAE and **120/133** events matched. The best screen fitness did not
improve during search. See `ga_results.json` for per-video details.

## Limits

The bundled result uses 41 streamed videos and a local group split. It cannot
support a state-of-the-art claim. Its timing metric differs from published
16-stage frame accuracy and segmental F1 metrics. Use the official benchmark
split and all eligible videos before making a comparative claim.
