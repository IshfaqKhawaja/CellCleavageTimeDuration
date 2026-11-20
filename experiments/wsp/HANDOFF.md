# WSP / GA handoff

## Current state

- Baseline checkpoint: `runs/weights/best.pt` (tracked; 14 MB).
- Public data: Nantes human embryo time-lapse set, [Zenodo record 7912264](https://zenodo.org/records/7912264), CC BY-NC-SA 4.0.
- Local sample: 41 videos, about 697 MB. Local workspace links point into `/private/tmp`; they are ignored by Git and will not exist on another machine.
- Annotation and elapsed-time metadata are also local links and are not in Git.
- Direct WSP selected scalar bias 0. Holdout: 4.3775 h MAE, 120/133 events matched, 90.2% coverage.
- Batch GA selected zero edits. It searched 24 unique candidates over five generations, with 14 groups × 9 genes (126 genes). Holdout matched baseline: 4.3775 h MAE, 120/133 events, 90.2% coverage.
- GA artifacts and full per-video metrics: `ga_results.json`; direct WSP results: `pilot_results.json`.
- GA raw-logit cache ran on Apple MPS; candidate decode and NMS ran on CPU. Baseline was unchanged. OpenCV emitted truncated-JPEG warnings during data loading; image integrity still needs a pass.

## Continue on the GPU desktop

1. Pull `main` and ensure `runs/weights/best.pt` is present.
2. Put the image sample and metadata on that machine. Do not commit raw data. Full source record is linked above; the current sample is only 41 videos.
3. Validate JPEG files and count unreadable/truncated frames before using results.
4. Confirm CUDA is visible in the installed PyTorch environment. Then run:

```sh
python experiments/wsp/evolve_cell_bias.py \
  --images /path/to/nantes_pilot \
  --annotations /path/to/embryo_dataset_annotations \
  --times /path/to/embryo_dataset_time_elapsed \
  --device cuda \
  --population 12 --generations 10 \
  --output experiments/wsp/ga_cuda_results.json
```

5. Run `validate_cleavage.py` with the same grouped split and compare direct WSP, GA+WSP, and baseline. Keep test embryos locked until parameter selection finishes.
6. If GA still selects zero edits, test global and grouped parameterizations with repeated embryo-group folds. Only compare to published results on a matching official split and metric.

The current GA speedups are: one cached detector pass, grouped batch genes (126 instead of up to 504), memoized candidate fitness, and subset screening before full-validation finalist scoring. `--device auto` selects CUDA first; NMS remains on CPU for compatibility.

## Abstract drafts

### Weight-Space Programming

We investigate whether cleavage-stage timing in human embryo time-lapse images can be improved by directly editing a pretrained detector’s cell-class biases, without gradient-based retraining. Cell confidence scores are converted into cell counts and cleavage-stage onset times, then compared with annotated phase timing. A scalar bias intervention is selected on validation embryos and evaluated on held-out embryo groups using matched-event mean absolute error and event coverage. In this pilot on a partial public dataset, WSP did not improve over the baseline detector. Confirmation requires the full dataset and official benchmark split.

### Genetic Algorithm with WSP

We investigate a genetic algorithm for selecting direct cell-class bias edits in a pretrained embryo-image detector. Each candidate encodes bias changes across groups of consecutive frames and detection scales. To reduce search cost, detector logits are cached once; candidates are screened on part of the validation set, then finalists are scored on all validation embryos. The selected parameters are evaluated on held-out embryo groups using cleavage-event timing error and event coverage. In this pilot, GA selected zero bias edits and matched baseline performance. Larger-scale evaluation is needed to determine whether the approach can improve timing.
