# modality_aware_fl

Heterogeneous multimodal federated learning for explainable, resource-aware tuberculosis (TB) screening. Research prototype; not a medical device and not clinically validated.

## Design
- Single-machine simulation of 5 federated clients: Montgomery CXR, Shenzhen CXR, two clinical-feature splits (Dryad), and one CT client.
- Each client holds one modality. Encoders are averaged only within the same modality; the fusion head is averaged across all clients.
- Fusion handles missing modalities with learned null embeddings and a presence-conditioned gate.
- Explainability and uncertainty: Grad-CAM (CXR), MC-Dropout, heuristic image-quality flags, clinician/patient views.
- Evaluation: stratified 70/15/15 per-client splits, thresholds chosen on validation only, bootstrap CIs on held-out test sets.

## Reproduce
```
python -m src.run_study --seeds 42 43 44 --rounds 5
python -m src.evaluate_robustness --checkpoint checkpoints/study_seed42.pt --seed 42
python -m src.analyze_checkpoint --checkpoint checkpoints/study_seed42.pt --seed 42
python -m src.make_figures
```
Datasets are not in the repository (see `configs/data_sources.yaml`). Figures and tables are written to the git-ignored `outputs/` folder.

## Results (3 seeds, 5 rounds, held-out test sets, mean over 5 clients)
| Method | Balanced acc. | AUROC |
|---|---|---|
| Modality-aware FedAvg | 0.738 ± 0.017 | 0.812 ± 0.024 |
| FedAvg, unweighted | 0.751 ± 0.023 | 0.828 ± 0.041 |
| Local-only | 0.716 ± 0.035 | 0.833 ± 0.048 |
| Centralized (not private) | 0.716 ± 0.036 | 0.826 ± 0.021 |

Differences between methods are within seed-to-seed variation; no method is conclusively better.

## Limitations
- Cohorts are unpaired (no patient has all modalities); CT has 6 test cases and clinical splits have 4-5 positives each.
- The "unweighted FedAvg" baseline is not a true modality-unaware baseline.
- Clinical-model calibration is poor; CXR models degrade under noise and strong blur.
- Single-machine simulation; no secure aggregation or differential privacy; no clinical validation.