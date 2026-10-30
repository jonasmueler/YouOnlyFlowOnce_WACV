# Reproducing the paper step by step

All commands run from the repository root with the virtual environment active and
`YOFO_DATA_ROOT` pointing at the dataset folder of the setting (or `--data_root`).

## 1. Checkpoints

```bash
python download_checkpoints.py            # all five MG-Prior checkpoints, 42 MB each
python download_checkpoints.py --verify_only
```

Files land in `checkpoints/<dataset>/<split>/gaussian/backbone.pt`, where the scripts
look for them by default. `checkpoints/manifest.json` holds the Google Drive ids and
SHA256 sums.

## 2. MH-NFPG rows (Tables 1, 3, 4)

One setting at a time; the three MM-Fi splits need `--split`.

```bash
# MM-Fi cross-subject (YOFO_DATA_ROOT = .../MMFi_Dataset)
python extract_features.py --dataset mmfi --split cross_subject
python train_flow.py --dataset mmfi --split cross_subject
python evaluate.py --model flow --dataset mmfi --split cross_subject \
    --checkpoint runs/mmFI/cross_subject/norm_flow_graph/run_001/final_model.pt

# mmRadPose (YOFO_DATA_ROOT = .../mmRadPose_pointclouds)
python extract_features.py --dataset mmradpose
python train_flow.py --dataset mmradpose
python evaluate.py --model flow --dataset mmradpose \
    --checkpoint runs/mmRadPose/default/norm_flow_graph/run_001/final_model.pt

# mRI (YOFO_DATA_ROOT = .../mri_dataset/aligned_data)
python extract_features.py --dataset mri
python train_flow.py --dataset mri
python evaluate.py --model flow --dataset mri \
    --checkpoint runs/MRI/protocol2/norm_flow_graph/run_001/final_model.pt
```

`reproduce.py --dataset D [--split S]` runs the same four steps and prints the
overall metrics.

Approximate cost on one NVIDIA A40: feature extraction 15 to 45 minutes per setting
(MM-Fi windows are the slowest), flow training 20 to 60 minutes, evaluation a few
minutes.

## 3. MG-Prior rows

The MG-Prior of these rows is trained as a standalone model and evaluated on its own:

```bash
python train_backbone.py --standalone --dataset mmradpose
python evaluate.py --model gaussian --dataset mmradpose \
    --checkpoint runs/mmRadPose/default/gaussian/run_001/final_model.pt
```

The MG-Prior draws 200 hypotheses per frame from its latent Gaussian; their mean is the
point estimate. Without `--checkpoint`, `evaluate.py --model gaussian` evaluates the
released weights of step 1.

## 4. MG-Prior + DiffPose rows

```bash
python train_diffpose.py --dataset mmradpose            # trains on the extracted features
python evaluate.py --model diffpose --dataset mmradpose \
    --checkpoint runs/mmRadPose/default/diffusion_multi_hyp/run_001/final_model.pt
```

## 5. PointTransformer and mmDiff rows

One command trains the backbone (5 epochs), extracts its coarse poses and features and
trains the denoiser (100 epochs); the evaluation then reports the three rows at once.

```bash
python train_mmdiff.py --dataset mmradpose              # or --phase backbone|extract|diffusion
python evaluate.py --model mmdiff --dataset mmradpose \
    --checkpoint runs/mmRadPose/default/mmdiff/run_001/final_model.pt
python tools/mmdiff_table_rows.py results/mmRadPose/eval_mmdiff_default_*.json
```

The backbone phase is the slow part (several hours on MM-Fi); both phases write
`epoch_NNN.pt` after every epoch and `--resume_from <run folder>` continues an
interrupted run. The leave-one-environment-out rows of Table 2 use `--dataset mmfi
--split loeo_E01` to `loeo_E04`.

## 6. Runtime benchmark (Table 5, Table S3)

```bash
python benchmark.py --output benchmark_results.json
```

Writes `benchmark_results.json` and `benchmark_results.md`.

## Output folders

| Folder | Content |
|---|---|
| `checkpoints/` | released MG-Prior checkpoints |
| `features/<dataset>/<split>/` | `train_features.pt`, `test_features.pt` |
| `runs/<dataset>/<split>/<model>/run_NNN/` | `final_model.pt`, `hyperparameters.json` |
| `results/<dataset>/` | `eval_<model>_<split>_<timestamp>.json` |

`<dataset>` is spelled `mmFI`, `mmRadPose` and `MRI` in these folders, as in the
feature files.
