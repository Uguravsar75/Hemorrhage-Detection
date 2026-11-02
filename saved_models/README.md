# Pretrained Model Weights

Trained PyTorch checkpoint files (`*.pth`) are excluded from the git repository via `.gitignore` because they exceed GitHub's file size recommendations (`convnext_best.pth` is ~108 MB and `hemornet_v5_best.pth` is ~43 MB).

## Required Files

Place the trained model checkpoints in this directory (`saved_models/`):

| File | Architecture | Size | Status in Repo |
| :--- | :--- | :---: | :--- |
| `hemornet_v5_best.pth` | HemorNet (`EfficientNet-B3`) | ~43 MB | Download from **Releases** or train via `python main.py --mode train` |
| `convnext_best.pth` | `ConvNeXt-Tiny` (2-Phase Fine-Tuned) | ~108 MB | Download from **Releases** or train via `python main.py --mode train` |
| `ensemble_config.json` | Ensemble weights & decision threshold | 92 B | Tracked in repo |

## Verification

After placing the `.pth` files into `saved_models/`, verify your setup by running:

```bash
python main.py --mode check
```

