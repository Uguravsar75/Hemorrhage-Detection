# Dataset Directory Structure

Raw CT scan datasets are excluded from git tracking via `.gitignore` due to their size (~253 MB). Small external test samples in `data/web_crawled/` and label metadata (`data/raw/head_ct/labels.csv`) are included in the repository.

## Expected Directory Layout

To train or evaluate from scratch, place the datasets in the following structure:

```text
data/
├── raw/
│   ├── head_ct/
│   │   ├── labels.csv        # Included in repo
│   │   ├── normal/           # 100 axial CT PNG slices (label 0)
│   │   └── hemorrhage/       # 100 axial CT PNG slices (label 1)
│   └── rsna/
│       ├── normal/           # RSNA normal PNG slices
│       └── hemorrhage/       # RSNA hemorrhage PNG slices
└── web_crawled/              # Included in repo (10 external test CT scans + metadata.json)
```

## Sources

1. **Head CT — Hemorrhage Dataset:** [Kaggle — Felipe Kitamura](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage)
2. **RSNA Intracranial Hemorrhage Detection:** [Kaggle RSNA Challenge](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)
3. **Web Crawled External Test Set:** Automatically collected from Wikimedia Commons via `python src/web_crawler.py`.

