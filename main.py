"""
BM 480 Derin Öğrenme — Proje 2
Kafa CT Kanama Tespiti

Kullanım:
  python main.py --mode train
  python main.py --mode evaluate
  python main.py --mode hp_search
  python main.py --mode gui
  python main.py --mode all
"""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_BASE    = os.path.dirname(os.path.abspath(__file__))
HEAD_CT  = os.path.join(_BASE, 'data', 'raw', 'head_ct')
RSNA     = os.path.join(_BASE, 'data', 'raw', 'rsna')
SAVE_DIR = os.path.join(_BASE, 'saved_models')
LOG_DIR  = os.path.join(_BASE, 'results', 'metrics')
PLOT_DIR = os.path.join(_BASE, 'results', 'plots')

# Ensemble ağırlıkları
ENSEMBLE_WEIGHTS = {'HemorNet': 0.55, 'ConvNeXt-Tiny': 0.45}


# ── Veri yükleme ───────────────────────────────────────────────────────────────
def load_data(batch_size=32):
    from src.data_preparation import collect_data, split_data, get_dataloaders
    df = collect_data(HEAD_CT, RSNA)
    train_df, val_df, test_df = split_data(df)
    return get_dataloaders(train_df, val_df, test_df, batch_size=batch_size)


# ── Eğitim ────────────────────────────────────────────────────────────────────
def run_train():
    from src.models import HemorNet, ConvNeXtClassifier
    from src.train import train_model, train_convnext_two_phase

    train_loader, val_loader, test_loader, pos_weight = load_data(batch_size=32)

    # 1. HemorNet (EfficientNet-B3)
    print("\n" + "="*55)
    print("  1/2 — HemorNet (EfficientNet-B3) eğitimi")
    print("="*55)

    # En iyi parametreler varsa yükle
    params_path = os.path.join(LOG_DIR, 'best_hemornet_params.json')
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        lr           = params.get('lr', 1e-4)
        weight_decay = params.get('weight_decay', 1e-5)
        dropout      = params.get('dropout', 0.4)
        print(f"Optuna parametreleri yüklendi: lr={lr:.6f} wd={weight_decay:.6f} do={dropout:.2f}")
    else:
        lr, weight_decay, dropout = 1e-4, 1e-5, 0.4

    hemornet = HemorNet(dropout=dropout, freeze_backbone=False, backbone='b3')
    train_model(hemornet, train_loader, val_loader, pos_weight,
                model_name='hemornet_v5', save_dir=SAVE_DIR, log_dir=LOG_DIR,
                lr=lr, epochs=60, patience=15, weight_decay=weight_decay, warmup_epochs=5)

    # 2. ConvNeXt
    print("\n" + "="*55)
    print("  2/2 — ConvNeXt-Tiny eğitimi (2 aşamalı)")
    print("="*55)
    convnext = ConvNeXtClassifier(freeze_backbone=True)
    train_convnext_two_phase(convnext, train_loader, val_loader, pos_weight,
                              save_dir=SAVE_DIR, log_dir=LOG_DIR)


# ── Değerlendirme ──────────────────────────────────────────────────────────────
def run_evaluate():
    from src.models import HemorNet, ConvNeXtClassifier
    from src.evaluate import (evaluate_model, plot_model_comparison,
                               plot_roc_comparison, plot_training_curves)
    from src.inference import optimize_ensemble_threshold
    import json

    _, val_loader, test_loader, _ = load_data(batch_size=32)
    results, probs_all = {}, {}
    test_labels = None

    # HemorNet
    hemornet_path = os.path.join(SAVE_DIR, 'hemornet_v5_best.pth')
    backbone = 'b3'
    if not os.path.exists(hemornet_path):
        hemornet_path = os.path.join(SAVE_DIR, 'hemornet_best.pth')
        backbone = 'b0'
    if os.path.exists(hemornet_path):
        hemornet = HemorNet(dropout=0.4, freeze_backbone=False, backbone=backbone, pretrained=False)
        ckpt = torch.load(hemornet_path, map_location=DEVICE, weights_only=False)
        hemornet.load_state_dict(ckpt['model_state'])
        m, p, test_labels = evaluate_model(
            hemornet, val_loader, test_loader, 'HemorNet',
            metrics_dir=LOG_DIR, plots_dir=PLOT_DIR
        )
        results['HemorNet']   = m
        probs_all['HemorNet'] = p

        log_path = os.path.join(LOG_DIR, 'hemornet_v5_training_log.json')
        if not os.path.exists(log_path):
            log_path = os.path.join(LOG_DIR, 'hemornet_best_training_log.json')
        if os.path.exists(log_path):
            with open(log_path) as f:
                plot_training_curves(json.load(f), 'HemorNet', PLOT_DIR)

    # ConvNeXt
    convnext_path = os.path.join(SAVE_DIR, 'convnext_best.pth')
    if os.path.exists(convnext_path):
        convnext = ConvNeXtClassifier(freeze_backbone=False, pretrained=False)
        ckpt = torch.load(convnext_path, map_location=DEVICE, weights_only=False)
        convnext.load_state_dict(ckpt['model_state'])
        m, p, test_labels = evaluate_model(
            convnext, val_loader, test_loader, 'ConvNeXt-Tiny',
            metrics_dir=LOG_DIR, plots_dir=PLOT_DIR
        )
        results['ConvNeXt-Tiny']   = m
        probs_all['ConvNeXt-Tiny'] = p

        log_path = os.path.join(LOG_DIR, 'convnext_best_training_log.json')
        if os.path.exists(log_path):
            with open(log_path) as f:
                plot_training_curves(json.load(f), 'ConvNeXt-Tiny', PLOT_DIR)

    if len(results) == 2 and test_labels is not None:
        plot_model_comparison(results, PLOT_DIR)
        plot_roc_comparison(probs_all, test_labels, PLOT_DIR)

        # Ensemble threshold optimizasyonu
        models_dict = {}
        if 'HemorNet' in results:
            models_dict['HemorNet'] = hemornet
        if 'ConvNeXt-Tiny' in results:
            models_dict['ConvNeXt-Tiny'] = convnext

        best_t, _, _ = optimize_ensemble_threshold(
            models_dict, val_loader, weights=ENSEMBLE_WEIGHTS
        )
        config = {'weights': ENSEMBLE_WEIGHTS, 'threshold': best_t}
        with open(os.path.join(SAVE_DIR, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Ensemble config kaydedildi.")

    print("\nDeğerlendirme tamamlandı.")


# ── Hiperparametre araması ─────────────────────────────────────────────────────
def run_hp_search():
    from src.hyperparameter_tuning import manual_grid_search, run_optuna
    train_loader, val_loader, _, pos_weight = load_data(batch_size=32)
    manual_grid_search(train_loader, val_loader, pos_weight)
    run_optuna(train_loader, val_loader, pos_weight, n_trials=30)


# ── GUI ───────────────────────────────────────────────────────────────────────
def run_gui():
    from app.app import launch_gui
    launch_gui()


# ── Durum kontrolü ─────────────────────────────────────────────────────────────
def run_check():
    print("\n-- Proje Durum Kontrolu --")
    checks = {
        'Hocanın veri seti (normal)' : 'data/raw/head_ct/normal',
        'Hocanın veri seti (hemor)'  : 'data/raw/head_ct/hemorrhage',
        'RSNA (normal)'              : 'data/raw/rsna/normal',
        'RSNA (hemorrhage)'          : 'data/raw/rsna/hemorrhage',
        'HemorNet modeli'            : 'saved_models/hemornet_v5_best.pth' if os.path.exists(os.path.join(_BASE, 'saved_models/hemornet_v5_best.pth')) else 'saved_models/hemornet_best.pth',
        'ConvNeXt modeli'            : 'saved_models/convnext_best.pth',
        'Ensemble config'            : 'saved_models/ensemble_config.json',
        'Metrik sonuçları'           : 'results/metrics/HemorNet_v5_metrics.json' if os.path.exists(os.path.join(_BASE, 'results/metrics/HemorNet_v5_metrics.json')) else 'results/metrics/hp_results.csv',
    }
    base = os.path.dirname(os.path.abspath(__file__))
    for name, path in checks.items():
        full = os.path.join(base, path)
        if os.path.isdir(full):
            count = len(os.listdir(full))
            print(f"  [OK]  {name}: {count} dosya")
        elif os.path.exists(full):
            size = os.path.getsize(full) / 1024 / 1024
            print(f"  [OK]  {name}: {size:.1f} MB")
        else:
            print(f"  [--]  {name}: bulunamadi")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Yok (CPU)'
    print(f"\n  GPU: {gpu}")
    print("------------------------------------------\n")


# ── Argparse ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BM480 Proje 2 — CT Kanama Tespiti')
    parser.add_argument('--mode', type=str, default='check',
                        choices=['check', 'train', 'evaluate', 'hp_search', 'gui', 'all'],
                        help='Çalıştırma modu')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    if args.mode == 'check':
        run_check()
    elif args.mode == 'train':
        run_train()
    elif args.mode == 'evaluate':
        run_evaluate()
    elif args.mode == 'hp_search':
        run_hp_search()
    elif args.mode == 'gui':
        run_gui()
    elif args.mode == 'all':
        run_check()
        run_hp_search()
        run_train()
        run_evaluate()
