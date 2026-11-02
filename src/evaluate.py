import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -- Tahmin toplama -------------------------------------------------------------
@torch.no_grad()
def collect_predictions(model, loader):
    model.eval()
    model.to(DEVICE)
    all_probs, all_labels = [], []

    for imgs, labels in tqdm(loader, desc='Tahmin', leave=False):
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    return np.array(all_probs), np.array(all_labels)


# -- Threshold optimizasyonu (validation setinde) -------------------------------
def find_best_threshold(probs, labels):
    """F1 skorunu maksimize eden threshold'u bulur."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    print(f"En iyi threshold: {best_t:.2f}  (F1={best_f1:.4f})")
    return best_t


# -- Metrik hesaplama -----------------------------------------------------------
def compute_metrics(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-8)

    metrics = {
        'threshold'   : round(float(threshold), 3),
        'accuracy'    : round(accuracy_score(labels, preds), 4),
        'precision'   : round(precision_score(labels, preds, zero_division=0), 4),
        'recall'      : round(recall_score(labels, preds, zero_division=0), 4),
        'specificity' : round(specificity, 4),
        'f1'          : round(f1_score(labels, preds, zero_division=0), 4),
        'auc_roc'     : round(auc(*roc_curve(labels, probs)[:2]), 4),
        'avg_precision': round(average_precision_score(labels, probs), 4),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }
    return metrics, cm


# -- Confusion matrix görseli ---------------------------------------------------
def plot_confusion_matrix(cm, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Hemorrhage'],
                yticklabels=['Normal', 'Hemorrhage'])
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gerçek')
    ax.set_title(f'{model_name} — Confusion Matrix')
    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Kaydedildi: {path}")


# -- ROC eğrisi -----------------------------------------------------------------
def plot_roc_curve(probs, labels, model_name, save_dir, ax=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    standalone  = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC={roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Eğrisi')
    ax.legend()
    if standalone:
        plt.tight_layout()
        path = os.path.join(save_dir, f'{model_name}_roc_curve.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Kaydedildi: {path}")
    return roc_auc


# -- PR eğrisi ------------------------------------------------------------------
def plot_pr_curve(probs, labels, model_name, save_dir, ax=None):
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f'{model_name} (AP={ap:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Eğrisi')
    ax.legend()
    if standalone:
        plt.tight_layout()
        path = os.path.join(save_dir, f'{model_name}_pr_curve.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Kaydedildi: {path}")


# -- Eğitim eğrisi grafiği ------------------------------------------------------
def plot_training_curves(history, model_name, save_dir):
    epochs     = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss   = [h['val_loss'] for h in history]
    train_acc  = [h['train_acc'] for h in history]
    val_acc    = [h['val_acc'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_loss, label='Train Loss')
    axes[0].plot(epochs, val_loss,   label='Val Loss')
    axes[0].set_title(f'{model_name} — Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label='Train Acc')
    axes[1].plot(epochs, val_acc,   label='Val Acc')
    axes[1].set_title(f'{model_name} — Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Kaydedildi: {path}")


# -- Karşılaştırma grafiği ------------------------------------------------------
def plot_model_comparison(results_dict, save_dir):
    """İki modeli yan yana karşılaştırır."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    model_names     = list(results_dict.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    x       = np.arange(len(metrics_to_plot))
    width   = 0.35

    for i, name in enumerate(model_names):
        vals = [results_dict[name][m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, vals, width, label=name)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Karşılaştırması')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Kaydedildi: {path}")


# -- ROC karşılaştırma grafiği --------------------------------------------------
def plot_roc_comparison(probs_dict, labels, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, probs in probs_dict.items():
        plot_roc_curve(probs, labels, name, save_dir, ax=axes[0])
        plot_pr_curve(probs, labels, name, save_dir, ax=axes[1])
    plt.tight_layout()
    path = os.path.join(save_dir, 'roc_pr_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Kaydedildi: {path}")


# -- Ana değerlendirme fonksiyonu -----------------------------------------------
def evaluate_model(model, val_loader, test_loader, model_name,
                   metrics_dir='results/metrics', plots_dir='results/plots'):

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir,   exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  {model_name} değerlendiriliyor...")
    print(f"{'='*50}")

    # Threshold'u validation setinde optimize et
    val_probs, val_labels = collect_predictions(model, val_loader)
    best_threshold = find_best_threshold(val_probs, val_labels)

    # Test setinde final değerlendirme
    test_probs, test_labels = collect_predictions(model, test_loader)
    metrics, cm = compute_metrics(test_probs, test_labels, threshold=best_threshold)

    print(f"\n{'-'*40}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
    print(f"  Avg Prec    : {metrics['avg_precision']:.4f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    print(f"{'-'*40}")

    # Görseller
    plot_confusion_matrix(cm, model_name, plots_dir)
    plot_roc_curve(test_probs, test_labels, model_name, plots_dir)
    plot_pr_curve(test_probs, test_labels, model_name, plots_dir)

    # Metrikleri kaydet
    path = os.path.join(metrics_dir, f'{model_name}_metrics.json')
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrikler kaydedildi: {path}")

    return metrics, test_probs, test_labels


if __name__ == '__main__':
    import sys
    _BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, _BASE)
    from src.data_preparation import collect_data, split_data, get_dataloaders
    from src.models import HemorNet, ConvNeXtClassifier

    HEAD_CT = os.path.join(_BASE, 'data', 'raw', 'head_ct')
    RSNA    = os.path.join(_BASE, 'data', 'raw', 'rsna')
    SAVE    = os.path.join(_BASE, 'saved_models')

    df = collect_data(HEAD_CT, RSNA)
    train_df, val_df, test_df = split_data(df)
    _, val_loader, test_loader, _ = get_dataloaders(
        train_df, val_df, test_df, batch_size=32
    )

    results  = {}
    probs_all = {}

    # HemorNet
    hemornet_path = os.path.join(SAVE, 'hemornet_v5_best.pth')
    backbone = 'b3'
    if not os.path.exists(hemornet_path):
        hemornet_path = os.path.join(SAVE, 'hemornet_best.pth')
        backbone = 'b0'
    hemornet = HemorNet(dropout=0.4, freeze_backbone=False, backbone=backbone, pretrained=False)
    ckpt = torch.load(hemornet_path, map_location=DEVICE, weights_only=False)
    hemornet.load_state_dict(ckpt['model_state'])
    m1, p1, labels = evaluate_model(hemornet, val_loader, test_loader, 'HemorNet')
    results['HemorNet']  = m1
    probs_all['HemorNet'] = p1

    # ConvNeXt
    convnext = ConvNeXtClassifier(freeze_backbone=False, pretrained=False)
    ckpt = torch.load(f'{SAVE}/convnext_best.pth', map_location=DEVICE, weights_only=False)
    convnext.load_state_dict(ckpt['model_state'])
    m2, p2, _ = evaluate_model(convnext, val_loader, test_loader, 'ConvNeXt-Tiny')
    results['ConvNeXt-Tiny']  = m2
    probs_all['ConvNeXt-Tiny'] = p2

    # Karşılaştırma grafikleri
    plot_model_comparison(results, 'results/plots')
    plot_roc_comparison(probs_all, labels, 'results/plots')
