import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Early Stopping ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float('inf')
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return True   # iyileşme var → kaydet
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False  # iyileşme yok


# ── Tek epoch eğitim ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc='Train', leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


# ── Validation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, desc='Val  ', leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


# ── Genel eğitim döngüsü ───────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, pos_weight,
                model_name, save_dir='saved_models', log_dir='results/metrics',
                lr=1e-3, epochs=50, patience=10, weight_decay=1e-4, warmup_epochs=0):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    model = model.to(DEVICE)
    # label_smoothing ile BCEWithLogitsLoss: 0.1 smoothing logit patlamasını önler
    class SmoothBCEWithLogitsLoss(nn.Module):
        def __init__(self, pos_weight=None, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        def forward(self, logits, targets):
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
            return self.bce(logits, targets)
    criterion = SmoothBCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE), smoothing=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    # Warmup + CosineAnnealing: ilk warmup_epochs'ta LR lineer artar, sonra cosine ile azalır
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    scaler       = None  # AMP kapatıldı — GTX 1650 ile NaN sorununu önler
    early_stop   = EarlyStopping(patience=patience)
    best_path    = os.path.join(save_dir, f'{model_name}_best.pth')
    history      = []

    print(f"\n{'='*55}")
    print(f"  {model_name} eğitimi — {DEVICE}")
    print(f"  LR: {lr}  |  Epochs: {epochs}  |  Patience: {patience}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        vl_loss, vl_acc = validate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        improved = early_stop.step(vl_loss)

        if improved:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': vl_loss, 'val_acc': vl_acc}, best_path)

        history.append({
            'epoch': epoch, 'train_loss': round(tr_loss, 5),
            'train_acc': round(tr_acc, 4), 'val_loss': round(vl_loss, 5),
            'val_acc': round(vl_acc, 4),
            'lr': round(optimizer.param_groups[0]['lr'], 8)
        })

        flag = '*' if improved else ' '
        print(f"[{epoch:03d}/{epochs}] {flag} "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f} | "
              f"{elapsed:.1f}s")

        if early_stop.should_stop:
            print(f"\nEarly stopping — {patience} epoch iyileşme yok.")
            break

    log_path = os.path.join(log_dir, f'{model_name}_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nEn iyi model kaydedildi : {best_path}")
    print(f"Eğitim logu kaydedildi  : {log_path}")
    return history


# ── ConvNeXt 2 aşamalı eğitim ─────────────────────────────────────────────────
def train_convnext_two_phase(model, train_loader, val_loader, pos_weight,
                              save_dir='saved_models', log_dir='results/metrics'):

    print("\n[*] ConvNeXt Phase 1 -- Backbone dondurulmus, sadece classifier")
    history_p1 = train_model(
        model, train_loader, val_loader, pos_weight,
        model_name='convnext_phase1',
        save_dir=save_dir, log_dir=log_dir,
        lr=1e-3, epochs=15, patience=7, weight_decay=1e-4
    )

    # En iyi Phase 1 ağırlıklarını yükle
    ckpt = torch.load(os.path.join(save_dir, 'convnext_phase1_best.pth'),
                      map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    print("\n[*] ConvNeXt Phase 2 -- Son 2 stage acildi, dusuk LR ile fine-tune")
    model.unfreeze_last_stages(n_stages=2)
    history_p2 = train_model(
        model, train_loader, val_loader, pos_weight,
        model_name='convnext',
        save_dir=save_dir, log_dir=log_dir,
        lr=1e-5, epochs=50, patience=15, weight_decay=1e-5
    )

    return history_p1, history_p2


if __name__ == '__main__':
    import sys
    _BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, _BASE)
    from src.data_preparation import collect_data, split_data, get_dataloaders
    from src.models import HemorNet, ConvNeXtClassifier

    HEAD_CT = os.path.join(_BASE, 'data', 'raw', 'head_ct')
    RSNA    = os.path.join(_BASE, 'data', 'raw', 'rsna')

    df = collect_data(HEAD_CT, RSNA)
    train_df, val_df, test_df = split_data(df)
    train_loader, val_loader, test_loader, pos_weight = get_dataloaders(
        train_df, val_df, test_df, batch_size=32
    )

    # HemorNet eğitimi
    hemornet = HemorNet(dropout=0.5)
    train_model(hemornet, train_loader, val_loader, pos_weight,
                model_name='hemornet_best', lr=1e-3, epochs=50, patience=10)

    # ConvNeXt 2 aşamalı eğitim
    convnext = ConvNeXtClassifier(freeze_backbone=True)
    train_convnext_two_phase(convnext, train_loader, val_loader, pos_weight)
