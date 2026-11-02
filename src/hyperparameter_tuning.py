import os
import json
import csv
import torch
import torch.nn as nn
import optuna
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'metrics')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Kısa eğitim (deneme başına) ────────────────────────────────────────────────
def quick_train(model, train_loader, val_loader, pos_weight,
                lr, weight_decay, epochs=8, patience=4):
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = torch.amp.GradScaler(enabled=DEVICE.type == 'cuda')

    best_val_loss = float('inf')
    no_improve    = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Validate
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item() * imgs.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total   += imgs.size(0)

        val_loss /= total
        val_acc   = correct / total

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val_loss, val_acc


# ── Optuna objective ───────────────────────────────────────────────────────────
def make_objective(train_loader, val_loader, pos_weight):
    def objective(trial):
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.models import HemorNet

        lr           = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        dropout      = trial.suggest_float('dropout', 0.2, 0.6)

        model = HemorNet(dropout=dropout)
        val_loss, val_acc = quick_train(
            model, train_loader, val_loader, pos_weight,
            lr=lr, weight_decay=weight_decay, epochs=8, patience=4
        )

        trial.set_user_attr('val_acc', round(val_acc, 4))
        return val_loss

    return objective


# ── Optuna çalıştır ────────────────────────────────────────────────────────────
def run_optuna(train_loader, val_loader, pos_weight, n_trials=30):
    print(f"\nOptuna başlatılıyor — {n_trials} deneme, cihaz: {DEVICE}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(
        make_objective(train_loader, val_loader, pos_weight),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best = study.best_trial
    print(f"\nEn iyi trial #{best.number}")
    print(f"  Val Loss : {best.value:.5f}")
    print(f"  Val Acc  : {best.user_attrs['val_acc']:.4f}")
    print(f"  Params   : {best.params}")

    # Tüm denemeleri CSV'ye kaydet
    csv_path = os.path.join(RESULTS_DIR, 'hp_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['trial', 'val_loss', 'val_acc',
                                                'lr', 'weight_decay', 'dropout'])
        writer.writeheader()
        for t in study.trials:
            if t.value is not None:
                writer.writerow({
                    'trial'       : t.number,
                    'val_loss'    : round(t.value, 5),
                    'val_acc'     : t.user_attrs.get('val_acc', ''),
                    'lr'          : round(t.params.get('lr', 0), 7),
                    'weight_decay': round(t.params.get('weight_decay', 0), 7),
                    'dropout'     : round(t.params.get('dropout', 0), 3),
                })
    print(f"Sonuçlar kaydedildi: {csv_path}")

    # En iyi parametreleri JSON'a kaydet
    best_params = {**best.params, 'val_loss': best.value,
                   'val_acc': best.user_attrs['val_acc']}
    json_path = os.path.join(RESULTS_DIR, 'best_hemornet_params.json')
    with open(json_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"En iyi parametreler: {json_path}")

    return best_params


# ── Manuel grid search tablosu (rapor için) ────────────────────────────────────
def manual_grid_search(train_loader, val_loader, pos_weight):
    """
    Hoca grid search tablosu istedi —
    sabit kombinasyonları dener, sonuçları CSV'ye yazar.
    """
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.models import HemorNet

    grid = [
        {'lr': 1e-3,  'weight_decay': 1e-4, 'dropout': 0.3},
        {'lr': 1e-3,  'weight_decay': 1e-4, 'dropout': 0.5},
        {'lr': 5e-4,  'weight_decay': 1e-4, 'dropout': 0.3},
        {'lr': 5e-4,  'weight_decay': 1e-4, 'dropout': 0.5},
        {'lr': 1e-4,  'weight_decay': 1e-5, 'dropout': 0.3},
        {'lr': 1e-4,  'weight_decay': 1e-5, 'dropout': 0.5},
        {'lr': 3e-4,  'weight_decay': 1e-4, 'dropout': 0.4},
        {'lr': 3e-4,  'weight_decay': 1e-3, 'dropout': 0.4},
    ]

    results = []
    print(f"\nGrid Search — {len(grid)} kombinasyon")
    for i, params in enumerate(grid, 1):
        print(f"  [{i}/{len(grid)}] lr={params['lr']} wd={params['weight_decay']} do={params['dropout']}")
        model = HemorNet(dropout=params['dropout'])
        val_loss, val_acc = quick_train(
            model, train_loader, val_loader, pos_weight,
            lr=params['lr'], weight_decay=params['weight_decay'],
            epochs=8, patience=4
        )
        results.append({**params, 'val_loss': round(val_loss, 5),
                        'val_acc': round(val_acc, 4)})
        print(f"         val_loss={val_loss:.5f}  val_acc={val_acc:.4f}")

    csv_path = os.path.join(RESULTS_DIR, 'grid_search_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lr','weight_decay','dropout',
                                                'val_loss','val_acc'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nGrid search tablosu: {csv_path}")
    return results


if __name__ == '__main__':
    import sys
    _BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, _BASE)
    from src.data_preparation import collect_data, split_data, get_dataloaders

    HEAD_CT = os.path.join(_BASE, 'data', 'raw', 'head_ct')
    RSNA    = os.path.join(_BASE, 'data', 'raw', 'rsna')

    df = collect_data(HEAD_CT, RSNA)
    train_df, val_df, test_df = split_data(df)
    train_loader, val_loader, _, pos_weight = get_dataloaders(
        train_df, val_df, test_df, batch_size=32
    )

    # Hem grid search hem Optuna çalıştır
    manual_grid_search(train_loader, val_loader, pos_weight)
    best_params = run_optuna(train_loader, val_loader, pos_weight, n_trials=30)
