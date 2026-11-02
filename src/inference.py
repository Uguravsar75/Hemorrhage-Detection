import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── TTA transform listesi ──────────────────────────────────────────────────────
def get_tta_transforms():
    """8 farklı deterministik augmentation — aynı görüntüde her zaman aynı sonucu üretir."""
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return [
        transforms.Compose(base),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)] + base),
        transforms.Compose([transforms.RandomVerticalFlip(p=1.0)] + base),
        transforms.Compose([transforms.Lambda(lambda img: TF.rotate(img, 10))] + base),
        transforms.Compose([transforms.Lambda(lambda img: TF.rotate(img, -10))] + base),
        transforms.Compose([transforms.CenterCrop(208)] + base),
        transforms.Compose([transforms.Lambda(lambda img: TF.adjust_contrast(img, 1.15))] + base),
        transforms.Compose([transforms.CenterCrop(196)] + base),
    ]


# ── TTA ile tek görüntü tahmini ────────────────────────────────────────────────
@torch.no_grad()
def predict_with_tta(model, img_pil):
    """
    Tek PIL görüntüsü için TTA uygular.
    Döndürür: ortalama olasılık skoru (0-1)
    """
    model.eval()
    model.to(DEVICE)
    tta_transforms = get_tta_transforms()
    batch = torch.stack([tfm(img_pil) for tfm in tta_transforms]).to(DEVICE)
    logits = model(batch).squeeze(1)
    probs = torch.sigmoid(logits).cpu().numpy().tolist()

    return float(np.mean(probs)), probs


# ── Ensemble tahmini ───────────────────────────────────────────────────────────
def ensemble_predict(models_dict, img_pil, weights=None, threshold=0.5):
    """
    Birden fazla modelin TTA tahminlerini ağırlıklı birleştirir.

    models_dict: {'HemorNet': model1, 'ConvNeXt': model2}
    weights    : {'HemorNet': 0.35, 'ConvNeXt': 0.65}
    """
    if weights is None:
        # ConvNeXt daha güçlü — daha fazla ağırlık
        weights = {name: 1.0 / len(models_dict) for name in models_dict}

    total_weight = sum(weights[name] for name in models_dict)
    ensemble_prob = 0.0
    details = {}

    for name, model in models_dict.items():
        prob, tta_probs = predict_with_tta(model, img_pil)
        w = weights.get(name, 1.0) / total_weight
        ensemble_prob += prob * w
        details[name] = {'prob': round(prob, 4), 'tta_probs': [round(p, 4) for p in tta_probs]}

    prediction = 'hemorrhage' if ensemble_prob >= threshold else 'normal'
    return {
        'ensemble_prob' : round(ensemble_prob, 4),
        'prediction'    : prediction,
        'threshold'     : threshold,
        'details'       : details,
    }


# ── Ensemble threshold optimizasyonu ──────────────────────────────────────────
@torch.no_grad()
def optimize_ensemble_threshold(models_dict, val_loader, weights=None):
    """Validation setinde ensemble için en iyi threshold'u bulur."""
    from sklearn.metrics import f1_score
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_preparation import IMAGENET_MEAN, IMAGENET_STD

    all_probs, all_labels = [], []

    for imgs, labels in val_loader:
        batch_probs = []
        for name, model in models_dict.items():
            model.eval()
            model.to(DEVICE)
            logits = model(imgs.to(DEVICE))
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            w = weights.get(name, 1.0) if weights else 1.0
            batch_probs.append(probs * w)

        total_w = sum(weights.values()) if weights else len(models_dict)
        ensemble = np.sum(batch_probs, axis=0) / total_w
        all_probs.extend(ensemble.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"Ensemble threshold: {best_t:.2f}  (F1={best_f1:.4f})")
    return float(best_t), all_probs, all_labels


# ── Görüntü yükleme (GUI ve CLI için) ─────────────────────────────────────────
def load_and_preprocess(img_path):
    """Ham görüntüyü preprocessing pipeline'ından geçirir."""
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_preparation import preprocess_image
    return preprocess_image(img_path)


if __name__ == '__main__':
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.models import HemorNet, ConvNeXtClassifier

    # Dummy test
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    hemornet = HemorNet(pretrained=False)
    convnext = ConvNeXtClassifier(freeze_backbone=False, pretrained=False)

    models_dict = {'HemorNet': hemornet, 'ConvNeXt': convnext}
    weights     = {'HemorNet': 0.35, 'ConvNeXt': 0.65}

    result = ensemble_predict(models_dict, dummy_img, weights=weights, threshold=0.5)
    print(f"\nEnsemble sonucu:")
    print(f"  Tahmin      : {result['prediction']}")
    print(f"  Olasılık    : {result['ensemble_prob']:.4f}")
    for name, d in result['details'].items():
        print(f"  {name}: {d['prob']:.4f}")
