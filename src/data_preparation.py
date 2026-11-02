import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── CLAHE ──────────────────────────────────────────────────────────────────────
def apply_clahe(img_np):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_np)

# ── Preprocessing pipeline ─────────────────────────────────────────────────────
def preprocess_image(img_path, size=224):
    """
    PNG CT goruntuleri icin:
    1. Grayscale yukle
    2. 0-255 normalize (DICOM kaynakli 16-bit icin)
    3. CLAHE — kontrast iyilestirme
    4. 224x224 resize + RGB
    """
    img = Image.open(img_path).convert('L')
    img_np = np.array(img, dtype=np.float32)

    if img_np.max() > 255:
        img_np = (img_np / img_np.max()) * 255.0
    elif img_np.max() > 0:
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6) * 255.0

    img_np = apply_clahe(img_np.astype(np.uint8))

    img_pil = Image.fromarray(img_np).resize((size, size), Image.Resampling.LANCZOS)
    img_rgb = Image.merge('RGB', [img_pil, img_pil, img_pil])
    return img_rgb

# ── Veri toplama ───────────────────────────────────────────────────────────────
def collect_data(head_ct_dir, rsna_dir):
    """
    Tüm görüntü yollarını ve etiketlerini toplar.
    Kaynak bilgisini de saklar (head_ct vs rsna).
    """
    data = []

    for label_name, label in [('normal', 0), ('hemorrhage', 1)]:
        # Hocanın seti
        hct_path = os.path.join(head_ct_dir, label_name)
        if os.path.exists(hct_path):
            for f in os.listdir(hct_path):
                if f.lower().endswith('.png'):
                    data.append({
                        'path': os.path.join(hct_path, f),
                        'label': label,
                        'source': 'head_ct'
                    })

        # RSNA seti
        rsna_path = os.path.join(rsna_dir, label_name)
        if os.path.exists(rsna_path):
            for f in os.listdir(rsna_path):
                if f.lower().endswith('.png'):
                    data.append({
                        'path': os.path.join(rsna_path, f),
                        'label': label,
                        'source': 'rsna'
                    })

    df = pd.DataFrame(data)
    print(f"\nToplam veri: {len(df)}")
    print(df.groupby(['source', 'label']).size().to_string())
    return df

# ── Stratified split ───────────────────────────────────────────────────────────
def split_data(df, rsna_max=None, rsna_in_val=False, mixed_split=False, rsna_train_headct_test=False):
    """
    rsna_in_val=False (varsayilan): val/test sadece head_ct — ConvNeXt icin
    rsna_in_val=True             : val/test'e RSNA da eklenir — HemorNet icin
                                   model her iki domain'i ogrenip her ikisinde test edilir
    rsna_max: kac RSNA goruntusu kullanilacak (None = hepsi)
    """
    # rsna_train_headct_test: RSNA ile eğit, head_ct ile test et (domain generalization)
    if rsna_train_headct_test:
        rsna_df    = df[df['source'] == 'rsna'].reset_index(drop=True)
        head_ct_df = df[df['source'] == 'head_ct'].reset_index(drop=True)
        train_df, val_df = train_test_split(
            rsna_df, test_size=0.2, stratify=rsna_df['label'], random_state=SEED
        )
        test_df = head_ct_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        print(f"\nTrain : {len(train_df)} RSNA ({int(train_df['label'].sum())} hemorrhage)")
        print(f"Val   : {len(val_df)} RSNA  ({int(val_df['label'].sum())} hemorrhage)")
        print(f"Test  : {len(test_df)} head_ct ({int(test_df['label'].sum())} hemorrhage)")
        return train_df, val_df, test_df

    # mixed_split: domain ayirimi olmadan tum veriyi karistir
    if mixed_split:
        train_df, temp_df = train_test_split(
            df, test_size=0.2, stratify=df['label'], random_state=SEED
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
        print(f"\nTrain : {len(train_df)} ({int(train_df['label'].sum())} hemorrhage)")
        print(f"Val   : {len(val_df)}   ({int(val_df['label'].sum())} hemorrhage)")
        print(f"Test  : {len(test_df)}   ({int(test_df['label'].sum())} hemorrhage)")
        return train_df, val_df, test_df

    head_ct_df = df[df['source'] == 'head_ct'].reset_index(drop=True)
    rsna_df    = df[df['source'] == 'rsna'].reset_index(drop=True)

    if rsna_max == 0:
        rsna_df = rsna_df.iloc[:0].reset_index(drop=True)
    elif rsna_max is not None and len(rsna_df) > rsna_max:
        rsna_df, _ = train_test_split(
            rsna_df, train_size=rsna_max, stratify=rsna_df['label'], random_state=SEED
        )
        rsna_df = rsna_df.reset_index(drop=True)

    # head_ct 50/25/25 bol
    hct_train, hct_temp = train_test_split(
        head_ct_df, test_size=0.5, stratify=head_ct_df['label'], random_state=SEED
    )
    hct_val, hct_test = train_test_split(
        hct_temp, test_size=0.5, stratify=hct_temp['label'], random_state=SEED
    )

    if rsna_in_val and len(rsna_df) > 0:
        # RSNA'yi de 60/20/20 bol — her uc sete dagit
        rsna_train, rsna_temp = train_test_split(
            rsna_df, test_size=0.4, stratify=rsna_df['label'], random_state=SEED
        )
        rsna_val, rsna_test = train_test_split(
            rsna_temp, test_size=0.5, stratify=rsna_temp['label'], random_state=SEED
        )
        train_df = pd.concat([hct_train, rsna_train], ignore_index=True)
        val_df   = pd.concat([hct_val,   rsna_val],   ignore_index=True)
        test_df  = pd.concat([hct_test,  rsna_test],  ignore_index=True)
    else:
        train_df = pd.concat([hct_train, rsna_df], ignore_index=True)
        val_df   = hct_val.reset_index(drop=True)
        test_df  = hct_test.reset_index(drop=True)

    print(f"\nTrain : {len(train_df)} ({int(train_df['label'].sum())} hemorrhage)")
    print(f"Val   : {len(val_df)}   ({int(val_df['label'].sum())} hemorrhage)")
    print(f"Test  : {len(test_df)}   ({int(test_df['label'].sum())} hemorrhage)")

    return train_df, val_df, test_df

# ── Dataset sınıfı ─────────────────────────────────────────────────────────────
class CTDataset(Dataset):
    def __init__(self, df, transform=None, preprocess=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.preprocess:
            img = preprocess_image(row['path'])
        else:
            img = Image.open(row['path']).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row['label'], dtype=torch.float32)
        return img, label

# ── CT'ye özel custom augmentation sınıfları ───────────────────────────────────
class AddGaussianNoise:
    """Scanner gürültüsünü simüle eder."""
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


class RandomGamma:
    """
    Farklı CT cihazlarının farklı kontrast üretmesini simüle eder.
    Gamma < 1 → parlaklaştır, Gamma > 1 → karartır.
    """
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, img_pil):
        gamma = random.uniform(*self.gamma_range)
        return TF.adjust_gamma(img_pil, gamma)


class RandomElasticDeformation:
    """
    Beyin dokusunun doğal şekil varyasyonunu simüle eder.
    Hafif elastic distortion — kanama bölgesini bozmadan çeşitlilik katar.
    """
    def __init__(self, alpha=30, sigma=4, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p     = p

    def __call__(self, img_pil):
        if random.random() > self.p:
            return img_pil
        img_np = np.array(img_pil, dtype=np.float32)
        h, w   = img_np.shape[:2]

        dx = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32),
            (0, 0), self.sigma
        ) * self.alpha
        dy = cv2.GaussianBlur(
            (np.random.rand(h, w) * 2 - 1).astype(np.float32),
            (0, 0), self.sigma
        ) * self.alpha

        x, y    = np.meshgrid(np.arange(w), np.arange(h))
        map_x   = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y   = np.clip(y + dy, 0, h - 1).astype(np.float32)

        if img_np.ndim == 3:
            channels = [cv2.remap(img_np[:, :, c], map_x, map_y,
                                   cv2.INTER_LINEAR) for c in range(img_np.shape[2])]
            result = np.stack(channels, axis=2)
        else:
            result = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


# ── Transforms ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transform(img_size=224):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
        transforms.RandomResizedCrop(img_size, scale=(0.80, 1.0)),
        RandomGamma(gamma_range=(0.75, 1.25)),
        RandomElasticDeformation(alpha=30, sigma=4, p=0.3),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        AddGaussianNoise(std=0.02),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.0)),
    ])

def get_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ── DataLoader fabrikası ───────────────────────────────────────────────────────
def get_dataloaders(train_df, val_df, test_df, batch_size=32, img_size=224):
    train_ds = CTDataset(train_df, transform=get_train_transform(img_size))
    val_ds   = CTDataset(val_df,   transform=get_val_transform(img_size))
    test_ds  = CTDataset(test_df,  transform=get_val_transform(img_size))

    # Sınıf ağırlığı hesapla (pos_weight için)
    n_neg = (train_df['label'] == 0).sum()
    n_pos = (train_df['label'] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    print(f"\nPos weight: {pos_weight.item():.3f}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader, pos_weight


if __name__ == '__main__':
    _BASE   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    HEAD_CT = os.path.join(_BASE, 'data', 'raw', 'head_ct')
    RSNA    = os.path.join(_BASE, 'data', 'raw', 'rsna')

    df = collect_data(HEAD_CT, RSNA)
    train_df, val_df, test_df = split_data(df)

    train_loader, val_loader, test_loader, pos_weight = get_dataloaders(
        train_df, val_df, test_df, batch_size=32
    )

    # Sanity check
    imgs, labels = next(iter(train_loader))
    print(f"\nBatch shape : {imgs.shape}")
    print(f"Label sample: {labels[:8]}")
    print(f"Pixel range : [{imgs.min():.2f}, {imgs.max():.2f}]")
