import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ── Temel blok ─────────────────────────────────────────────────────────────────
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ── Residual blok ──────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(channels, channels),
            nn.Dropout2d(dropout),
            ConvBNReLU(channels, channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


# ── Channel Attention ──────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


# ── Spatial Attention ──────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


# ── CBAM: Channel + Spatial Attention ─────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return self.spatial(self.channel(x))


# ── HemorNet (EfficientNet-B0/B3 backbone + custom classifier) ────────────────
_EFFICIENTNET_CONFIGS = {
    'b0': (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280, 224),
    'b3': (tv_models.efficientnet_b3, tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1, 1536, 300),
}

class HemorNet(nn.Module):
    """
    EfficientNet-B0/B3 pretrained backbone + optional CBAM + custom classifier.
    backbone: 'b0' (hafif) veya 'b3' (daha guclu, 300x300 input)
    freeze_backbone=False: tek fazli egitim, tum parametreler acik
    use_cbam=False: True ise backbone cikisina CBAM (Channel+Spatial Attention) uygular
    """
    def __init__(self, dropout=0.5, freeze_backbone=False, backbone='b0',
                 pretrained=True, use_cbam=False):
        super().__init__()

        build_fn, weights, feat_dim, _ = _EFFICIENTNET_CONFIGS[backbone]
        efficientnet = build_fn(weights=weights if pretrained else None)

        self.backbone = efficientnet.features
        self.attention = CBAM(feat_dim) if use_cbam else None
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feat_dim, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self._backbone_frozen = False
        if freeze_backbone:
            self._freeze_backbone()

        self._init_new_layers()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        # BN running stats'larinin guncellenmesini engelle
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
        self._backbone_frozen = True

    def _init_new_layers(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        # Frozen parametrelere sahip BN katmanlarini her zaman eval'da tut
        # (Phase 1 ve Phase 2'de kısmen frozen bloklar icin gecerli)
        if mode:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    params = list(m.parameters())
                    if params and not params[0].requires_grad:
                        m.eval()
        return self

    def unfreeze_last_blocks(self, n=2):
        """Sadece EfficientNet'in son n blogunu ac — catastrophic forgetting onler."""
        blocks = list(self.backbone.children())
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def forward(self, x):
        x = self.backbone(x)
        if self.attention is not None:
            x = self.attention(x)
        x = self.gap(x)
        return self.classifier(x)


# ── Model özeti ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = HemorNet(dropout=0.5)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape : {out.shape}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params : {total:,}")
    print(f"Trainable    : {trainable:,}")
