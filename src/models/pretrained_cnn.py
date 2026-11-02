import torch
import torch.nn as nn
import torchvision.models as models


class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Tiny transfer learning wrapper.
    2 aşamalı fine-tuning için freeze/unfreeze desteği var.
    """
    def __init__(self, dropout=0.4, freeze_backbone=True, pretrained=True):
        super().__init__()

        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        # Özellik çıkarıcı (features) ve adaptive pool
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        # Orijinal classifier'ın giriş boyutunu al
        in_features = backbone.classifier[-1].in_features  # 768

        # Özel sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

        self._init_classifier()

        if freeze_backbone:
            self.freeze_backbone()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
        print("Backbone donduruldu — sadece classifier eğitiliyor.")

    def unfreeze_last_stages(self, n_stages=2):
        """Son n aşamayı çözer (Phase 2 fine-tuning için)."""
        # ConvNeXt-Tiny features: 0=stem, 1=layer, 2=stage1, 3=layer, 4=stage2, 5=layer, 6=stage3, 7=layer
        # Toplam 8 blok — son n_stages * 2 bloğu aç
        blocks = list(self.features.children())
        unfreeze_from = len(blocks) - n_stages * 2

        for i, block in enumerate(blocks):
            for param in block.parameters():
                param.requires_grad = (i >= unfreeze_from)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Son {n_stages} stage açıldı — eğitilebilir: {trainable:,} parametre")

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)

    def get_trainable_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Toplam parametre  : {total:,}")
        print(f"Eğitilebilir      : {trainable:,}")
        print(f"Dondurulmuş       : {total - trainable:,}")


if __name__ == '__main__':
    # Phase 1 testi
    model = ConvNeXtClassifier(freeze_backbone=True)
    model.get_trainable_params()

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"\nOutput shape: {out.shape}")

    # Phase 2 testi
    print("\n--- Phase 2 ---")
    model.unfreeze_last_stages(n_stages=2)
    model.get_trainable_params()
