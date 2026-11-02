import os
import sys
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preparation import preprocess_image, get_train_transform, get_val_transform
from src.models import HemorNet, ConvNeXtClassifier
from src.inference import predict_with_tta, ensemble_predict


class TestHemorrhagePipeline(unittest.TestCase):
    def setUp(self):
        self.dummy_np = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        self.dummy_pil = Image.fromarray(self.dummy_np).convert('RGB').resize((224, 224))

    def test_clahe_preprocessing(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            Image.fromarray(self.dummy_np).save(tmp.name)
            tmp_path = tmp.name
        try:
            out = preprocess_image(tmp_path, size=224)
            self.assertEqual(out.size, (224, 224))
            self.assertEqual(out.mode, 'RGB')
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_transforms(self):
        train_tfm = get_train_transform(224)
        val_tfm = get_val_transform(224)
        t_out = train_tfm(self.dummy_pil)
        v_out = val_tfm(self.dummy_pil)
        self.assertEqual(tuple(t_out.shape), (3, 224, 224))
        self.assertEqual(tuple(v_out.shape), (3, 224, 224))

    def test_hemornet_forward_and_cbam(self):
        x = torch.randn(2, 3, 224, 224)
        model_b0 = HemorNet(backbone='b0', pretrained=False, use_cbam=False)
        model_cbam = HemorNet(backbone='b0', pretrained=False, use_cbam=True)
        out1 = model_b0(x)
        out2 = model_cbam(x)
        self.assertEqual(tuple(out1.shape), (2, 1))
        self.assertEqual(tuple(out2.shape), (2, 1))

    def test_convnext_two_phase_freeze_unfreeze(self):
        x = torch.randn(2, 3, 224, 224)
        model = ConvNeXtClassifier(freeze_backbone=True, pretrained=False)
        frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.unfreeze_last_stages(n_stages=2)
        unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(unfrozen_trainable, frozen_trainable)
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 1))

    def test_deterministic_tta_and_ensemble(self):
        m1 = HemorNet(backbone='b0', pretrained=False)
        m2 = ConvNeXtClassifier(freeze_backbone=False, pretrained=False)
        p1_first, _ = predict_with_tta(m1, self.dummy_pil)
        p1_second, _ = predict_with_tta(m1, self.dummy_pil)
        self.assertAlmostEqual(p1_first, p1_second, places=6)

        res = ensemble_predict(
            {'HemorNet': m1, 'ConvNeXt-Tiny': m2},
            self.dummy_pil,
            weights={'HemorNet': 0.55, 'ConvNeXt-Tiny': 0.45},
            threshold=0.40,
        )
        self.assertIn(res['prediction'], ('normal', 'hemorrhage'))
        self.assertTrue(0.0 <= res['ensemble_prob'] <= 1.0)


if __name__ == '__main__':
    unittest.main()
