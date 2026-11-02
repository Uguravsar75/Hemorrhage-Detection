import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import json
import threading

DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'saved_models')

COLORS = {
    'bg'        : '#1a1a2e',
    'panel'     : '#16213e',
    'accent'    : '#0f3460',
    'normal'    : '#00b894',
    'hemor'     : '#d63031',
    'text'      : '#ffffff',
    'subtext'   : '#b2bec3',
    'bar_bg'    : '#2d3436',
}


# ── Model yükleme ──────────────────────────────────────────────────────────────
def load_models():
    from src.models import HemorNet, ConvNeXtClassifier

    models = {}

    hemornet_path = os.path.join(SAVE_DIR, 'hemornet_v5_best.pth')
    backbone = 'b3'
    if not os.path.exists(hemornet_path):
        hemornet_path = os.path.join(SAVE_DIR, 'hemornet_best.pth')
        backbone = 'b0'
    if os.path.exists(hemornet_path):
        m = HemorNet(dropout=0.4, freeze_backbone=False, backbone=backbone, pretrained=False)
        ckpt = torch.load(hemornet_path, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        models['HemorNet'] = m

    convnext_path = os.path.join(SAVE_DIR, 'convnext_best.pth')
    if os.path.exists(convnext_path):
        m = ConvNeXtClassifier(freeze_backbone=False, pretrained=False)
        ckpt = torch.load(convnext_path, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        models['ConvNeXt-Tiny'] = m

    config_path = os.path.join(SAVE_DIR, 'ensemble_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        weights   = config.get('weights', {'HemorNet': 0.55, 'ConvNeXt-Tiny': 0.45})
        threshold = config.get('threshold', 0.4)
    else:
        weights   = {'HemorNet': 0.55, 'ConvNeXt-Tiny': 0.45}
        threshold = 0.4

    return models, weights, threshold


# ── Ana uygulama ───────────────────────────────────────────────────────────────
class HemorrhageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('CT Hemorrhage Detection — CE480')
        self.geometry('920x680')
        self.minsize(760, 560)
        self.resizable(True, True)
        self.configure(bg=COLORS['bg'])

        self.models    = {}
        self.weights   = {}
        self.threshold = 0.4
        self.img_path  = None

        self._build_ui()
        self._load_models_async()

    # ── UI inşası ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Başlık
        tk.Label(self, text='CT Hemorrhage Detection', font=('Helvetica', 18, 'bold'),
                 bg=COLORS['bg'], fg=COLORS['text']).pack(pady=(18, 2))
        tk.Label(self, text='CE 480 Deep Learning — Project 2',
                 font=('Helvetica', 10), bg=COLORS['bg'],
                 fg=COLORS['subtext']).pack(pady=(0, 14))

        # Ana çerçeve
        main = tk.Frame(self, bg=COLORS['bg'])
        main.pack(fill='both', expand=True, padx=20)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Sol panel — görüntü (pencerenin yarısı)
        left = tk.Frame(main, bg=COLORS['panel'], relief='flat', bd=0)
        left.pack(side='left', fill='both', expand=True, padx=(0, 12))

        self.img_label = tk.Label(left, text='Goruntu secilmedi',
                                  bg=COLORS['panel'], fg=COLORS['subtext'],
                                  font=('Helvetica', 11))
        self.img_label.place(relx=0.5, rely=0.5, anchor='center')

        # Sağ panel — kontroller ve sonuç
        right = tk.Frame(main, bg=COLORS['bg'])
        right.pack(side='left', fill='both', expand=True)

        # Model seçimi
        tk.Label(right, text='Model Selection', font=('Helvetica', 11, 'bold'),
                 bg=COLORS['bg'], fg=COLORS['subtext']).pack(anchor='w', pady=(4, 4))

        self.model_var = tk.StringVar(value='Ensemble (Both)')
        model_options  = ['Ensemble (Both)', 'HemorNet', 'ConvNeXt-Tiny']
        self.model_combo = ttk.Combobox(right, textvariable=self.model_var,
                                         values=model_options, state='readonly',
                                         font=('Helvetica', 11), width=28)
        self.model_combo.pack(anchor='w', pady=(0, 10))

        # Butonlar yan yana
        btn_frame = tk.Frame(right, bg=COLORS['bg'])
        btn_frame.pack(fill='x', pady=(0, 14))
        tk.Button(btn_frame, text='Select Image', command=self._select_image,
                  font=('Helvetica', 11, 'bold'), bg=COLORS['accent'],
                  fg=COLORS['text'], relief='flat', cursor='hand2',
                  padx=10, pady=7).pack(side='left', expand=True, fill='x', padx=(0, 6))
        tk.Button(btn_frame, text='Predict', command=self._predict,
                  font=('Helvetica', 11, 'bold'), bg='#6c5ce7',
                  fg=COLORS['text'], relief='flat', cursor='hand2',
                  padx=10, pady=7).pack(side='left', expand=True, fill='x')

        # Sonuç etiketi
        self.result_label = tk.Label(right, text='—', font=('Helvetica', 22, 'bold'),
                                      bg=COLORS['bg'], fg=COLORS['text'])
        self.result_label.pack(pady=(0, 4))

        self.prob_label = tk.Label(right, text='', font=('Helvetica', 13),
                                    bg=COLORS['bg'], fg=COLORS['subtext'])
        self.prob_label.pack(pady=(0, 8))

        # Olasılık bar
        bar_frame = tk.Frame(right, bg=COLORS['bg'])
        bar_frame.pack(fill='x', pady=(0, 4))
        tk.Label(bar_frame, text='Hemorrhage', font=('Helvetica', 9),
                 bg=COLORS['bg'], fg=COLORS['hemor']).pack(side='left')
        tk.Label(bar_frame, text='Normal', font=('Helvetica', 9),
                 bg=COLORS['bg'], fg=COLORS['normal']).pack(side='right')

        self.canvas_bar = tk.Canvas(right, height=22, bg=COLORS['bar_bg'],
                                     highlightthickness=0)
        self.canvas_bar.pack(fill='x', pady=(0, 10))

        # Threshold çizgisi için canvas'ı yeniden çiz
        self.canvas_bar.bind('<Configure>', lambda e: self._redraw_bar())

        # Model detayları
        self.detail_frame = tk.Frame(right, bg=COLORS['panel'])
        self.detail_frame.pack(fill='x')
        self.detail_labels = {}
        self.detail_bars  = {}
        for name in ['HemorNet', 'ConvNeXt-Tiny']:
            row = tk.Frame(self.detail_frame, bg=COLORS['panel'])
            row.pack(fill='x', padx=8, pady=4)
            tk.Label(row, text=f'{name}:', font=('Helvetica', 10, 'bold'),
                     bg=COLORS['panel'], fg=COLORS['subtext'], width=14,
                     anchor='w').pack(side='left')
            lbl = tk.Label(row, text='—', font=('Helvetica', 10),
                           bg=COLORS['panel'], fg=COLORS['text'])
            lbl.pack(side='left')
            self.detail_labels[name] = lbl
            bar = tk.Canvas(self.detail_frame, height=16, bg=COLORS['bar_bg'],
                            highlightthickness=0)
            bar.pack(fill='x', padx=8, pady=(0, 4))
            self.detail_bars[name] = bar

        # Durum çubuğu
        self.status_var = tk.StringVar(value='Loading models...')
        tk.Label(self, textvariable=self.status_var, font=('Helvetica', 9),
                 bg=COLORS['bg'], fg=COLORS['subtext']).pack(pady=(6, 4))

    def _redraw_bar(self):
        if not hasattr(self, '_last_hemor_prob') or self._last_hemor_prob is None:
            return
        self._draw_bar(self._last_hemor_prob)

    # ── Model yükleme ──────────────────────────────────────────────────────────
    def _load_models_async(self):
        def _worker():
            try:
                models, weights, threshold = load_models()
                self.models, self.weights, self.threshold = models, weights, threshold
                if self.models:
                    names = list(self.models.keys())
                    msg = f'Ready — {", ".join(names)} loaded ({DEVICE})'
                else:
                    msg = 'No models found.'
                self.after(0, lambda: self.status_var.set(msg))
            except Exception as e:
                self.after(0, lambda err=e: self.status_var.set(f'Model yuklenemedi: {err}'))

        threading.Thread(target=_worker, daemon=True).start()

    # ── Görüntü seçimi ─────────────────────────────────────────────────────────
    def _select_image(self):
        path = filedialog.askopenfilename(
            title='Select CT Image',
            filetypes=[('Image Files', '*.png *.jpg *.jpeg *.bmp *.tiff'),
                       ('All Files', '*.*')]
        )
        if not path:
            return
        self.img_path = path
        self._show_image(path)
        self.result_label.config(text='—', fg=COLORS['text'])
        self.prob_label.config(text='')
        self.canvas_bar.delete('all')
        self._last_hemor_prob = None
        for lbl in self.detail_labels.values():
            lbl.config(text='—', fg=COLORS['text'])
        for bar in self.detail_bars.values():
            bar.delete('all')
        self.status_var.set(f'Selected: {os.path.basename(path)}')

    def _show_image(self, path):
        self.update_idletasks()
        panel_w = self.img_label.master.winfo_width() or 380
        panel_h = self.img_label.master.winfo_height() or 500
        w = max(panel_w - 4, 100)
        h = max(panel_h - 4, 100)
        img = Image.open(path).convert('RGB').resize((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.img_label.config(image=photo, text='')
        self.img_label.image = photo
        self.img_label.place(relx=0.5, rely=0.5, anchor='center')

    # ── Tahmin ────────────────────────────────────────────────────────────────
    def _predict(self):
        if not self.img_path:
            messagebox.showwarning('Warning', 'Please select an image first.')
            return
        if not self.models:
            messagebox.showerror('Error', 'No models loaded.')
            return

        self.status_var.set('Running inference...')
        threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        try:
            from src.data_preparation import preprocess_image
            from src.inference import predict_with_tta, ensemble_predict

            img_pil   = preprocess_image(self.img_path)
            selection = self.model_var.get()

            if selection == 'Ensemble (Both)' and len(self.models) >= 2:
                result = ensemble_predict(self.models, img_pil,
                                          weights=self.weights,
                                          threshold=self.threshold)
                prob       = result['ensemble_prob']
                prediction = result['prediction']
                details    = result['details']
            else:
                model_name = selection if selection in self.models else list(self.models.keys())[0]
                prob, _ = predict_with_tta(self.models[model_name], img_pil)
                prediction = 'hemorrhage' if prob >= self.threshold else 'normal'
                details    = {model_name: {'prob': prob}}

            self.after(0, lambda: self._show_result(prediction, prob, details))
            self.after(0, lambda: self.status_var.set('Prediction complete.'))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Prediction Error', str(e)))
            self.after(0, lambda: self.status_var.set('An error occurred.'))

    def _show_result(self, prediction, prob, details=None):
        self._last_hemor_prob = prob
        normal_prob = 1.0 - prob

        if prediction == 'hemorrhage':
            self.result_label.config(text='HEMORRHAGE DETECTED', fg=COLORS['hemor'])
        else:
            self.result_label.config(text='NORMAL', fg=COLORS['normal'])

        self.prob_label.config(
            text=f'Hemorrhage: %{prob*100:.1f}  |  Normal: %{normal_prob*100:.1f}'
        )
        self._draw_bar(prob)

        if details:
            for name, lbl in self.detail_labels.items():
                bar = self.detail_bars[name]
                bar.delete('all')
                if name in details:
                    p = details[name]['prob']
                    lbl.config(text=f"{p*100:.1f}% hemorrhage probability", fg=COLORS['hemor'])
                    w = bar.winfo_width() or 200
                    split = int(w * p)
                    bar.create_rectangle(0, 0, split, 16, fill=COLORS['hemor'], outline='')
                    bar.create_rectangle(split, 0, w, 16, fill=COLORS['normal'], outline='')
                else:
                    lbl.config(text='—', fg=COLORS['text'])

    def _draw_bar(self, hemor_prob):
        self.canvas_bar.delete('all')
        w = self.canvas_bar.winfo_width() or 300
        split = int(w * hemor_prob)
        self.canvas_bar.create_rectangle(0, 0, split, 22, fill=COLORS['hemor'], outline='')
        self.canvas_bar.create_rectangle(split, 0, w, 22, fill=COLORS['normal'], outline='')


def launch_gui():
    app = HemorrhageApp()
    app.mainloop()


if __name__ == '__main__':
    launch_gui()
