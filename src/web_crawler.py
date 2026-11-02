import os
import sys
import json
import time
import requests
from PIL import Image
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SAVE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'web_crawled')
META_FILE = os.path.join(SAVE_DIR, 'metadata.json')
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; BM480-Project/1.0; academic use)'
}

# ── Wikimedia Commons API ──────────────────────────────────────────────────────
WIKIMEDIA_QUERIES = [
    'intracranial hemorrhage CT scan axial',
    'brain hemorrhage computed tomography',
    'subdural hematoma CT scan',
    'epidural hematoma CT axial',
    'subarachnoid hemorrhage CT brain',
    'normal brain CT scan axial',
    'intracerebral hemorrhage CT',
]

def search_wikimedia(query, max_results=4):
    """Wikimedia Commons'ta arama yapar, görüntü URL'lerini döner."""
    url = 'https://commons.wikimedia.org/w/api.php'
    params = {
        'action'      : 'query',
        'generator'   : 'search',
        'gsrsearch'   : f'{query} filetype:bitmap',
        'gsrnamespace': 6,
        'gsrlimit'    : max_results,
        'prop'        : 'imageinfo',
        'iiprop'      : 'url|mime',
        'format'      : 'json',
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = r.json()
        pages = data.get('query', {}).get('pages', {})
        results = []
        for page in pages.values():
            info = page.get('imageinfo', [{}])[0]
            mime = info.get('mime', '')
            if mime in ('image/jpeg', 'image/png'):
                results.append({
                    'url'   : info.get('url', ''),
                    'title' : page.get('title', ''),
                    'source': 'wikimedia',
                    'query' : query,
                })
        return results
    except Exception as e:
        print(f"  Wikimedia hata: {e}")
        return []


# ── Görüntü indir ve işle ─────────────────────────────────────────────────────
def process_pil_image(img_pil, size=224):
    """PIL görüntüsünü temp dosya olmadan preprocessing'den geçirir."""
    import cv2
    import numpy as np

    img_np = np.array(img_pil.convert('L'), dtype=np.float32)
    if img_np.max() > 255:
        img_np = (img_np / img_np.max()) * 255.0
    elif img_np.max() > 0:
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6) * 255.0

    # CLAHE
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_np = clahe.apply(img_np.astype(np.uint8))

    img_pil = Image.fromarray(img_np).resize((size, size), Image.LANCZOS)
    return Image.merge('RGB', [img_pil, img_pil, img_pil])


def download_and_process(url, filename):
    """
    Thumbnail URL ile indir, preprocessing pipeline'ından geçir, kaydet.
    """
    try:
        # Wikimedia thumbnail URL — rate limit'i önler
        thumb_url = url
        if 'wikimedia.org/wikipedia/commons/' in url:
            parts = url.split('/commons/')
            if len(parts) == 2:
                fname = parts[1].split('/')[-1]
                thumb_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{fname}?width=512"

        r = requests.get(thumb_url, headers=HEADERS, timeout=15)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content))
        w, h = img.size
        if w < 80 or h < 80:
            return False

        processed = process_pil_image(img, size=224)
        save_path  = os.path.join(SAVE_DIR, filename)
        processed.save(save_path)
        return True
    except Exception as e:
        print(f"  Indirme hatasi ({filename}): {e}")
        return False


# ── Ana crawler ───────────────────────────────────────────────────────────────
def run_crawler(target=25):
    """
    Wikimedia'dan beyin CT görselleri toplar.
    target: kaç görsel hedefleniyor
    """
    print(f"\nWeb crawler basliyor — hedef: {target} gorsel")
    print(f"Kayit klasoru: {SAVE_DIR}\n")

    metadata  = []
    collected = 0
    seen_urls = set()

    for query in WIKIMEDIA_QUERIES:
        if collected >= target:
            break

        print(f"Arama: '{query}'")
        results = search_wikimedia(query, max_results=5)

        for item in results:
            if collected >= target:
                break
            if item['url'] in seen_urls or not item['url']:
                continue

            seen_urls.add(item['url'])
            filename = f"web_{collected+1:03d}.png"
            print(f"  [{collected+1}/{target}] {item['title'][:60]}")

            success = download_and_process(item['url'], filename)
            if success:
                metadata.append({
                    'filename': filename,
                    'source'  : item['source'],
                    'query'   : item['query'],
                    'url'     : item['url'],
                    'title'   : item['title'],
                })
                collected += 1
                print(f"  Kaydedildi: {filename}")
            else:
                print(f"  Atlandi.")

            time.sleep(1.5)

    # Metadata kaydet
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nToplanan gorsel : {collected}/{target}")
    print(f"Metadata        : {META_FILE}")
    return metadata


# ── Mevcut görselleri göster ───────────────────────────────────────────────────
def show_crawled_samples(n=6):
    """Toplanan görsellerden örnek gösterir (matplotlib ile)."""
    import matplotlib.pyplot as plt

    files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.png')][:n]
    if not files:
        print("Gorsel bulunamadi.")
        return

    cols = min(3, len(files))
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = [axes] if rows * cols == 1 else axes.flatten()

    # Metadata oku
    meta_dict = {}
    if os.path.exists(META_FILE):
        with open(META_FILE, encoding='utf-8') as f:
            for item in json.load(f):
                meta_dict[item['filename']] = item.get('query', '')

    for ax, fname in zip(axes, files):
        img = Image.open(os.path.join(SAVE_DIR, fname))
        ax.imshow(img, cmap='gray')
        ax.set_title(meta_dict.get(fname, fname)[:40], fontsize=8)
        ax.axis('off')

    for ax in axes[len(files):]:
        ax.axis('off')

    plt.suptitle('Web Crawler — Toplanan CT Gorselleri', fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'plots', 'web_crawled_samples.png')
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Ornek grafik kaydedildi: {save_path}")


if __name__ == '__main__':
    metadata = run_crawler(target=25)
    if metadata:
        show_crawled_samples(n=6)
