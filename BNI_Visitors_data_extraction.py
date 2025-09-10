import pandas as pd
import numpy as np
import os, re
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import streamlit as st

# Set environment variables before importing PaddleOCR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Global OCR instance
ocr_instance = None

try:
    from paddleocr import PaddleOCR


    # Initialize with cloud-friendly settings
    @st.cache_resource
    def load_ocr():
        return PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False,
            use_gpu=False,
            enable_mkldnn=False,
            cpu_threads=1
        )


    ocr_instance = load_ocr()

except Exception as e:
    st.error(f"Failed to initialize PaddleOCR: {e}")
    ocr_instance = None

# ========= CONFIG =========
COLUMNS = ['Name', 'Company Name', 'Category', 'Invited by', 'Fees', 'Payment Mode', 'Date']


def extract_data_from_image(image_path):
    """Extract data from image using PaddleOCR"""

    if ocr_instance is None:
        raise RuntimeError("PaddleOCR not initialized properly")

    try:
        # Use the cached OCR instance
        result = ocr_instance.ocr(image_path, cls=True)

        if not result or not result[0]:
            raise RuntimeError("No OCR result returned")

        # Extract OCR data
        ocr_items = []
        for line in result[0]:
            bbox, (text, confidence) = line
            if not text or confidence < 0.5:  # Skip low confidence results
                continue

            # Calculate bounding box center and dimensions
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            ocr_items.append({
                "text": text.strip(),
                "score": float(confidence),
                "x": (x_min + x_max) / 2.0,
                "y": (y_min + y_max) / 2.0,
                "xmin": x_min, "xmax": x_max,
                "ymin": y_min, "ymax": y_max,
                "w": x_max - x_min,
                "h": y_max - y_min
            })

        if not ocr_items:
            raise RuntimeError("No valid OCR items parsed")

        return process_ocr_data(ocr_items)

    except Exception as e:
        st.error(f"OCR processing failed: {str(e)}")
        return pd.DataFrame(columns=COLUMNS)  # Return empty DataFrame


def process_ocr_data(ocr_items):
    """Process OCR items and return structured DataFrame"""

    # ========= Helpers =========
    num_re = re.compile(r'^\d{1,2}\.?$')
    moneyish_re = re.compile(r'^\s*[\d,.]{3,}\s*$')
    cash_tokens = {"cash", "done", "online", "upi", "od", "cheque", "dd"}

    def is_row_number(txt):
        t = txt.strip().replace(' ', '')
        return bool(num_re.match(t))

    def is_money(txt):
        t = txt.strip().lower()
        if t in cash_tokens:
            return True
        if any(k in t for k in ["rs", "₹"]):
            return True
        return bool(moneyish_re.match(t)) or re.match(r'^[1-9]\d{2,}$', t) is not None

    def is_alphaish(txt):
        t = txt.strip()
        return (len(re.sub(r'[^A-Za-z]', '', t)) >= 2) and not re.match(r'^\d', t)

    # ========= 1) Find table region start & isolate data =========
    rownum_candidates = [it for it in ocr_items if is_row_number(it["text"])]
    rownum_candidates.sort(key=lambda d: (d["y"], d["x"]))

    if not rownum_candidates:
        data_items = sorted(ocr_items, key=lambda d: (d["y"], d["x"]))
    else:
        first_y = rownum_candidates[0]["y"]
        data_items = [it for it in ocr_items if it["y"] >= first_y - 10]
        data_items.sort(key=lambda d: (d["y"], d["x"]))

    # ========= 2) Build row bands =========
    rownums = [it for it in data_items if is_row_number(it["text"])]
    rownums.sort(key=lambda d: d["y"])

    # Remove duplicates based on Y position
    deduped = []
    for it in rownums:
        if not deduped or abs(it["y"] - deduped[-1]["y"]) > np.median([r["h"] for r in rownums]) * 0.6:
            deduped.append(it)
    rownums = deduped

    # Create row bands using K-means if not enough row numbers
    if len(rownums) < 5:
        ys = np.array([d["y"] for d in data_items]).reshape(-1, 1)
        median_h = np.median([d["h"] for d in data_items])
        est_rows = int(max(5, min(40, (max(ys)[0] - min(ys)[0]) / max(8.0, median_h * 1.2))))
        km = KMeans(n_clusters=est_rows, random_state=42, n_init=10).fit(ys)
        centers = sorted(km.cluster_centers_.flatten())
        row_centers = centers
    else:
        row_centers = [r["y"] for r in rownums]

    row_centers = sorted(row_centers)
    bands = []
    for i, yc in enumerate(row_centers):
        y_top = (row_centers[i - 1] + yc) / 2 if i > 0 else yc - 1000
        y_bot = (yc + row_centers[i + 1]) / 2 if i < len(row_centers) - 1 else yc + 1000
        bands.append((y_top, y_bot, yc))

    # Assign items to rows
    rows = [[] for _ in bands]
    for it in data_items:
        y = it["y"]
        idx = None
        for j, (yt, yb, yc) in enumerate(bands):
            if yt <= y < yb:
                idx = j
                break
        if idx is None:
            dists = [abs(y - b[2]) for b in bands]
            idx = int(np.argmin(dists))
        rows[idx].append(it)

    # Remove empty bands at extremes
    def trim_empty_row_edges(rlist):
        start = 0
        while start < len(rlist) and len(rlist[start]) == 0:
            start += 1
        end = len(rlist) - 1
        while end >= 0 and len(rlist[end]) == 0:
            end -= 1
        return rlist[start:end + 1] if end >= start else []

    rows = trim_empty_row_edges(rows)

    # ========= 3) Column detection =========
    flat = []
    for row in rows:
        for it in row:
            if not is_row_number(it["text"]):
                flat.append(it)

    if not flat:
        raise RuntimeError("No table-like cells after banding")

    X = np.array([it["x"] for it in flat]).reshape(-1, 1)
    k = len(COLUMNS)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    centers = sorted(kmeans.cluster_centers_.flatten())

    def cluster_idx(x):
        return int(np.argmin([abs(x - c) for c in centers]))

    # Sort clusters left to right
    sorted_clusters = sorted(range(k), key=lambda i: centers[i])

    # Map clusters to columns
    cluster_to_col = {cluster: i for i, cluster in enumerate(sorted_clusters)}

    # ========= 4) Row assembly =========
    processed = []
    for r_elems in rows:
        if not r_elems:
            continue
        col_dict = defaultdict(list)

        # Process non-row-number items
        other_items = [it for it in r_elems if not is_row_number(it["text"])]

        for it in sorted(other_items, key=lambda d: d["x"]):
            ci = cluster_idx(it["x"])
            col_idx = cluster_to_col.get(ci, len(COLUMNS) - 1)
            col_dict[col_idx].append(it["text"])

        row = [''] * len(COLUMNS)
        for ci, toks in col_dict.items():
            if ci < len(COLUMNS):
                txt = ' '.join(toks).strip()
                row[ci] = txt

        # Skip completely empty rows
        if not any(x.strip() for x in row):
            continue

        processed.append(row)

    # Filter valid rows
    final_rows = []
    for r in processed:
        has_name = bool(re.search(r'[A-Za-z]', r[0] or '')) if len(r) > 0 else False
        has_content = any(x.strip() for x in r)

        if has_name or has_content:
            final_rows.append(r)

    df = pd.DataFrame(final_rows, columns=COLUMNS)

    # Remove completely empty rows
    df = df[df.apply(lambda x: any(x.astype(str).str.strip() != ''), axis=1)]

    # Clean fees column
    if 'Fees' in df.columns:
        df['Fees'] = df['Fees'].str.replace(r'[^\d.]', '', regex=True)

    return df