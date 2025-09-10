from paddleocr import PaddleOCR
import pandas as pd
import numpy as np
import os, re
from collections import defaultdict, Counter
from sklearn.cluster import KMeans

# ========= CONFIG =========
COLUMNS = ['Name', 'Company Name', 'Category', 'Invited by', 'Fees',' Payment Mode', 'Date']  # Removed '#' column


def extract_data_from_image(image_path):
    # Initialize OCR
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    res = ocr.predict(image_path)

    if not (isinstance(res, list) and len(res) and isinstance(res[0], dict)):
        raise RuntimeError("No OCR result")

    rec_texts = res[0].get('rec_texts', [])
    dt_polys = res[0].get('dt_polys', [])
    rec_scores = res[0].get('rec_scores', [1.0] * len(rec_texts))

    ocr_items = []
    for t, poly, sc in zip(rec_texts, dt_polys, rec_scores):
        if not t:
            continue
        xs = [p[0] for p in poly];
        ys = [p[1] for p in poly]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        ocr_items.append({
            "text": t.strip(),
            "score": float(sc),
            "x": (x_min + x_max) / 2.0,
            "y": (y_min + y_max) / 2.0,
            "xmin": x_min, "xmax": x_max,
            "ymin": y_min, "ymax": y_max,
            "w": x_max - x_min,
            "h": y_max - y_min
        })

    if not ocr_items:
        raise RuntimeError("No OCR items parsed")

    # ========= Helpers =========
    num_re = re.compile(r'^\d{1,2}\.?$')
    #time_re = re.compile(r'^\s*(?:[0-2]?\d)[:.][0-5]\d\s*$')  # 7:35, 7.35, 07:25 etc
    moneyish_re = re.compile(r'^\s*[\d,.]{3,}\s*$')  # 7000, 9,000, 12135
    cash_tokens = {"cash", "done", "online", "upi", "od", "cheque", "dd"}

    #return bool(num_re.match(t))
    def is_row_number(txt):
        t = txt.strip().replace(' ', '')
        return bool(num_re.match(t))
    #def is_time(txt):
        #t = txt.strip().lower().replace('~', '').replace('^', '')
       # return (':' in t or '.' in t) and bool(time_re.match(t))

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

    # ========= 2) Build row bands from the "#" column =========
    rownums = [it for it in data_items if is_row_number(it["text"])]
    rownums.sort(key=lambda d: d["y"])

    deduped = []
    for it in rownums:
        if not deduped or abs(it["y"] - deduped[-1]["y"]) > np.median([r["h"] for r in rownums]) * 0.6:
            deduped.append(it)
    rownums = deduped

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
        y_top = (row_centers[i - 1] + yc) / 2 if i > 0 else yc - 1000  # large top
        y_bot = (yc + row_centers[i + 1]) / 2 if i < len(row_centers) - 1 else yc + 1000
        bands.append((y_top, y_bot, yc))

    rows = [[] for _ in bands]
    for it in data_items:
        y = it["y"]
        lo, hi = 0, len(bands) - 1
        idx = None
        for j, (yt, yb, yc) in enumerate(bands):
            if yt <= y < yb:
                idx = j;
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

    # ========= 3) Column detection (cluster x, then label) =========
    # Filter out row numbers from clustering to avoid confusion
    flat = []
    for row in rows:
        for it in row:
            if not is_row_number(it["text"]):  # Exclude row numbers from column clustering
                flat.append(it)

    if not flat:
        raise RuntimeError("No table-like cells after banding")

    X = np.array([it["x"] for it in flat]).reshape(-1, 1)
    k = len(COLUMNS)  # Use the actual number of columns we want (5)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    centers = sorted(kmeans.cluster_centers_.flatten())

    def cluster_idx(x):
        return int(np.argmin([abs(x - c) for c in centers]))

    stats = {i: Counter() for i in range(k)}
    examples = {i: [] for i in range(k)}
    for it in flat:
        ci = cluster_idx(it["x"])
        t = it["text"]
        if is_row_number(t): stats[ci]["rownum"] += 1
        if is_alphaish(t):   stats[ci]["alpha"] += 1
        if is_money(t):      stats[ci]["money"] += 1
        #if is_time(t):       stats[ci]["time"] += 1
        examples[ci].append(t)

    # Build cluster-to-column assignment
    ordered_clusters = list(range(k))  # assuming k == 5

    # Sort clusters left to right based on x-coordinate
    sorted_clusters = sorted(ordered_clusters, key=lambda i: centers[i])

    # Assign them left-to-right to expected columns
    expected_columns = ['Name', 'Company Name', 'Category', 'Invited by', 'Fees', 'Payment Mode','Date']
    label_map = {col: cluster for col, cluster in zip(expected_columns, sorted_clusters)}

    # Now map cluster -> column index
    cluster_to_col = {cluster: COLUMNS.index(col) for col, cluster in label_map.items()}

        # ========= 4) Row assembly =========
    processed = []
    for r_elems in rows:
        if not r_elems:
            continue
        col_dict = defaultdict(list)

        # Separate row numbers from other elements
        row_num_items = [it for it in r_elems if is_row_number(it["text"])]
        other_items = [it for it in r_elems if not is_row_number(it["text"])]

        # Process non-row-number items
        for it in sorted(other_items, key=lambda d: d["x"]):
            ci = cluster_idx(it["x"])
            col_idx = cluster_to_col.get(ci, len(COLUMNS) - 1)  # Default to last column if not found
            col_dict[col_idx].append(it["text"])

        row = [''] * len(COLUMNS)
        for ci, toks in col_dict.items():
            if ci < len(COLUMNS):  # Ensure valid column index
                txt = ' '.join(toks).strip()
                row[ci] = txt

        # Data cleaning and validation
        # If Payment column has money-like text and TOA is empty, swap them
        if len(row) > 2 and row[2] and is_money(row[2]) and not row[1]:
            row[1], row[2] = row[2], ''

        # If Name is empty and Payment has alpha text, swap them
        if len(row) > 2 and not row[0] and row[2] and is_alphaish(row[2]):
            row[0], row[2] = row[2], ''

        # Clean signature column (remove very short non-alphabetic entries)
        if len(row) > 4 and row[4] and len(row[4]) <= 2 and not is_alphaish(row[4]):
            row[4] = ''

        # Skip completely empty rows
        if not any(x.strip() for x in row):
            continue

        processed.append(row)

    # Filter valid rows - should have at least a name or some meaningful content
    final_rows = []
    for r in processed:
        has_name = bool(re.search(r'[A-Za-z]', r[0] or '')) if len(r) > 0 else False
        has_content = any(x.strip() for x in r)

        if has_name or has_content:
            final_rows.append(r)

    df = pd.DataFrame(final_rows, columns=COLUMNS)

    # Remove any rows that are completely empty after DataFrame creation
    df = df[df.apply(lambda x: any(x.astype(str).str.strip() != ''), axis=1)]
    df['Fees'] = df['Fees'].str.replace(r'[^\d.]', '', regex=True)

    return df