"""
================================================================================
Machine Vision Course                 MiniProject-6: FD + kNN Digit Classifier
================================================================================
Professor          : Prof. Hamidreza Pourreza
Institution        : Ferdowsi University of Mashhad
Term               : Spring 2025

Student Name       : Mustafa Sadeghi
Student ID         : 4027390423

Delivery Deadline  : 2025-09-06
Delivery Date      : 2025-09-06

Description:
    This script recognizes Persian digits (0–9) from a single page image
    ("Im321.png") using Fourier Descriptors (FD) and a single-stage kNN
    classifier. The pipeline:
      1) Preprocess & components:
         - Grayscale → Otsu (inverted) → external contours.
         - Group into 8 rows (adaptive gaps; fallback K-Means on y).
         - Per row: remove tiny round bullet, filter noise, split merged blobs,
           keep leftmost 10, and assign labels 9→0 (left→right).
      2) FD feature extraction:
         - External contour → arc-length resampling to N=512 points.
         - Center (C0=0), scale normalize by |C1|, phase-lock to C1
           (translation/scale/rotation/start invariance).
         - Keep lowest M harmonics for each retention {20%, 10%, 5%, 2%, 1%};
           feature = [Re(C1..M), Im(C1..M)] (dim = 2M).
      3) Classification & evaluation:
         - kNN (k=5, distance weighting).
         - Leave-One-Row-Out (LORO): hold out each row in turn.
      4) Outputs (saved to disk):
         - Plots: LORO overlays (loro_labeled_keepXX.png),
                  confusion-matrix heatmaps (cm_heatmap_keepXX.png).
         - Metrics: accuracy_keepXX.txt, confusion_keepXX.csv/txt,
                    classification_report_keepXX.txt, summary_keepXX.txt.
         - Config: outputs/config.txt
         - Console prints only [INFO] and [DONE] (no DEBUG).


Dependencies:
    - OpenCV        (pip install opencv-python)
    - NumPy         (pip install numpy)
    - Matplotlib    (pip install matplotlib)
    - scikit-learn  (pip install scikit-learn)

Author            : Mustafa Sadeghi
================================================================================
"""

import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import collections
from datetime import datetime

# ========================== CONFIG PARAMETERS ==========================
image_path = "Im321.png"

# FD & kNN
reduction_percentages = [0.20, 0.10, 0.05, 0.02, 0.01]
n_points = 512
knn_k = 5
knn_weights = "distance"

# optional mid-band emphasis
MID_BAND_K_LO = 4
MID_BAND_K_HI = 12
MID_BAND_BOOST = 1.4  # set 1.0 to disable

# dataset layout (8 rows × 10 digits per row; each row is 9..0 left→right)
EXPECTED_ROWS = 8
select_leftmost_n = 10

# rightmost bullet removal
drop_right_bullet = True
bullet_area_rel = 0.18
bullet_circularity_min = 0.85

# preprocessing
use_blur = False
blur_sigma_rel = 0.015
binarize = "otsu"         # 'otsu' or 'manual'
manual_thresh = 128
morph_open_ksize = 0
morph_close_ksize = 0

# row grouping
row_grouping = "adaptive" # 'adaptive' or 'kmeans'
row_gap_tol_rel = 0.45
row_gap_min = 6
kmeans_n_init = 10
kmeans_random_state = 0

# per-row filtering
min_component_area_rel = 0.03
min_component_height_rel = 0.35

# splitting merged components
split_wide_enable = True
split_width_rel_threshold = 1.5
split_area_rel_threshold  = 1.7
split_smooth_window = 5
split_margin_rel = 0.08
split_cut_width_rel = 0.01
split_guard_loops = 6

# outputs
OUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
METRICS_DIR = os.path.join(OUT_DIR, "metrics")
SAVE_ROWS_VIS = True
SAVE_LORO_VIS = True
SAVE_CM_HEATMAP = True
# ===========================================================

digit_colors = {
    0:(0,255,0),1:(0,255,255),2:(255,0,0),3:(255,0,255),4:(0,0,255),
    5:(255,255,0),6:(128,0,128),7:(0,128,255),8:(128,128,128),9:(0,0,0)
}

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_text(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# --------- geometry / resampling ----------
def resample_contour(cnt, n_points=256):
    pts = cnt.squeeze().astype(float)
    if pts.ndim == 1: pts = pts.reshape(-1, 2)
    if not np.allclose(pts[0], pts[-1]): pts = np.vstack([pts, pts[0]])
    d = np.hypot(np.diff(pts[:,0]), np.diff(pts[:,1]))
    s = np.concatenate(([0.0], np.cumsum(d)))
    if s[-1] <= 1e-12: return np.tile(pts[0], (n_points, 1))
    t = np.linspace(0.0, s[-1], n_points)
    out = np.zeros((n_points, 2))
    j = 1
    for i, ti in enumerate(t):
        while j < len(s) and s[j] < ti: j += 1
        if j >= len(s): out[i] = pts[-1]; continue
        a, b = s[j-1], s[j]
        if b <= a: out[i] = pts[j]
        else:
            u = (ti - a) / (b - a)
            out[i] = pts[j-1]*(1-u) + pts[j]*u
    return out

def complex_from_xy(xy): 
    return xy[:,0] + 1j*xy[:,1]

def circularity_cnt(cnt):
    A = cv2.contourArea(cnt); P = cv2.arcLength(cnt, True)
    if P <= 1e-9: return 0.0
    return float(4*np.pi*A/(P*P))

# --------- FD (outer, phase-locked) ----------
def fd_outer_phase_locked(contour, n_points=256):
    xy = resample_contour(contour, n_points)
    z = complex_from_xy(xy)
    z -= z.mean()                    # translation invariance
    fd = np.fft.fft(z)
    if len(fd) < 2: return np.zeros(max(1, n_points-1), complex)
    denom = max(np.abs(fd[1]), 1e-8) # scale
    fd = fd / denom
    phi1 = np.angle(fd[1])           # rotation/start
    fd = fd * np.exp(-1j * phi1)
    return fd[1:]                    # drop DC

# --------- preprocessing & components ----------
def preprocess_externals(path):
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(f"Cannot read {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if use_blur:
        _, thr0 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnt0,_ = cv2.findContours(thr0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects0 = [cv2.boundingRect(c) for c in cnt0]
        med_h = np.median([h for (_,_,_,h) in rects0]) if rects0 else max(8, gray.shape[0]/40)
        sigma = max(0.5, blur_sigma_rel * med_h)
        k = int(2 * round(3 * sigma) + 1)
        gray = cv2.GaussianBlur(gray, (k,k), sigmaX=sigma, sigmaY=sigma)

    if binarize == 'manual':
        _, bw = cv2.threshold(gray, manual_thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if morph_open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open_ksize, morph_open_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    if morph_close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close_ksize, morph_close_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return img, bw, contours

def extract_components(contours):
    comps = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        comps.append({'outer':c, 'bbox':(x,y,w,h), 'y_top':y, 'x_left':x})
    return comps

# --------- row building ----------
def rows_by_adaptive_gaps(comps):
    comps = sorted(comps, key=lambda d: d['y_top'])
    hs = [c['bbox'][3] for c in comps]
    med_h = np.median(hs) if hs else 1.0
    tol = max(row_gap_min, int(row_gap_tol_rel * med_h))
    rows, cur = [], []
    if not comps: return rows
    cur.append(comps[0]); cy = comps[0]['y_top']
    for c in comps[1:]:
        if abs(c['y_top'] - cy) <= tol:
            cur.append(c); cy = (cy*(len(cur)-1) + c['y_top']) / len(cur)
        else:
            rows.append(cur); cur=[c]; cy=c['y_top']
    if cur: rows.append(cur)
    return rows

def kmeans_rows_by_y(comps, k, random_state=0, n_init=10):
    if not comps: return []
    Ys = np.array([c['bbox'][1] + c['bbox'][3]/2 for c in comps]).reshape(-1,1)
    k = min(k, len(comps))
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(Ys)
    labs = km.labels_; ctr = km.cluster_centers_.flatten()
    order = np.argsort(ctr)
    rows = [[] for _ in range(k)]
    for comp, lab in zip(comps, labs):
        rows[np.where(order==lab)[0][0]].append(comp)
    for r in rows: r.sort(key=lambda d: d['x_left'])
    return rows

def remove_right_bullet_if_present(row):
    if not drop_right_bullet or len(row) <= select_leftmost_n:
        return row, False
    row_sorted = sorted(row, key=lambda d: d['x_left'])
    areas = np.array([cv2.contourArea(c['outer']) for c in row_sorted])
    medA  = np.median(areas) if len(areas) else 1.0
    cand  = row_sorted[-1]
    a_last = cv2.contourArea(cand['outer'])
    circ_last = circularity_cnt(cand['outer'])
    is_tiny = a_last < (bullet_area_rel * medA)
    is_round= circ_last > bullet_circularity_min
    if is_tiny and is_round:
        return row_sorted[:-1], True
    return row_sorted, False

def filter_noise_in_row(row):
    if not row: return row
    areas  = np.array([cv2.contourArea(c['outer']) for c in row])
    heights= np.array([c['bbox'][3] for c in row])
    medA = np.median(areas)  if len(areas)  else 1.0
    medH = np.median(heights)if len(heights)else 1.0
    row = [c for c,a,h in zip(row,areas,heights)
           if (a >= min_component_area_rel*medA and h >= min_component_height_rel*medH)]
    row.sort(key=lambda d: d['x_left'])
    return row

def split_wide_component(bw, comp):
    x,y,w,h = comp['bbox']
    roi = bw[y:y+h, x:x+w].copy()
    col_sum = roi.sum(axis=0).astype(np.float64)
    if split_smooth_window > 0:
        ker = np.ones(split_smooth_window)/max(1,split_smooth_window)
        col_sum = np.convolve(col_sum, ker, mode='same')
    margin = max(2, int(split_margin_rel * w))
    if w - 2*margin <= 2: return None
    mid = np.argmin(col_sum[margin: w-margin]) + margin
    cut_w = max(2, int(split_cut_width_rel * w))
    L = max(0, mid - cut_w//2); R = min(w, mid + cut_w//2 + 1)
    roi[:, L:R] = 0

    cs, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cs is None or len(cs) < 2: return None
    externals = []
    for cnt in cs:
        cnt_abs = cnt + np.array([[x,y]])
        bx, by, bw_, bh_ = cv2.boundingRect(cnt_abs)
        externals.append({'outer':cnt_abs, 'bbox':(bx,by,bw_,bh_), 'y_top':by, 'x_left':bx})
    externals.sort(key=lambda d: d['x_left'])
    return externals

def enforce_select_leftmost_n(row, bw):
    row = sorted(row, key=lambda d: d['x_left'])
    row = filter_noise_in_row(row)
    row, _ = remove_right_bullet_if_present(row)
    if len(row) > select_leftmost_n:
        return row[:select_leftmost_n]
    if split_wide_enable:
        guard = 0
        while len(row) < select_leftmost_n and len(row) > 0 and guard < split_guard_loops:
            guard += 1
            widths = np.array([d['bbox'][2] for d in row])
            areas  = np.array([cv2.contourArea(d['outer']) for d in row])
            medW = np.median(widths) if len(widths) else 0
            medA = np.median(areas)  if len(areas)  else 0
            idx = np.argmax(widths * np.maximum(areas,1))
            cand = row[idx]
            if (medW == 0 or medA == 0) or \
               (cand['bbox'][2] < split_width_rel_threshold*medW and
                cv2.contourArea(cand['outer']) < split_area_rel_threshold*medA):
                break
            parts = split_wide_component(bw, cand)
            if parts is None: break
            row = row[:idx] + parts + row[idx+1:]
            row.sort(key=lambda d: d['x_left'])
    return row[:select_leftmost_n]

def build_rows_and_select(img, bw, contours, save_vis=True):
    comps = extract_components(contours)

    if row_grouping == "kmeans":
        rows_raw = kmeans_rows_by_y(comps, k=EXPECTED_ROWS,
                                    random_state=kmeans_random_state, n_init=kmeans_n_init)
    else:
        rows_raw = rows_by_adaptive_gaps(comps)
        if len(rows_raw) != EXPECTED_ROWS:
            rows_raw = kmeans_rows_by_y(comps, k=EXPECTED_ROWS,
                                        random_state=kmeans_random_state, n_init=kmeans_n_init)

    # visualize rows BEFORE selection
    if save_vis and SAVE_ROWS_VIS:
        vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        palette = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
                   (0,255,255),(128,0,128),(0,128,255),(128,128,128),(0,0,0)]
        for rid, r in enumerate(rows_raw):
            color = palette[rid % len(palette)]
            for comp in r:
                x,y,w,h = comp['bbox']
                cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
                cv2.putText(vis, f"r{rid}", (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        out_name = os.path.join(PLOTS_DIR, "rows_before_selection.png")
        cv2.imwrite(out_name, vis)

    selected, final_rows = [], []
    for rid, row in enumerate(rows_raw):
        row_sel = enforce_select_leftmost_n(row, bw)
        lbl_row = []
        for j, comp in enumerate(row_sel):
            comp_ = {'outer': comp['outer'], 'bbox': comp['bbox'],
                     'row_id': rid, 'digit_label': 9 - j}
            selected.append(comp_); lbl_row.append(comp_)
        final_rows.append(lbl_row)

    if save_vis and SAVE_ROWS_VIS:
        overlay = img.copy()
        for rid, row in enumerate(final_rows):
            for comp in row:
                x,y,w,h = comp['bbox']
                cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(overlay, f"r{rid}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        out_name = os.path.join(PLOTS_DIR, "rows_after_selection.png")
        cv2.imwrite(out_name, overlay)

    return selected, final_rows

# --------- FD bank ----------
def compute_fd_bank(selected, n_points=256):
    bank = []
    for comp in selected:
        fd_pl = fd_outer_phase_locked(comp['outer'], n_points=n_points)
        bank.append({
            'row_id': comp['row_id'],
            'label': comp['digit_label'],
            'outer_pl_fd': fd_pl,
            'bbox': comp['bbox']
        })
    return bank

# --------- features / kNN ----------
def _mid_band_weights(M, k_lo, k_hi, boost):
    w = np.ones(M, dtype=float)
    lo = max(1, k_lo)
    hi = min(M, k_hi)
    if hi >= lo and boost != 1.0:
        w[lo-1:hi] = boost
    return w

def make_feature_matrix(bank, pct):
    L_out = len(bank[0]['outer_pl_fd'])
    M = max(1, int(L_out * pct))     # lowest M harmonics (k=1…M)
    w = _mid_band_weights(M, MID_BAND_K_LO, MID_BAND_K_HI, MID_BAND_BOOST)
    X, y = [], []
    for it in bank:
        fd = it['outer_pl_fd'][:M]
        feat = np.concatenate([fd.real * w, fd.imag * w])
        X.append(feat); y.append(it['label'])
    return np.array(X,float), np.array(y,int), M

def loro_evaluate_fd_only(bank, pct, k_neighbors=5, weights='distance'):
    X, y, M = make_feature_matrix(bank, pct)
    rid = np.array([it['row_id'] for it in bank], dtype=int)
    y_true_all, y_pred_all = [], []
    for r in np.unique(rid):
        te = (rid == r)
        tr = ~te
        if np.sum(te) == 0 or np.sum(tr) == 0: continue
        knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights=weights)
        knn.fit(X[tr], y[tr])
        y_pred = knn.predict(X[te])
        y_true_all.append(y[te]); y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all) if len(y_true_all) else np.array([], int)
    y_pred = np.concatenate(y_pred_all) if len(y_pred_all) else np.array([], int)
    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(10))) if len(y_true) else np.zeros((10,10), int)
    report = classification_report(y_true, y_pred, labels=list(range(10)), zero_division=0)
    return acc, cm, M, report, (y_true, y_pred)

# --------- visualization ----------
def _single_feat_from_item(it, M):
    fd = it['outer_pl_fd'][:M]
    w = _mid_band_weights(M, MID_BAND_K_LO, MID_BAND_K_HI, MID_BAND_BOOST)
    feat = np.concatenate([fd.real * w, fd.imag * w]).reshape(1, -1)
    return feat

def visualize_loro_predictions(img, bank, pct, k_neighbors=5, weights='distance', save_path=None):
    X, y, M = make_feature_matrix(bank, pct)
    rid = np.array([it['row_id'] for it in bank], dtype=int)
    out = img.copy()
    for r in np.unique(rid):
        te = (rid == r); tr = ~te
        if np.sum(te) == 0 or np.sum(tr) == 0: continue
        knn = KNeighborsClassifier(n_neighbors=k_neighbors, weights=weights)
        knn.fit(X[tr], y[tr])
        idxs = np.where(te)[0]
        for i_glob in idxs:
            feat = _single_feat_from_item(bank[i_glob], M)
            pred = int(knn.predict(feat)[0])
            x, yb, w, h = bank[i_glob]['bbox']
            clr = digit_colors[pred]
            cv2.rectangle(out, (x, yb), (x+w, yb+h), clr, 2)
            cv2.putText(out, str(pred), (x, yb-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
    if save_path is not None:
        cv2.imwrite(save_path, out)

def plot_and_save_cm(cm, title, save_png_path):
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(10)); plt.yticks(range(10))
    plt.colorbar()
    plt.tight_layout()
    fig.savefig(save_png_path, dpi=160)
    plt.close(fig)

# ============================= MAIN =============================
def main():
    ensure_dir(OUT_DIR); ensure_dir(PLOTS_DIR); ensure_dir(METRICS_DIR)

    cfg = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_path": image_path,
        "reduction_percentages": reduction_percentages,
        "n_points": n_points,
        "knn_k": knn_k,
        "knn_weights": knn_weights,
        "mid_band": [MID_BAND_K_LO, MID_BAND_K_HI, MID_BAND_BOOST],
        "EXPECTED_ROWS": EXPECTED_ROWS,
        "select_leftmost_n": select_leftmost_n
    }
    save_text(os.path.join(OUT_DIR, "config.txt"), json.dumps(cfg, ensure_ascii=False, indent=2))

    print(f"[INFO] Phase-locked FD, mid-band [{MID_BAND_K_LO}..{MID_BAND_K_HI}] boost={MID_BAND_BOOST}, "
          f"n_points={n_points}, kNN k={knn_k}/{knn_weights}")

    img, bw, contours = preprocess_externals(image_path)
    selected, rows = build_rows_and_select(img, bw, contours, save_vis=True)

    bank = compute_fd_bank(selected, n_points=n_points)

    for pct in reduction_percentages:
        acc_l, cm_l, M_l, report_l, (y_true, y_pred) = loro_evaluate_fd_only(
            bank, pct, k_neighbors=knn_k, weights=knn_weights
        )
        feat_dim = 2 * M_l
        pct_tag = f"{int(pct*100)}"

        if SAVE_LORO_VIS:
            out_loro = os.path.join(PLOTS_DIR, f"loro_labeled_keep{pct_tag}.png")
            visualize_loro_predictions(img, bank, pct, k_neighbors=knn_k, weights=knn_weights, save_path=out_loro)

        if SAVE_CM_HEATMAP:
            plot_and_save_cm(cm_l, title=f"Confusion Matrix (keep {pct_tag}%, dim={feat_dim})",
                             save_png_path=os.path.join(PLOTS_DIR, f"cm_heatmap_keep{pct_tag}.png"))

        # metrics to files
        save_text(os.path.join(METRICS_DIR, f"accuracy_keep{pct_tag}.txt"), f"{acc_l:.6f}")
        np.savetxt(os.path.join(METRICS_DIR, f"confusion_keep{pct_tag}.csv"), cm_l, fmt="%d", delimiter=",")
        cm_lines = [" ".join(f"{v:4d}" for v in row) for row in cm_l]
        save_text(os.path.join(METRICS_DIR, f"confusion_keep{pct_tag}.txt"), "\n".join(cm_lines))
        save_text(os.path.join(METRICS_DIR, f"classification_report_keep{pct_tag}.txt"), report_l)
        summary = []
        summary.append(f"KEEP = {pct*100:.1f}%  | feature_dim = {feat_dim}")
        summary.append(f"ACCURACY = {acc_l:.6f}")
        per_class_recall = []
        for i in range(10):
            total_i = cm_l[i].sum()
            correct_i = cm_l[i,i]
            r = (correct_i / total_i) if total_i > 0 else 0.0
            per_class_recall.append(r)
        summary.append("Per-class recall (0..9): " + ", ".join(f"{r:.3f}" for r in per_class_recall))
        save_text(os.path.join(METRICS_DIR, f"summary_keep{pct_tag}.txt"), "\n".join(summary))

    print(f"[DONE] All artifacts saved under: {OUT_DIR}/")

if __name__ == "__main__":
    main()
