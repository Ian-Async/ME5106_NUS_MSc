import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Division")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FS = 128_000  # Hz
MIN_LEN = 256
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

param_map = {
    "A1": (350, 1200), "A2": (400, 600), "A3": (250, 600), "A4": (250, 1000),
    "A5": (400, 1000), "A6": (375, 1050), "A7": (350, 700), "A8": (200, 500),
    "A9": (350, 500), "A10": (200, 400), "A11": (200, 700), "A12": (250, 500),
    "A13": (400, 1200), "B1": (200, 200), "B2": (300, 1000), "B3": (150, 600),
    "B4": (400, 800), "B5": (150, 500), "B6": (300, 800), "B7": (150, 400),
    "B8": (300, 700), "B9": (400, 1600), "B10": (330, None), "B11": (150, 200),
    "B12": (300, 500), "B13": (400, 2000), "C1": (150, 800), "C2": (200, 1600),
    "C3": (200, 2000), "C4": (250, 400), "C5": (250, 1600), "C6": (250, 2000),
    "C7": (300, 1200), "C8": (300, 2000), "C9": (None, 600), "C10": (330, 1050),
    "C11": (350, 800), "C12": (350, 1600), "C13": (None, 700), "C14": (None, None)
}

def read_csv_series(fp):
    s = pd.read_csv(fp, header=None).iloc[:, 0].astype(float).values
    return s[np.isfinite(s)]

def detrend_and_bandpass(x, fs=FS, f_low=10.0, f_high=60_000.0, order=4):
    if len(x) < MIN_LEN:
        return None
    y = signal.detrend(x, type='constant')
    nyq = 0.5 * fs
    b, a = signal.butter(order, [f_low/nyq, f_high/nyq], btype='bandpass')
    return signal.filtfilt(b, a, y)

def safe_log10(x, eps=1e-12):
    return np.log10(np.maximum(np.abs(x), eps))

def time_features(y):
    y = y - np.mean(y)
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    crest = peak / (rms + 1e-12)
    kurt = stats.kurtosis(y)
    zcr = ((y[:-1]*y[1:]) < 0).mean()
    return {"rms": rms, "peak": peak, "crest": crest, "kurt": kurt, "zcr": zcr}

def welch_features(y, fs=FS):
    f, Pxx = signal.welch(y, fs=fs, nperseg=min(8192, len(y)))
    Pxx = np.maximum(Pxx, 1e-20)
    psd_sum = np.sum(Pxx)
    centroid = np.sum(f * Pxx) / psd_sum
    bw = np.sqrt(np.sum(((f - centroid)**2) * Pxx) / psd_sum)
    peak_idx = np.argmax(Pxx)
    return {"psd_centroid": centroid, "psd_bw": bw, "f_peak": f[peak_idx]}

def band_energy_features(y, fs=FS):
    bands = [(500, 2000), (2000, 8000), (8000, 20000), (20000, 40000), (40000, 64000)]
    f, Pxx = signal.welch(y, fs=fs, nperseg=min(8192, len(y)))
    feats = {}
    total = np.sum(Pxx)
    for i, (fl, fh) in enumerate(bands, 1):
        mask = (f >= fl) & (f < fh)
        feats[f"band{i}_ratio"] = np.sum(Pxx[mask]) / (total + 1e-12)
    return feats

def extract_features(y, global_rms_ref):
    tf = time_features(y)
    wf = welch_features(y)
    bf = band_energy_features(y)
    L_rel = 20 * safe_log10((tf["rms"] + 1e-12) / (global_rms_ref + 1e-12))
    return {**tf, **wf, **bf, "L_rel": L_rel}

def parse_sample_id(filename):
    m = re.search(r"([ABC]\d+)", filename)
    return m.group(1) if m else None

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
records = []
for fp in files:
    sid = parse_sample_id(os.path.basename(fp))
    if sid:
        y = read_csv_series(fp)
        records.append((sid, fp, y))

global_rms_ref = np.median([
    np.sqrt(np.mean(detrend_and_bandpass(y)**2))
    for _, _, y in records if detrend_and_bandpass(y) is not None
])

rows = []
for sid, fp, y in records:
    y_f = detrend_and_bandpass(y)
    if y_f is None:
        continue
    feats = extract_features(y_f, global_rms_ref)
    P, V = param_map.get(sid, (None, None))
    rows.append({"sample_id": sid, "file": fp, "P": P, "V": V, **feats})

df = pd.DataFrame(rows)
feature_cols = [c for c in df.columns if c not in ["sample_id", "file", "P", "V"]]

known_P = df[df["P"].notna()]
known_V = df[df["V"].notna()]
need_P = df[df["P"].isna()]
need_V = df[df["V"].isna()]

X_P, y_P = known_P[feature_cols].to_numpy(), known_P["P"].to_numpy(dtype=float)
X_V, y_V = known_V[feature_cols].to_numpy(), known_V["V"].to_numpy(dtype=float)

print(f"P: {X_P.shape}, V: {X_V.shape}")

model_P = RandomForestRegressor(n_estimators=600, random_state=42, max_features='sqrt')
model_V = RandomForestRegressor(n_estimators=400, random_state=42)
model_P.fit(X_P, y_P)
model_V.fit(X_V, y_V)

yP_pred = model_P.predict(X_P)
yV_pred = model_V.predict(X_V)
P_mae = mean_absolute_error(y_P, yP_pred)
P_mape = (np.abs((y_P - yP_pred) / np.maximum(y_P, 1e-9))).mean() * 100
P_r2 = r2_score(y_P, yP_pred)
print(f"P: MAE={P_mae:.2f}, MAPE={P_mape:.1f}%, R²={P_r2:.3f}")

V_mae = mean_absolute_error(y_V, yV_pred)
V_mape = (np.abs((y_V - yV_pred) / np.maximum(y_V, 1e-9))).mean() * 100
V_r2 = r2_score(y_V, yV_pred)
print(f"V: MAE={V_mae:.2f}, MAPE={V_mape:.1f}%, R²={V_r2:.3f}")

pred_rows = []

if not need_P.empty:
    Xp = need_P[feature_cols].to_numpy()
    p_hat = model_P.predict(Xp)
    for sid, v_true, p_pred in zip(
        need_P["sample_id"].tolist(),
        need_P["V"].tolist(),
        p_hat.tolist()
    ):
        pred_rows.append({
            "sample_id": sid,
            "P_true": None,
            "V_true": v_true,
            "P_pred": float(p_pred),
            "V_pred": None
        })

if not need_V.empty:
    Xv = need_V[feature_cols].to_numpy()
    v_hat = model_V.predict(Xv)
    for sid, p_true, v_pred in zip(
        need_V["sample_id"].tolist(),
        need_V["P"].tolist(),
        v_hat.tolist()
    ):
        pred_rows.append({
            "sample_id": sid,
            "P_true": p_true,
            "V_true": None,
            "P_pred": None,
            "V_pred": float(v_pred)
        })

pred_df = pd.DataFrame(pred_rows)
pred_path = os.path.join(RESULTS_DIR, "predictions.csv")
pred_df.to_csv(pred_path, index=False)

def scatter_true_pred(y_true, y_pred, label, savepath):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.8)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{label} True vs Predicted")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

scatter_true_pred(
    y_P, yP_pred,
    "Laser Power (P)",
    os.path.join(RESULTS_DIR, "true_pred_P.png")
)
scatter_true_pred(
    y_V, yV_pred,
    "Scan Speed (V)",
    os.path.join(RESULTS_DIR, "true_pred_V.png")
)

def feature_importance_bar(model, feature_names, fname):
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:15]
    plt.figure(figsize=(6,5))
    plt.barh(range(len(idx)), imp[idx][::-1])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx][::-1])
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importance")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

feature_importance_bar(
    model_P, feature_cols,
    os.path.join(RESULTS_DIR, "feat_imp_P.png")
)
feature_importance_bar(
    model_V, feature_cols,
    os.path.join(RESULTS_DIR, "feat_imp_V.png")
)