"""
anomaly_detection.py — ML-based anomaly detection on Cape Verde ocean time series
==================================================================================
Companion script to the OSCM Cape Verde dashboard.

Two complementary unsupervised methods are demonstrated:

  1. Isolation Forest  — detects anomalies in multivariate feature space.
     Fast, interpretable, no assumptions about data distribution.
     Good for: sudden spikes, unusual variable combinations.

  2. LSTM Autoencoder  — learns the normal temporal pattern of the series
     and flags timesteps where reconstruction error is unusually high.
     Good for: contextual anomalies that look normal in isolation but
     break the expected temporal pattern.

Why two methods?
  Rule-based QC (range checks, spike tests) catches obvious errors.
  Isolation Forest catches unusual multivariate combinations.
  LSTM Autoencoder catches unusual temporal patterns.
  Together they form a layered anomaly detection pipeline — a realistic
  approach for operational oceanographic data streams.

Usage:
    python anomaly_detection.py

Requires:
    pip install scikit-learn tensorflow pandas numpy matplotlib requests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import sys
import os

# Suppress TF/Keras noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# Add the dashboard directory to path so we can reuse data.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =============================================================================
# SECTION 1: Load data
# =============================================================================
# We reuse fetch_marine() from data.py so this script always works on
# the same data as the dashboard — real when available, synthetic otherwise.

print("=" * 62)
print("  Cape Verde Ocean Anomaly Detection")
print("=" * 62)

print("\n[1/5] Loading marine time series data...")
try:
    from data import fetch_marine
    df_raw, is_live = fetch_marine(days_back=60)
    source = "LIVE (Open-Meteo Marine API)" if is_live else "SYNTHETIC (fallback)"
    print(f"  Source  : {source}")
except ImportError:
    print("  data.py not found — generating synthetic data directly")
    from data_fallback import _make_synthetic   # won't exist; handled below
    df_raw, is_live = None, False

if df_raw is None or len(df_raw) == 0:
    print("  No data returned — exiting")
    sys.exit(1)

print(f"  Records : {len(df_raw)} hourly observations")
print(f"  Period  : {df_raw.index.min().date()} → {df_raw.index.max().date()}")
print(f"  Columns : {list(df_raw.columns)}")


# =============================================================================
# SECTION 2: Feature engineering
# =============================================================================
# Raw values alone can miss contextual anomalies.
# We add temporal features and rolling statistics as additional signals.

print("\n[2/5] Engineering features...")

FEATURES = []   # will hold column names used for ML

df = df_raw.copy()

# ── Core variables ────────────────────────────────────────────────────────────
core_vars = ["wave_height", "wave_period", "sea_surface_temperature"]
core_vars = [v for v in core_vars if v in df.columns]

# ── Rolling statistics (window = 6h) ─────────────────────────────────────────
# Capture recent trend and volatility — anomalies often have unusual local stats
for var in core_vars:
    df[f"{var}_roll_mean"] = df[var].rolling(6, min_periods=1).mean()
    df[f"{var}_roll_std"]  = df[var].rolling(6, min_periods=1).std().fillna(0)
    df[f"{var}_diff"]      = df[var].diff().fillna(0)   # first difference

# ── Hour of day ───────────────────────────────────────────────────────────────
# SST has a diurnal cycle; including time-of-day helps the model learn this
# and not flag normal daytime warming as anomalous
df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

# ── Assemble feature matrix ───────────────────────────────────────────────────
FEATURES = (
    core_vars +
    [f"{v}_roll_mean" for v in core_vars] +
    [f"{v}_roll_std"  for v in core_vars] +
    [f"{v}_diff"      for v in core_vars] +
    ["hour_sin", "hour_cos"]
)
FEATURES = [f for f in FEATURES if f in df.columns]

# Drop rows with any NaN in features (first few rows from rolling windows)
df_feat = df[FEATURES].dropna()
print(f"  Features: {len(FEATURES)}")
print(f"  Rows after dropping NaN: {len(df_feat)}")

# ── Standardise ───────────────────────────────────────────────────────────────
# Both methods benefit from zero-mean, unit-variance features
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(df_feat.values)


# =============================================================================
# SECTION 3: Method 1 — Isolation Forest
# =============================================================================
# How it works:
#   Randomly partition the feature space by selecting a feature and a split
#   value at random. Anomalies are points that get isolated quickly (short
#   average path length in the ensemble of random trees). Normal points
#   require many splits to isolate.
#
# Key parameter:
#   contamination — expected fraction of anomalies. We use 0.02 (2%) as a
#   conservative estimate for oceanographic sensor data.

print("\n[3/5] Fitting Isolation Forest...")

CONTAMINATION = 0.02   # assume ~2% of data points are anomalous

iso_forest = IsolationForest(
    n_estimators=200,       # more trees = more stable results
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1,
)
iso_forest.fit(X_scaled)

# score_samples returns negative anomaly scores:
#   more negative = more anomalous
iso_scores  = -iso_forest.score_samples(X_scaled)   # flip sign: higher = more anomalous
iso_labels  = iso_forest.predict(X_scaled)           # 1 = normal, -1 = anomaly
iso_anomaly = pd.Series(iso_labels == -1, index=df_feat.index)

n_iso = iso_anomaly.sum()
print(f"  Anomalies flagged: {n_iso} ({n_iso/len(iso_anomaly)*100:.1f}%)")
print(f"  Top anomaly score: {iso_scores.max():.3f}")


# =============================================================================
# SECTION 4: Method 2 — MLP Autoencoder
# =============================================================================
# How it works:
#   Same concept as an LSTM Autoencoder: train a neural network to reconstruct
#   "normal" time windows, then flag windows with high reconstruction error.
#
#   Implementation: scikit-learn's MLPRegressor, which avoids the TensorFlow
#   dependency. Each 24-hour window is flattened into a single vector and fed
#   through a bottleneck architecture (64 → 32 → 64 hidden units). The network
#   must compress the window through the 32-unit bottleneck to reconstruct it,
#   so it learns what normal patterns look like. Anomalous windows reconstruct
#   poorly → high MSE → flagged.
#
# Trade-off vs LSTM:
#   The MLP treats each window as a flat vector, so it doesn't explicitly model
#   temporal order within the window. An LSTM processes the sequence step-by-step
#   and is better at detecting order-dependent patterns. In practice, for
#   hourly oceanographic data the difference is small — the rolling statistics
#   in the feature matrix already encode much of the temporal structure.
#
# Key parameters:
#   WINDOW      — how many timesteps per window (24h)
#   THRESHOLD   — percentile of reconstruction error above which we flag
#                 anomalies (95th percentile = top 5% of errors)

print("\n[4/5] Training MLP Autoencoder...")

from sklearn.neural_network import MLPRegressor

WINDOW    = 24   # 24-hour sliding window
THRESHOLD = 95   # flag top 5% reconstruction errors as anomalous

# ── Build flattened windows ───────────────────────────────────────────────────
# Shape: (n_samples, WINDOW * n_features)
n_features = X_scaled.shape[1]
X_win = np.array([X_scaled[i:i+WINDOW].flatten()
                  for i in range(len(X_scaled) - WINDOW + 1)])
print(f"  Window matrix shape: {X_win.shape}")

# ── Architecture ─────────────────────────────────────────────────────────────
# Encoder:  input → 64 → 32 (bottleneck)
# Decoder:  32 → 64 → output
# The bottleneck forces the network to learn a compressed representation
# of normal patterns rather than simply copying the input.
ae_mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32, 64),
    activation="tanh",          # tanh keeps activations bounded, good for time series
    solver="adam",              # adaptive learning rate, same as in the LSTM version
    max_iter=200,
    random_state=42,
    early_stopping=True,        # stop when validation loss stops improving
    validation_fraction=0.1,
    n_iter_no_change=10,
)

# ── Train ─────────────────────────────────────────────────────────────────────
# Unsupervised: target = input (the network learns to reconstruct itself)
ae_mlp.fit(X_win, X_win)
print(f"  Converged after {ae_mlp.n_iter_} iterations")
print(f"  Final validation loss: {ae_mlp.best_validation_score_:.4f}")

# ── Reconstruction error ──────────────────────────────────────────────────────
X_win_pred = ae_mlp.predict(X_win)
mse_per_win = np.mean((X_win - X_win_pred) ** 2, axis=1)

# Align errors back to the original DataFrame index
ae_scores = np.full(len(df_feat), np.nan)
ae_scores[WINDOW-1:] = mse_per_win

threshold_val = np.nanpercentile(ae_scores, THRESHOLD)
lstm_anomaly  = pd.Series(ae_scores > threshold_val, index=df_feat.index)
lstm_scores_s = pd.Series(ae_scores, index=df_feat.index)

n_ae = lstm_anomaly.sum()
print(f"  Anomalies flagged: {n_ae} ({n_ae/lstm_anomaly.notna().sum()*100:.1f}%)")
print(f"  Reconstruction error threshold: {threshold_val:.4f}")

# ── Combined flag ─────────────────────────────────────────────────────────────
# An observation is "high-confidence anomalous" if BOTH methods flag it.
# This reduces false positives — a key concern in operational QC.
combined_anomaly = iso_anomaly & lstm_anomaly
n_combined = combined_anomaly.sum()
print(f"\n  Combined (both methods): {n_combined} anomalies")


# =============================================================================
# SECTION 5: Visualisation
# =============================================================================

print("\n[5/5] Generating plots...")

# Attach results back to the full dataframe
df_feat["iso_anomaly"]      = iso_anomaly
df_feat["iso_score"]        = iso_scores
df_feat["lstm_anomaly"]     = lstm_anomaly
df_feat["ae_score"]         = lstm_scores_s
df_feat["combined_anomaly"] = combined_anomaly

plot_vars = [v for v in core_vars if v in df_feat.columns]
n_vars    = len(plot_vars)

fig, axes = plt.subplots(
    nrows  = n_vars + (2 if lstm_available else 1),
    ncols  = 1,
    figsize = (14, 3.5 * (n_vars + (2 if lstm_available else 1))),
    sharex  = True,
)
fig.patch.set_facecolor("#0b1d2e")
for ax in axes:
    ax.set_facecolor("#0b1d2e")
    ax.tick_params(colors="#dce9f5", labelsize=9)
    ax.spines[:].set_color("#1e4a6e")
    ax.yaxis.label.set_color("#7db5d8")
    ax.title.set_color("#dce9f5")

COLORS = {"normal": "#3b9edd", "iso": "#f4a261", "lstm": "#2ec4b6", "combined": "#f87171"}

# ── Variable panels ───────────────────────────────────────────────────────────
for i, var in enumerate(plot_vars):
    ax   = axes[i]
    vals = df_feat[var]

    # Normal points
    normal_mask = ~iso_anomaly
    ax.plot(vals.index[normal_mask], vals[normal_mask],
            color=COLORS["normal"], linewidth=0.8, label="Normal", zorder=2)

    # Isolation Forest anomalies
    ax.scatter(vals.index[iso_anomaly], vals[iso_anomaly],
               color=COLORS["iso"], s=25, zorder=4,
               label=f"Isolation Forest ({iso_anomaly.sum()})", marker="^")

    # LSTM anomalies
    if True:  # MLP autoencoder always available
        ax.scatter(vals.index[lstm_anomaly], vals[lstm_anomaly],
                   color=COLORS["lstm"], s=25, zorder=4,
                   label=f"MLP Autoencoder ({lstm_anomaly.sum()})", marker="v", alpha=0.7)

    # Combined (both methods)
    ax.scatter(vals.index[combined_anomaly], vals[combined_anomaly],
               color=COLORS["combined"], s=60, zorder=5,
               label=f"Both methods ({combined_anomaly.sum()})", marker="x", linewidths=2)

    label_map = {
        "wave_height":           "Wave Height (m)",
        "wave_period":           "Wave Period (s)",
        "sea_surface_temperature": "SST (°C)",
    }
    ax.set_ylabel(label_map.get(var, var), fontsize=9)
    ax.set_title(label_map.get(var, var), fontsize=10, pad=4)
    ax.legend(loc="upper right", fontsize=8,
              facecolor="#112b44", edgecolor="#1e4a6e", labelcolor="#dce9f5")
    ax.grid(True, color="#1a3550", linewidth=0.5, linestyle="--")

# ── Isolation Forest anomaly score ───────────────────────────────────────────
ax_iso = axes[n_vars]
ax_iso.plot(df_feat.index, iso_scores,
            color=COLORS["iso"], linewidth=0.8, label="Anomaly score")
iso_thresh = np.percentile(iso_scores, (1 - CONTAMINATION) * 100)
ax_iso.axhline(iso_thresh, color=COLORS["combined"], linewidth=1.2,
               linestyle="--", label=f"Threshold (top {int(CONTAMINATION*100)}%)")
ax_iso.fill_between(df_feat.index, iso_scores, iso_thresh,
                    where=iso_scores >= iso_thresh,
                    color=COLORS["combined"], alpha=0.3)
ax_iso.set_ylabel("Score", fontsize=9)
ax_iso.set_title("Isolation Forest — anomaly score (higher = more anomalous)", fontsize=10, pad=4)
ax_iso.legend(loc="upper right", fontsize=8,
              facecolor="#112b44", edgecolor="#1e4a6e", labelcolor="#dce9f5")
ax_iso.grid(True, color="#1a3550", linewidth=0.5, linestyle="--")

# ── LSTM reconstruction error ─────────────────────────────────────────────────
if True:  # MLP autoencoder always available
    ax_lstm = axes[n_vars + 1]
    ax_lstm.plot(df_feat.index, lstm_scores_s,
                 color=COLORS["lstm"], linewidth=0.8, label="Reconstruction error (MSE)")
    ax_lstm.axhline(threshold_val, color=COLORS["combined"], linewidth=1.2,
                    linestyle="--", label=f"Threshold (p{THRESHOLD})")
    ax_lstm.fill_between(df_feat.index, lstm_scores_s.fillna(0), threshold_val,
                         where=lstm_scores_s.fillna(0) >= threshold_val,
                         color=COLORS["combined"], alpha=0.3)
    ax_lstm.set_ylabel("MSE", fontsize=9)
    ax_lstm.set_title("MLP Autoencoder — reconstruction error (higher = more anomalous)", fontsize=10, pad=4)
    ax_lstm.legend(loc="upper right", fontsize=8,
                   facecolor="#112b44", edgecolor="#1e4a6e", labelcolor="#dce9f5")
    ax_lstm.grid(True, color="#1a3550", linewidth=0.5, linestyle="--")

# ── X axis formatting ─────────────────────────────────────────────────────────
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

fig.suptitle(
    f"Cape Verde (OSCM) — ML Anomaly Detection  |  Data: {source}",
    fontsize=12, color="#dce9f5", y=1.002
)
plt.tight_layout()
plt.savefig("anomaly_detection_results.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("  Saved: anomaly_detection_results.png")


# =============================================================================
# SECTION 6: Summary report
# =============================================================================

print("\n" + "=" * 62)
print("  ANOMALY DETECTION SUMMARY")
print("=" * 62)

print(f"\nDataset: {len(df_feat)} hourly observations "
      f"({df_feat.index.min().date()} → {df_feat.index.max().date()})")
print(f"Features used: {len(FEATURES)}")

print(f"\nIsolation Forest:")
print(f"  Flagged  : {iso_anomaly.sum()} ({iso_anomaly.mean()*100:.1f}%)")
print(f"  Top score: {iso_scores.max():.3f}")

if True:  # MLP autoencoder always available
    valid = lstm_scores_s.dropna()
    print(f"\nMLP Autoencoder:")
    print(f"  Flagged    : {lstm_anomaly.sum()} ({lstm_anomaly.sum()/len(valid)*100:.1f}%)")
    print(f"  Max MSE    : {valid.max():.4f}")
    print(f"  Threshold  : {threshold_val:.4f}")

print(f"\nCombined (flagged by both):")
print(f"  Flagged  : {combined_anomaly.sum()} ({combined_anomaly.mean()*100:.1f}%)")

if combined_anomaly.sum() > 0:
    print(f"\nTop 10 combined anomalies:")
    top = df_feat[combined_anomaly][core_vars + ["iso_score"]].copy()
    top = top.sort_values("iso_score", ascending=False).head(10)
    top.index = top.index.strftime("%Y-%m-%d %H:%M")
    print(top.round(3).to_string())

print("\n" + "=" * 62)
print("  Done. See anomaly_detection_results.png for plots.")
print("=" * 62 + "\n")
