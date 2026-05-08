import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# 0. Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ACL Load Predictor | CNN-LSTM-Attention",
    layout="wide",
    initial_sidebar_state="collapsed"
)

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.unicode_minus': False,
    'axes.grid': False,
})

# ─────────────────────────────────────────────
# 1. CSS — 沿用 XGB 页面风格，深蓝学术色调
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;600;800&family=Source+Code+Pro:wght@400;600&display=swap');

html, body, .main { background: #FFFFFF !important; }

h1.sci-title {
    font-family: 'EB Garamond', 'Times New Roman', serif;
    font-size: 2.4rem; font-weight: 800;
    color: #1A3A5C; text-align: center;
    margin-bottom: 4px; letter-spacing: -0.5px;
}
p.sci-subtitle {
    font-family: 'Times New Roman', serif;
    color: #7F8C8D; text-align: center;
    font-size: 0.95rem; margin-bottom: 28px;
}
.section-header {
    font-family: 'EB Garamond', serif;
    font-size: 1.15rem; font-weight: 600;
    color: #1A3A5C; border-bottom: 1.5px solid #1A3A5C;
    padding-bottom: 4px; margin-bottom: 14px; margin-top: 6px;
}
.result-card {
    background: #F4F7FB;
    border: 1px solid #D5E3F0;
    border-left: 6px solid #1A5276;
    border-radius: 6px;
    padding: 22px 24px 18px;
    margin-top: 12px;
}
.label-text {
    font-family: 'Times New Roman', serif;
    color: #566573; font-size: 0.82rem;
    text-transform: uppercase; letter-spacing: 1.2px; font-weight: bold;
}
.value-text {
    font-family: 'Source Code Pro', 'Courier New', monospace;
    color: #111; font-size: 3.2rem; font-weight: bold; line-height: 1.1;
}
.status-text {
    font-family: 'EB Garamond', 'Arial Black', sans-serif;
    font-size: 2.2rem; font-weight: 900;
    margin-left: 28px; letter-spacing: -1px;
}
.result-row { display: flex; align-items: center; margin-top: 4px; }
.meta-text { color: #7F8C8D; font-size: 0.82rem; margin-top: 8px; font-family: 'Times New Roman', serif; }
.timeshap-note {
    background: #EBF5FB; border-left: 4px solid #2E86C1;
    padding: 10px 14px; border-radius: 4px;
    font-size: 0.83rem; color: #1A3A5C;
    font-family: 'Times New Roman', serif; margin-bottom: 12px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Times New Roman', serif !important;
    font-size: 0.92rem !important;
}
div[data-testid="stNumberInput"] { margin-bottom: -6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. Model definition (must match notebook)
# ─────────────────────────────────────────────
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_dense = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        scores  = self.attention_dense(inputs)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(inputs * weights, dim=1)
        return context, weights


class CNNLSTMAttentionModel(nn.Module):
    def __init__(self, input_shape, output_dim=1,
                 cnn_channels=64, hidden_size=128,
                 lstm_layers=2, dropout_rate=0.3):
        super().__init__()
        self.sequence_length = input_shape[0]
        self.input_features  = input_shape[1]
        self.hidden_size     = hidden_size

        self.conv1d = nn.Conv1d(self.input_features, cnn_channels,
                                kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm1d(cnn_channels)
        self.lstm   = nn.LSTM(input_size=cnn_channels,
                              hidden_size=hidden_size,
                              num_layers=lstm_layers,
                              dropout=0.2 if lstm_layers > 1 else 0.0,
                              batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.dropout   = nn.Dropout(dropout_rate)
        self.fc        = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x_cnn  = x.permute(0, 2, 1)
        x_cnn  = F.relu(self.bn1(self.conv1d(x_cnn)))
        x_lstm = x_cnn.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_lstm)
        context, attn_w = self.attention(lstm_out)
        out = self.fc(self.dropout(context))
        return out, attn_w


# ─────────────────────────────────────────────
# 3. Load assets
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    cfg       = json.load(open("model_config.json"))
    threshold = json.load(open("acl_threshold.json"))["acl_threshold"]
    scaler_X  = joblib.load("scaler_X.pkl")
    scaler_y  = joblib.load("scaler_y_acl.pkl")

    model = CNNLSTMAttentionModel(
        input_shape  = tuple(cfg["input_shape"]),
        output_dim   = cfg["output_dim"],
        cnn_channels = cfg["cnn_channels"],
        hidden_size  = cfg["hidden_size"],
        lstm_layers  = cfg["lstm_layers"],
        dropout_rate = cfg["dropout_rate"],
    )
    model.load_state_dict(torch.load("best_model_acl.pt", map_location="cpu"))
    model.eval()
    return model, scaler_X, scaler_y, threshold, cfg

try:
    model, scaler_X, scaler_y, ACL_THRESHOLD, cfg = load_assets()
    FEATURE_COLS = cfg["feature_cols"]       # 9 features
    TIME_AXIS    = np.array(cfg["time_axis_ms"])   # 21 points, 0-100 ms
    assets_ok    = True
except Exception as e:
    st.error(f"资源加载失败: {e}")
    assets_ok = False


# ─────────────────────────────────────────────
# 4. Prediction + TimeSHAP helpers
# ─────────────────────────────────────────────
def predict_single(X_seq: np.ndarray):
    """X_seq: [21, 9] raw (unscaled)"""
    X_flat   = X_seq.reshape(-1, 9)
    X_scaled = scaler_X.transform(X_flat).reshape(1, 21, 9)
    tensor   = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred, attn_w = model(tensor)
    pred_inv  = scaler_y.inverse_transform(pred.numpy())[0, 0]
    attn_np   = attn_w.squeeze().numpy()          # [21]
    return pred_inv, attn_np


def compute_timeshap_event(X_seq: np.ndarray, n_samples: int = 200):
    """
    Event-level TimeSHAP via perturbation (replacing each time-step with
    its feature-wise mean across all time-steps).
    Returns shap_event [21]
    """
    X_flat   = X_seq.reshape(-1, 9)
    X_scaled = scaler_X.transform(X_flat).reshape(21, 9)
    baseline = X_scaled.mean(axis=0)          # [9]

    def _fwd(x_3d):
        t = torch.tensor(x_3d, dtype=torch.float32)
        with torch.no_grad():
            out, _ = model(t)
        return scaler_y.inverse_transform(out.numpy())[0, 0]

    full_pred = _fwd(X_scaled[np.newaxis])
    shap_event = np.zeros(21)
    for t in range(21):
        x_masked = X_scaled.copy()
        x_masked[t] = baseline
        masked_pred = _fwd(x_masked[np.newaxis])
        shap_event[t] = full_pred - masked_pred
    return shap_event


def compute_timeshap_feature(X_seq: np.ndarray):
    """
    Feature-level TimeSHAP: marginalise over time.
    Returns shap_feature [9]
    """
    X_flat   = X_seq.reshape(-1, 9)
    X_scaled = scaler_X.transform(X_flat).reshape(21, 9)
    baseline = X_scaled.mean(axis=0)

    def _fwd(x_3d):
        t = torch.tensor(x_3d, dtype=torch.float32)
        with torch.no_grad():
            out, _ = model(t)
        return scaler_y.inverse_transform(out.numpy())[0, 0]

    full_pred = _fwd(X_scaled[np.newaxis])
    shap_feat = np.zeros(9)
    for f in range(9):
        x_masked = X_scaled.copy()
        x_masked[:, f] = baseline[f]
        shap_feat[f] = full_pred - _fwd(x_masked[np.newaxis])
    return shap_feat


def compute_timeshap_cell(X_seq: np.ndarray):
    """
    Cell-level TimeSHAP: [21 x 9] joint perturbation.
    Returns shap_cell [21, 9]
    """
    X_flat   = X_seq.reshape(-1, 9)
    X_scaled = scaler_X.transform(X_flat).reshape(21, 9)
    baseline = X_scaled.mean(axis=0)

    def _fwd(x_3d):
        t = torch.tensor(x_3d, dtype=torch.float32)
        with torch.no_grad():
            out, _ = model(t)
        return scaler_y.inverse_transform(out.numpy())[0, 0]

    full_pred = _fwd(X_scaled[np.newaxis])
    shap_cell = np.zeros((21, 9))
    for t in range(21):
        for f in range(9):
            x_masked = X_scaled.copy()
            x_masked[t, f] = baseline[f]
            shap_cell[t, f] = full_pred - _fwd(x_masked[np.newaxis])
    return shap_cell


# ─────────────────────────────────────────────
# 5. Plot helpers
# ─────────────────────────────────────────────
BLUE_DARK  = "#1A3A5C"
BLUE_MED   = "#2E86C1"
RED_ACCENT = "#C0392B"
GREEN_ACC  = "#1E8449"
GRAY_LIGHT = "#ECF0F1"

def _apply_spine(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')


def plot_event_level(shap_event, attn_weights):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5),
                             gridspec_kw={'hspace': 0.55})

    # — TimeSHAP event —
    ax = axes[0]
    colors = [RED_ACCENT if v >= 0 else BLUE_MED for v in shap_event]
    bars = ax.bar(TIME_AXIS, shap_event, width=4, color=colors, alpha=0.88, zorder=3)
    ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Time post-contact (ms)", fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)
    ax.set_title("Local Event-level TimeSHAP", fontsize=10, fontweight='bold', color=BLUE_DARK)
    ax.set_xticks(TIME_AXIS[::2])
    ax.tick_params(labelsize=8)
    _apply_spine(ax)

    # — Attention weights —
    ax2 = axes[1]
    ax2.fill_between(TIME_AXIS, attn_weights, alpha=0.35, color=BLUE_MED)
    ax2.plot(TIME_AXIS, attn_weights, color=BLUE_DARK, linewidth=1.6)
    peak_t = TIME_AXIS[np.argmax(attn_weights)]
    ax2.axvline(peak_t, color=RED_ACCENT, linewidth=1.2, linestyle=':', alpha=0.85)
    ax2.set_xlabel("Time post-contact (ms)", fontsize=9)
    ax2.set_ylabel("Attention weight", fontsize=9)
    ax2.set_title("Temporal Attention Weights", fontsize=10, fontweight='bold', color=BLUE_DARK)
    ax2.set_xticks(TIME_AXIS[::2])
    ax2.tick_params(labelsize=8)
    _apply_spine(ax2)

    fig.patch.set_facecolor('white')
    return fig


def plot_feature_level(shap_feat):
    idx_sorted = np.argsort(np.abs(shap_feat))[::-1]
    feat_sorted = [FEATURE_COLS[i] for i in idx_sorted]
    vals_sorted  = shap_feat[idx_sorted]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [RED_ACCENT if v >= 0 else BLUE_MED for v in vals_sorted]
    ax.barh(feat_sorted[::-1], vals_sorted[::-1],
            color=colors[::-1], alpha=0.88, height=0.6, zorder=3)
    ax.axvline(0, color='#999999', linewidth=0.8, linestyle='--')
    ax.set_xlabel("SHAP value (mean over time)", fontsize=9)
    ax.set_title("Local Feature-level TimeSHAP", fontsize=10, fontweight='bold', color=BLUE_DARK)
    ax.tick_params(labelsize=8.5)
    _apply_spine(ax)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


def plot_cell_level(shap_cell):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vmax = np.abs(shap_cell).max()
    im = ax.imshow(shap_cell.T, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, origin='upper')
    ax.set_xticks(np.arange(21)[::2])
    ax.set_xticklabels([f"{int(t)}" for t in TIME_AXIS[::2]], fontsize=8)
    ax.set_yticks(np.arange(9))
    ax.set_yticklabels(FEATURE_COLS, fontsize=8.5)
    ax.set_xlabel("Time post-contact (ms)", fontsize=9)
    ax.set_title("Local Cell-level TimeSHAP  (Feature × Time)", fontsize=10,
                 fontweight='bold', color=BLUE_DARK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("SHAP value", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 6. Header
# ─────────────────────────────────────────────
st.markdown("<h1 class='sci-title'>ACL Peak Load Prediction — CNN-LSTM-Attention</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='sci-subtitle'>Time-series biomechanical input · Deep learning regression · TimeSHAP interpretability</p>",
            unsafe_allow_html=True)

if not assets_ok:
    st.stop()

# ─────────────────────────────────────────────
# 7. Layout: left = inputs + result, right = tabs
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.35], gap="large")

with col_left:
    st.markdown("<div class='section-header'>📋 Input Parameters (per frame)</div>",
                unsafe_allow_html=True)
    st.caption("Each slider sets a **constant value across all 21 frames** (0–100 ms). "
               "For full time-series upload, use the CSV uploader below.")

    # — Sliders —
    c1, c2 = st.columns(2)
    with c1:
        hfa = st.number_input("Hip Flexion — HFA (°)",    value=21.20, step=0.1, format="%.2f")
        hra = st.number_input("Hip Rotation — HRA (°)",   value=5.00,  step=0.1, format="%.2f")
        haa = st.number_input("Hip Adduction — HAA (°)",  value=21.32, step=0.1, format="%.2f")
        kfa = st.number_input("Knee Flexion — KFA (°)",   value=30.10, step=0.1, format="%.2f")
        itr = st.number_input("Tibial Rotation — ITR (°)",value=6.00,  step=0.1, format="%.2f")
    with c2:
        kva = st.number_input("Knee Valgus — KVA (°)",    value=0.22,  step=0.01, format="%.2f")
        adf = st.number_input("Ankle Dorsifl. — ADF (°)", value=20.00, step=0.1, format="%.2f")
        fpa = st.number_input("Foot Prog. — FPA (°)",     value=12.00, step=0.1, format="%.2f")
        tfa = st.number_input("Trunk Flexion — TFA (°)",  value=24.00, step=0.1, format="%.2f")

    # Build [21, 9] sequence from constant values
    row_vals = [hfa, hra, haa, kfa, itr, kva, adf, fpa, tfa]
    X_seq = np.tile(row_vals, (21, 1)).astype(np.float32)

    # — Optional CSV upload —
    with st.expander("📂 Upload time-series CSV (21 rows × 9 features)"):
        uploaded = st.file_uploader("CSV must have columns: " + ", ".join(FEATURE_COLS),
                                    type=["csv"])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)[FEATURE_COLS].values
                if df_up.shape == (21, 9):
                    X_seq = df_up.astype(np.float32)
                    st.success("✅ CSV loaded — using uploaded time-series.")
                else:
                    st.error(f"Shape mismatch: expected (21, 9), got {df_up.shape}")
            except Exception as e:
                st.error(f"CSV error: {e}")

    # — Predict —
    run_btn = st.button("▶  Run Prediction", use_container_width=True, type="primary")

    # Store in session state
    if run_btn:
        with st.spinner("Computing prediction + TimeSHAP (cell-level may take ~10s)…"):
            pred_val, attn_np   = predict_single(X_seq)
            shap_event          = compute_timeshap_event(X_seq)
            shap_feat           = compute_timeshap_feature(X_seq)
            shap_cell           = compute_timeshap_cell(X_seq)
        st.session_state["pred_val"]   = pred_val
        st.session_state["attn_np"]    = attn_np
        st.session_state["shap_event"] = shap_event
        st.session_state["shap_feat"]  = shap_feat
        st.session_state["shap_cell"]  = shap_cell
        st.session_state["X_seq"]      = X_seq

    # — Result card —
    if "pred_val" in st.session_state:
        pred_val = st.session_state["pred_val"]
        is_high  = pred_val >= ACL_THRESHOLD
        s_color  = RED_ACCENT if is_high else GREEN_ACC
        s_label  = "HIGH LOAD" if is_high else "LOW LOAD"
        st.markdown(f"""
        <div class="result-card">
            <div class="label-text">Predicted Peak ACL Load</div>
            <div class="result-row">
                <span class="value-text">{pred_val:.4f}</span>
                <span class="status-text" style="color:{s_color};">{s_label}</span>
            </div>
            <div class="meta-text">
                Threshold: {ACL_THRESHOLD:.4f} &nbsp;|&nbsp;
                Rule: ≥ threshold → High Load<br>
                Model: CNN-LSTM-Attention &nbsp;|&nbsp; Window: 0–100 ms post-contact
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Model Info"):
            st.caption("Architecture: 1D CNN → LSTM (2 layers) → Temporal Attention → FC  |  "
                       "Input: 21 frames × 9 joint features  |  "
                       "Output: peak ACL load (inverse-transformed)")

        # Export
        df_export = pd.DataFrame([row_vals], columns=FEATURE_COLS)
        df_export["pred_ACL_peak"] = pred_val
        df_export["risk"]          = s_label
        st.download_button("📥 Export Report CSV",
                           data=df_export.to_csv(index=False).encode(),
                           file_name="acl_cnn_report.csv",
                           use_container_width=True)


# ─────────────────────────────────────────────
# 8. Right panel — TimeSHAP tabs
# ─────────────────────────────────────────────
with col_right:
    st.markdown("<div class='section-header'>🔍 TimeSHAP Interpretability</div>",
                unsafe_allow_html=True)

    if "shap_event" not in st.session_state:
        st.markdown("""
        <div class="timeshap-note">
        ▶ Click <b>Run Prediction</b> to generate TimeSHAP explanations.<br>
        Three levels of interpretability will appear here:<br>
        &nbsp;&nbsp;• <b>Event-level</b>: which time frames matter most<br>
        &nbsp;&nbsp;• <b>Feature-level</b>: which joint angles contribute most<br>
        &nbsp;&nbsp;• <b>Cell-level</b>: joint feature × time interaction heatmap
        </div>
        """, unsafe_allow_html=True)
    else:
        tab1, tab2, tab3 = st.tabs([
            "📈 Event-level",
            "📊 Feature-level",
            "🗺️ Cell-level"
        ])

        with tab1:
            st.markdown("""
            <div class="timeshap-note">
            <b>Event-level TimeSHAP</b>: SHAP value for each of the 21 time frames
            (0–100 ms post-contact). Positive bars (red) = frames that <i>increase</i>
            the predicted ACL load; negative (blue) = frames that <i>decrease</i> it.
            The attention curve below shows where the model focused.
            </div>
            """, unsafe_allow_html=True)
            fig1 = plot_event_level(st.session_state["shap_event"],
                                    st.session_state["attn_np"])
            st.pyplot(fig1, clear_figure=True)
            st.caption("Fig 1. Local event-level TimeSHAP + temporal attention weights.")

        with tab2:
            st.markdown("""
            <div class="timeshap-note">
            <b>Feature-level TimeSHAP</b>: Each feature's contribution is computed
            by masking that feature across <i>all</i> time frames and measuring the
            drop in predicted ACL load. Features are sorted by absolute impact.
            </div>
            """, unsafe_allow_html=True)
            fig2 = plot_feature_level(st.session_state["shap_feat"])
            st.pyplot(fig2, clear_figure=True)
            st.caption("Fig 2. Local feature-level TimeSHAP — feature importance aggregated over time.")

        with tab3:
            st.markdown("""
            <div class="timeshap-note">
            <b>Cell-level TimeSHAP</b>: The full Feature × Time interaction heatmap.
            Each cell shows the SHAP value when that specific (feature, frame) pair
            is masked. Red = contribution toward High Load; Blue = toward Low Load.
            </div>
            """, unsafe_allow_html=True)
            fig3 = plot_cell_level(st.session_state["shap_cell"])
            st.pyplot(fig3, clear_figure=True)
            st.caption("Fig 3. Local cell-level TimeSHAP heatmap (Feature × Time).")

# ─────────────────────────────────────────────
# 9. Footer
# ─────────────────────────────────────────────
st.markdown("""
<br><hr>
<div style='color:#95A5A6; font-size:0.78rem; font-family: Times New Roman;'>
CNN-LSTM-Attention ACL Load Predictor &nbsp;|&nbsp;
TimeSHAP: Bento et al. (2021) NeurIPS &nbsp;|&nbsp;
Reference: Zhang et al. (2026). DOI: 10.1016/j.jsams.2026.04.01
</div>
""", unsafe_allow_html=True)
