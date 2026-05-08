import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
import matplotlib.pyplot as plt
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
# 1. CSS
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
    margin-bottom: 20px;
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
.shap-label {
    font-family: 'EB Garamond', serif;
    font-size: 1.05rem; font-weight: 600; color: #1A3A5C;
    margin-top: 18px; margin-bottom: 4px;
}
.shap-desc {
    background: #EBF5FB; border-left: 4px solid #2E86C1;
    padding: 8px 12px; border-radius: 4px;
    font-size: 0.81rem; color: #1A3A5C;
    font-family: 'Times New Roman', serif; margin-bottom: 10px;
}
.upload-box {
    background: #F8FAFB; border: 1.5px dashed #AEC6CF;
    border-radius: 8px; padding: 18px 22px;
    margin-bottom: 14px; font-family: 'Times New Roman', serif;
    font-size: 0.88rem; color: #2E4053;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. Model definition
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
    m = CNNLSTMAttentionModel(
        input_shape  = tuple(cfg["input_shape"]),
        output_dim   = cfg["output_dim"],
        cnn_channels = cfg["cnn_channels"],
        hidden_size  = cfg["hidden_size"],
        lstm_layers  = cfg["lstm_layers"],
        dropout_rate = cfg["dropout_rate"],
    )
    m.load_state_dict(torch.load("best_model_acl.pt", map_location="cpu"))
    m.eval()
    return m, scaler_X, scaler_y, threshold, cfg

try:
    model, scaler_X, scaler_y, ACL_THRESHOLD, cfg = load_assets()
    FEATURE_COLS = cfg["feature_cols"]
    TIME_AXIS    = np.array(cfg["time_axis_ms"])
    assets_ok    = True
except Exception as e:
    st.error(f"Asset loading failed: {e}")
    assets_ok = False


# ─────────────────────────────────────────────
# 4. Prediction + TimeSHAP helpers
# ─────────────────────────────────────────────
def predict_single(X_seq):
    X_sc = scaler_X.transform(X_seq.reshape(-1, 9)).reshape(1, 21, 9)
    t    = torch.tensor(X_sc, dtype=torch.float32)
    with torch.no_grad():
        pred, attn_w = model(t)
    return scaler_y.inverse_transform(pred.numpy())[0, 0], attn_w.squeeze().numpy()


def _fwd(X_sc):
    t = torch.tensor(X_sc[np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        out, _ = model(t)
    return scaler_y.inverse_transform(out.numpy())[0, 0]


def compute_timeshap_event(X_seq):
    X_sc = scaler_X.transform(X_seq.reshape(-1, 9)).reshape(21, 9)
    base = X_sc.mean(axis=0)
    full = _fwd(X_sc)
    shap = np.zeros(21)
    for t in range(21):
        xm = X_sc.copy(); xm[t] = base
        shap[t] = full - _fwd(xm)
    return shap


def compute_timeshap_feature(X_seq):
    X_sc = scaler_X.transform(X_seq.reshape(-1, 9)).reshape(21, 9)
    base = X_sc.mean(axis=0)
    full = _fwd(X_sc)
    shap = np.zeros(9)
    for f in range(9):
        xm = X_sc.copy(); xm[:, f] = base[f]
        shap[f] = full - _fwd(xm)
    return shap


def compute_timeshap_cell(X_seq):
    X_sc = scaler_X.transform(X_seq.reshape(-1, 9)).reshape(21, 9)
    base = X_sc.mean(axis=0)
    full = _fwd(X_sc)
    shap = np.zeros((21, 9))
    for t in range(21):
        for f in range(9):
            xm = X_sc.copy(); xm[t, f] = base[f]
            shap[t, f] = full - _fwd(xm)
    return shap


# ─────────────────────────────────────────────
# 5. Plot functions
# ─────────────────────────────────────────────
BLUE_DARK = "#1A3A5C"
BLUE_MED  = "#2E86C1"
RED_ACC   = "#C0392B"
GREEN_ACC = "#1E8449"

def _spine(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')


def plot_event_level(shap_event, attn_weights):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5),
                             gridspec_kw={'hspace': 0.55})
    ax = axes[0]
    colors = [RED_ACC if v >= 0 else BLUE_MED for v in shap_event]
    ax.bar(TIME_AXIS, shap_event, width=4, color=colors, alpha=0.88, zorder=3)
    ax.axhline(0, color='#999', linewidth=0.8, linestyle='--')
    ax.set_xlabel("Time post-contact (ms)", fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)
    ax.set_title("Local Event-level TimeSHAP", fontsize=10,
                 fontweight='bold', color=BLUE_DARK)
    ax.set_xticks(TIME_AXIS[::2]); ax.tick_params(labelsize=8)
    _spine(ax)

    ax2 = axes[1]
    ax2.fill_between(TIME_AXIS, attn_weights, alpha=0.3, color=BLUE_MED)
    ax2.plot(TIME_AXIS, attn_weights, color=BLUE_DARK, linewidth=1.6)
    ax2.axvline(TIME_AXIS[np.argmax(attn_weights)], color=RED_ACC,
                linewidth=1.2, linestyle=':', alpha=0.85)
    ax2.set_xlabel("Time post-contact (ms)", fontsize=9)
    ax2.set_ylabel("Attention weight", fontsize=9)
    ax2.set_title("Temporal Attention Weights", fontsize=10,
                  fontweight='bold', color=BLUE_DARK)
    ax2.set_xticks(TIME_AXIS[::2]); ax2.tick_params(labelsize=8)
    _spine(ax2)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


def plot_feature_level(shap_feat):
    idx  = np.argsort(np.abs(shap_feat))
    feat = [FEATURE_COLS[i] for i in idx]
    vals = shap_feat[idx]
    fig, ax = plt.subplots(figsize=(8, 4.0))
    colors = [RED_ACC if v >= 0 else BLUE_MED for v in vals]
    ax.barh(feat, vals, color=colors, alpha=0.88, height=0.6, zorder=3)
    ax.axvline(0, color='#999', linewidth=0.8, linestyle='--')
    ax.set_xlabel("SHAP value (aggregated over time)", fontsize=9)
    ax.set_title("Local Feature-level TimeSHAP", fontsize=10,
                 fontweight='bold', color=BLUE_DARK)
    ax.tick_params(labelsize=8.5)
    _spine(ax)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    return fig


def plot_cell_level(shap_cell):
    fig, ax = plt.subplots(figsize=(10, 4.0))
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
# 6. Page layout
# ─────────────────────────────────────────────
st.markdown("<h1 class='sci-title'>ACL Peak Load Prediction — CNN-LSTM-Attention</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='sci-subtitle'>Time-series biomechanical input · Deep learning regression · TimeSHAP interpretability</p>",
            unsafe_allow_html=True)

if not assets_ok:
    st.stop()

col_left, col_right = st.columns([1, 1.5], gap="large")

# ── LEFT ──────────────────────────────────────
with col_left:
    st.markdown("<div class='section-header'>📂 Upload Time-Series CSV</div>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class='upload-box'>
    <b>Format:</b> 21 rows × 9 columns<br>
    <b>Required columns:</b><br>
    <code style='font-size:0.82rem'>{', '.join(FEATURE_COLS)}</code>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    X_seq = None
    if uploaded:
        try:
            df_up   = pd.read_csv(uploaded)
            missing = [c for c in FEATURE_COLS if c not in df_up.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            elif df_up.shape[0] != 21:
                st.error(f"Expected 21 rows, got {df_up.shape[0]}")
            else:
                X_seq = df_up[FEATURE_COLS].values.astype(np.float32)
                st.success(f"✅ Loaded — shape {X_seq.shape}")
                st.dataframe(df_up[FEATURE_COLS].round(3), height=200)
        except Exception as e:
            st.error(f"CSV error: {e}")

    run_btn = st.button("▶  Run Prediction + TimeSHAP",
                        use_container_width=True,
                        type="primary",
                        disabled=(X_seq is None))

    if run_btn and X_seq is not None:
        with st.spinner("Computing prediction + all TimeSHAP levels (~10 s)…"):
            pred_val, attn_np = predict_single(X_seq)
            shap_event        = compute_timeshap_event(X_seq)
            shap_feat         = compute_timeshap_feature(X_seq)
            shap_cell         = compute_timeshap_cell(X_seq)
        st.session_state.update({
            "pred_val":   pred_val,
            "attn_np":    attn_np,
            "shap_event": shap_event,
            "shap_feat":  shap_feat,
            "shap_cell":  shap_cell,
            "X_seq":      X_seq,
        })

    # Result card
    if "pred_val" in st.session_state:
        pv      = st.session_state["pred_val"]
        is_high = pv >= ACL_THRESHOLD
        s_color = RED_ACC if is_high else GREEN_ACC
        s_label = "HIGH LOAD" if is_high else "LOW LOAD"

        st.markdown(f"""
        <div class="result-card">
            <div class="label-text">Predicted Peak ACL Load</div>
            <div class="result-row">
                <span class="value-text">{pv:.4f}</span>
                <span class="status-text" style="color:{s_color};">{s_label}</span>
            </div>
            <div class="meta-text">
                Threshold: {ACL_THRESHOLD:.4f} &nbsp;|&nbsp; ≥ threshold → High Load<br>
                Model: CNN-LSTM-Attention &nbsp;|&nbsp; Window: 0–100 ms post-contact
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("ℹ️ Model Info"):
            st.caption("Architecture: 1D CNN → LSTM (2 layers) → Temporal Attention → FC  |  "
                       "Input: 21 frames × 9 joint features  |  "
                       "Output: peak ACL load (inverse-transformed)")

        df_exp = pd.DataFrame(st.session_state["X_seq"], columns=FEATURE_COLS)
        df_exp.insert(0, "frame_ms", TIME_AXIS)
        df_exp["pred_ACL_peak"] = pv
        df_exp["risk"]          = s_label
        st.download_button("📥 Export Report CSV",
                           data=df_exp.to_csv(index=False).encode(),
                           file_name="acl_cnn_report.csv",
                           use_container_width=True)


# ── RIGHT: all 3 SHAP plots stacked ───────────
with col_right:
    st.markdown("<div class='section-header'>🔍 TimeSHAP Interpretability</div>",
                unsafe_allow_html=True)

    if "shap_event" not in st.session_state:
        st.info("Upload a CSV and click **Run Prediction + TimeSHAP** to see all three "
                "levels of TimeSHAP explanations here.")
    else:
        # ① Event-level
        st.markdown("<div class='shap-label'>① Event-level TimeSHAP</div>",
                    unsafe_allow_html=True)
        st.markdown("""<div class='shap-desc'>
            SHAP value for each of the 21 time frames (0–100 ms post-contact).
            <b>Red</b> bars increase predicted ACL load; <b>blue</b> bars decrease it.
            The lower panel shows where the model's attention focused.
        </div>""", unsafe_allow_html=True)
        st.pyplot(plot_event_level(st.session_state["shap_event"],
                                   st.session_state["attn_np"]),
                  clear_figure=True)
        st.caption("Fig 1. Local event-level TimeSHAP + temporal attention weights.")

        st.markdown("<hr style='border:none;border-top:1px solid #E5EAF0;margin:10px 0 4px;'>",
                    unsafe_allow_html=True)

        # ② Feature-level
        st.markdown("<div class='shap-label'>② Feature-level TimeSHAP</div>",
                    unsafe_allow_html=True)
        st.markdown("""<div class='shap-desc'>
            Each feature's contribution computed by masking it across <i>all</i> time frames.
            Sorted by absolute impact magnitude.
        </div>""", unsafe_allow_html=True)
        st.pyplot(plot_feature_level(st.session_state["shap_feat"]),
                  clear_figure=True)
        st.caption("Fig 2. Local feature-level TimeSHAP — importance aggregated over time.")

        st.markdown("<hr style='border:none;border-top:1px solid #E5EAF0;margin:10px 0 4px;'>",
                    unsafe_allow_html=True)

        # ③ Cell-level
        st.markdown("<div class='shap-label'>③ Cell-level TimeSHAP</div>",
                    unsafe_allow_html=True)
        st.markdown("""<div class='shap-desc'>
            Full Feature × Time interaction heatmap. Each cell = SHAP value when that
            (feature, frame) pair is masked.
            <b>Red</b> → toward High Load; <b>Blue</b> → toward Low Load.
        </div>""", unsafe_allow_html=True)
        st.pyplot(plot_cell_level(st.session_state["shap_cell"]),
                  clear_figure=True)
        st.caption("Fig 3. Local cell-level TimeSHAP heatmap (Feature × Time).")


# ─────────────────────────────────────────────
# 7. Footer
# ─────────────────────────────────────────────
st.markdown("""
<br><hr>
<div style='color:#95A5A6;font-size:0.78rem;font-family:Times New Roman;'>
CNN-LSTM-Attention ACL Load Predictor &nbsp;|&nbsp;
TimeSHAP: Bento et al. (2021) NeurIPS &nbsp;|&nbsp;
Reference: Zhang et al. (2026). DOI: 10.1016/j.jsams.2026.04.01
</div>
""", unsafe_allow_html=True)
