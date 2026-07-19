import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import copy
import matplotlib
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import altair as alt

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
    cfg        = json.load(open("model_config.json"))
    threshold  = json.load(open("acl_threshold.json"))["acl_threshold"]
    scaler_X   = joblib.load("scaler_X.pkl")
    scaler_y   = joblib.load("scaler_y_acl.pkl")

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

DEFAULT_CSV_PATH = "Data.csv"  # Default sample dataset path; must be placed in the same directory as app.py

try:
    model, scaler_X, scaler_y, ACL_THRESHOLD, cfg = load_assets()
    FEATURE_COLS = list(cfg["feature_cols"])
    TIME_AXIS    = np.asarray(cfg["time_axis_ms"], dtype=float)
    N_TIMEPOINTS = int(tuple(cfg["input_shape"])[0])
    N_FEATURES   = int(tuple(cfg["input_shape"])[1])
    assets_ok    = True
except Exception as e:
    st.error(f"Asset loading failed: {e}")
    assets_ok = False

# ─────────────────────────────────────────────
# 4. Official TimeSHAP helpers
# ─────────────────────────────────────────────
class ACLTimeSHAPPredictor:
    """
    Prediction-only wrapper for TimeSHAP.
    Input:  [n_samples, time_steps, n_features]
    Output: [n_samples] on the model's scaled ACL-output scale.
    """
    def __init__(self, model, output_index=0):
        self.model = model
        self.output_index = output_index

    def __call__(self, X_np):
        X_np = np.asarray(X_np).astype(np.float32)
        if X_np.ndim == 2:
            X_np = np.expand_dims(X_np, axis=0)
        assert X_np.ndim == 3, "TimeSHAP input must be [n_samples, time_steps, n_features]"

        self.model.eval()
        preds = []
        batch_size = 256
        with torch.no_grad():
            for start in range(0, X_np.shape[0], batch_size):
                xb = torch.tensor(X_np[start:start + batch_size], dtype=torch.float32)
                out = self.model(xb)
                if isinstance(out, (tuple, list)):
                    out = out[self.output_index]
                out = out.detach().cpu().numpy()
                out = np.asarray(out).reshape(out.shape[0], -1)[:, 0]
                preds.append(out)
        return np.concatenate(preds, axis=0)


def _scale_sequence(X_seq):
    X_seq = np.asarray(X_seq).astype(np.float32)
    return scaler_X.transform(X_seq.reshape(-1, N_FEATURES)).reshape(N_TIMEPOINTS, N_FEATURES).astype(np.float32)


def predict_single(X_seq):
    X_sc = _scale_sequence(X_seq).reshape(1, N_TIMEPOINTS, N_FEATURES)
    t = torch.tensor(X_sc, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled, _ = model(t)
    pred_real = scaler_y.inverse_transform(pred_scaled.detach().cpu().numpy())[0, 0]
    return float(pred_real)


@st.cache_resource
def get_timeshap_kernel_class():
    try:
        import shap
        import shap.explainers._kernel as shap_kernel
        if not hasattr(shap_kernel, "Kernel"):
            if hasattr(shap_kernel, "KernelExplainer"):
                shap_kernel.Kernel = shap_kernel.KernelExplainer
            else:
                raise ImportError("Current SHAP version has neither Kernel nor KernelExplainer.")
        from timeshap.explainer.kernel import TimeShapKernel
        return TimeShapKernel, None
    except Exception as e:
        return None, str(e)


def load_baseline_sequence(X_sc_current):
    """
    Preferred: use the median training sequence saved from model development.
    Supported files, in order:
      1) timeshap_baseline_sequence.npy / baseline_sequence.npy / baseline_sequence_acl.npy
      2) X_train_scaled.npy / X_train_acl_scaled.npy / X_train.npy, then median over samples.
    Fallback: median of the uploaded sequence repeated across time.
    """
    baseline_files = [
        cfg.get("timeshap_baseline_file") if isinstance(cfg, dict) else None,
        "timeshap_baseline_sequence.npy",
        "baseline_sequence.npy",
        "baseline_sequence_acl.npy",
    ]
    train_files = [
        cfg.get("x_train_scaled_file") if isinstance(cfg, dict) else None,
        "X_train_scaled.npy",
        "X_train_acl_scaled.npy",
        "X_train.npy",
    ]

    for p in [x for x in baseline_files if x]:
        if os.path.exists(p):
            arr = np.load(p).astype(np.float32)
            if arr.shape == (N_TIMEPOINTS, N_FEATURES):
                return arr, f"baseline file: {p}"
            if arr.ndim == 3 and arr.shape[1:] == (N_TIMEPOINTS, N_FEATURES):
                return np.median(arr, axis=0).astype(np.float32), f"median from baseline file: {p}"

    for p in [x for x in train_files if x]:
        if os.path.exists(p):
            arr = np.load(p).astype(np.float32)
            if arr.ndim == 3 and arr.shape[1:] == (N_TIMEPOINTS, N_FEATURES):
                return np.median(arr, axis=0).astype(np.float32), f"median from training file: {p}"

    fallback = np.tile(np.median(X_sc_current, axis=0), (N_TIMEPOINTS, 1)).astype(np.float32)
    return fallback, "fallback: uploaded-sequence median repeated across time"


def _ensure_single_sample(x):
    x = np.asarray(x).astype(np.float32)
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    assert x.ndim == 3 and x.shape[0] == 1, "Input must be one sample: [1, T, F]"
    return x


def compute_event_timeshap_single(f, sample, baseline, TimeShapKernel,
                                  random_seed=42, nsamples=None, pruning_idx=0):
    sample = _ensure_single_sample(sample)
    if nsamples is None:
        nsamples = 2 * sample.shape[1] + 2048

    explainer = TimeShapKernel(model=f, background=baseline, rs=random_seed, mode="event")
    shap_values = explainer.shap_values(sample, pruning_idx=pruning_idx, nsamples=nsamples)
    shap_values = np.asarray(shap_values).reshape(-1)

    if len(shap_values) == sample.shape[1]:
        shap_values_natural = shap_values[::-1]
        time_indices = np.arange(sample.shape[1])
    else:
        non_pruned = shap_values[:-1] if pruning_idx > 0 else shap_values
        shap_values_natural = non_pruned[::-1]
        time_indices = np.arange(sample.shape[1] - len(shap_values_natural), sample.shape[1])

    return pd.DataFrame({
        "time_index": time_indices,
        "time_ms": TIME_AXIS[time_indices],
        "shap_value": shap_values_natural,
        "sample_index": 0,
    })


def compute_feature_timeshap_single(f, sample, baseline, feature_names, TimeShapKernel,
                                    random_seed=42, nsamples=None, pruning_idx=0):
    sample = _ensure_single_sample(sample)
    if nsamples is None:
        nsamples = 2 * sample.shape[2] + 2048

    explainer = TimeShapKernel(model=f, background=baseline, rs=random_seed, mode="feature")
    shap_values = explainer.shap_values(sample, pruning_idx=pruning_idx, nsamples=nsamples)
    shap_values = np.asarray(shap_values).reshape(-1)

    if len(shap_values) > len(feature_names):
        names = list(feature_names) + ["Pruned Events"]
    else:
        names = list(feature_names)

    return pd.DataFrame({
        "feature": names[:len(shap_values)],
        "shap_value": shap_values,
        "sample_index": 0,
    })


def compute_cell_timeshap_single(f, sample, baseline, selected_time_indices,
                                 selected_feature_indices, feature_names, TimeShapKernel,
                                 random_seed=42, nsamples=None, pruning_idx=0):
    sample = _ensure_single_sample(sample)
    selected_time_indices = list(map(int, selected_time_indices))
    selected_feature_indices = list(map(int, selected_feature_indices))

    if nsamples is None:
        n_cells = len(selected_time_indices) * len(selected_feature_indices)
        nsamples = 2 * n_cells + 2048

    explainer = TimeShapKernel(
        model=f,
        background=baseline,
        rs=random_seed,
        mode="cell",
        varying=(selected_time_indices, selected_feature_indices)
    )
    shap_values = explainer.shap_values(sample, pruning_idx=pruning_idx, nsamples=nsamples)
    shap_values = np.asarray(shap_values).reshape(-1)

    rows = []
    k = 0
    for ti in selected_time_indices:
        for fi in selected_feature_indices:
            rows.append({
                "time_index": ti,
                "time_ms": TIME_AXIS[ti],
                "feature_index": fi,
                "feature": feature_names[fi],
                "shap_value": shap_values[k],
                "sample_index": 0,
            })
            k += 1

    if k < len(shap_values):
        for j in range(k, len(shap_values)):
            rows.append({
                "time_index": np.nan,
                "time_ms": np.nan,
                "feature_index": np.nan,
                "feature": f"Special group {j-k+1}",
                "shap_value": shap_values[j],
                "sample_index": 0,
            })
    return pd.DataFrame(rows)


def format_local_event_for_official_plot(local_event_df, n_timepoints):
    df = local_event_df.copy().sort_values("time_index")
    df["Feature"] = df["time_ms"].apply(lambda x: f"Event {x:.0f} ms")
    df["Shapley Value"] = df["shap_value"].astype(float)
    return df[["Feature", "Shapley Value", "time_index", "time_ms"]].copy()


def format_local_feature_for_official_plot(local_feature_df):
    df = local_feature_df.copy()
    df = df[df["feature"].isin(FEATURE_COLS)].copy()
    df["Feature"] = df["feature"].astype(str)
    df["Shapley Value"] = df["shap_value"].astype(float)
    df["sort_col"] = df["Shapley Value"].abs()
    return df[["Feature", "Shapley Value", "sort_col"]].copy()


def format_local_cell_for_official_plot(local_cell_df):
    df = local_cell_df.copy()
    df = df.dropna(subset=["time_index", "feature_index"]).copy()
    df["time_index"] = df["time_index"].astype(int)
    df["feature_index"] = df["feature_index"].astype(int)
    df["time_ms"] = df["time_ms"].astype(float)
    df["Event"] = df["time_ms"].apply(lambda x: f"Event {x:.0f} ms")
    df["Feature"] = df["feature"].astype(str)
    df["Shapley Value"] = df["shap_value"].astype(float)
    return df[["Event", "Feature", "Shapley Value", "time_index", "time_ms", "feature_index"]].copy()


def run_official_timeshap(X_seq):
    TimeShapKernel, import_error = get_timeshap_kernel_class()
    if TimeShapKernel is None:
        raise ImportError(
            "TimeSHAP is not available. Install required packages, e.g. `pip install timeshap shap==0.42.1`. "
            f"Original error: {import_error}"
        )

    X_sc = _scale_sequence(X_seq)
    sample = X_sc[np.newaxis, :, :]
    baseline_sequence, baseline_note = load_baseline_sequence(X_sc)
    timeshap_f = ACLTimeSHAPPredictor(model=model, output_index=0)

    RANDOM_SEED = 42
    PRUNING_IDX = 0
    EVENT_NSAMPLES = 2 * N_TIMEPOINTS + 2048
    FEATURE_NSAMPLES = 2 * N_FEATURES + 2048
    CELL_NSAMPLES = int(cfg.get("cell_nsamples", 4096)) if isinstance(cfg, dict) else 4096

    event_raw = compute_event_timeshap_single(
        f=timeshap_f,
        sample=sample,
        baseline=baseline_sequence,
        TimeShapKernel=TimeShapKernel,
        random_seed=RANDOM_SEED,
        nsamples=EVENT_NSAMPLES,
        pruning_idx=PRUNING_IDX,
    )

    feature_raw = compute_feature_timeshap_single(
        f=timeshap_f,
        sample=sample,
        baseline=baseline_sequence,
        feature_names=FEATURE_COLS,
        TimeShapKernel=TimeShapKernel,
        random_seed=RANDOM_SEED,
        nsamples=FEATURE_NSAMPLES,
        pruning_idx=PRUNING_IDX,
    )

    local_event_official = format_local_event_for_official_plot(event_raw, N_TIMEPOINTS)
    local_feature_official = format_local_feature_for_official_plot(feature_raw)

    local_top_event_rows = (
        local_event_official
        .assign(abs_shap=lambda d: d["Shapley Value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(min(5, N_TIMEPOINTS))
    )
    local_top_time_indices = sorted(local_top_event_rows["time_index"].astype(int).tolist())

    local_top_feat_rows = (
        local_feature_official
        .assign(abs_shap=lambda d: d["Shapley Value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(min(5, N_FEATURES))
    )
    local_top_features = local_top_feat_rows["Feature"].tolist()
    local_top_feat_indices = [FEATURE_COLS.index(f) for f in local_top_features]

    cell_raw = compute_cell_timeshap_single(
        f=timeshap_f,
        sample=sample,
        baseline=baseline_sequence,
        selected_time_indices=local_top_time_indices,
        selected_feature_indices=local_top_feat_indices,
        feature_names=FEATURE_COLS,
        TimeShapKernel=TimeShapKernel,
        random_seed=RANDOM_SEED,
        nsamples=CELL_NSAMPLES,
        pruning_idx=PRUNING_IDX,
    )
    local_cell_official = format_local_cell_for_official_plot(cell_raw)

    unified_feature_order = local_top_features
    unified_event_order = [f"Event {float(TIME_AXIS[i]):.0f} ms" for i in local_top_time_indices]

    return {
        "event": local_event_official,
        "feature": local_feature_official,
        "cell": local_cell_official,
        "unified_feature_order": unified_feature_order,
        "unified_event_order": unified_event_order,
        "baseline_note": baseline_note,
    }

# ─────────────────────────────────────────────
# 5. Official-style plot functions
# ─────────────────────────────────────────────
def plot_local_event_heatmap_official_style(event_data: pd.DataFrame,
                                            sample_label="Uploaded sample",
                                            axis_lim=None,
                                            height=360,
                                            width=95):
    event_data = copy.deepcopy(event_data).sort_values("time_ms", ascending=True).copy()
    if axis_lim is None:
        max_abs = np.nanmax(np.abs(event_data["Shapley Value"].values))
        if max_abs == 0 or np.isnan(max_abs):
            max_abs = 1e-6
        axis_lim = [-max_abs, max_abs]

    c_range = ["#5f8fd6", "#99c3fb", "#f5f5f5", "#ffaa92", "#d16f5b"]
    event_data["rounded"] = event_data["Shapley Value"].apply(lambda x: round(float(x), 3))
    event_data["rounded_str"] = event_data["Shapley Value"].apply(
        lambda x: "0.000" if round(float(x), 3) == 0 else f"{float(x):.3f}"
    )
    event_data["column"] = 1
    sort_events = list(event_data["Feature"].values)

    base = alt.Chart(event_data).encode(
        y=alt.Y(
            "Feature:O",
            axis=alt.Axis(domain=False, labelFontSize=12, title="Event", titleFontSize=13, titleX=-45),
            sort=sort_events,
        )
    )

    rect = base.mark_rect().encode(
        x=alt.X("column:O", axis=alt.Axis(title="Shapley Value", titleFontSize=13, labels=False, domain=False)),
        color=alt.Color(
            "rounded:Q",
            title=None,
            legend=alt.Legend(gradientLength=height, gradientThickness=10, orient="right", labelFontSize=11),
            scale=alt.Scale(domain=axis_lim, range=c_range),
        ),
    )

    text = base.mark_text(align="right", baseline="middle", dx=22, fontSize=11, color="#798184").encode(
        x=alt.X("column:O", axis=alt.Axis(labels=False, title="Shapley Value", domain=False, titleX=52)),
        text="rounded_str:N",
    )

    return alt.layer(rect, text, data=event_data).properties(
        width=width,
        height=height,
        title=alt.TitleParams(text=f"Local event-level TimeSHAP | {sample_label}", anchor="start", fontSize=14, fontWeight="bold"),
    )


def plot_local_feature_barplot_official_style(feat_data: pd.DataFrame,
                                              sample_label="Uploaded sample",
                                              top_x_feats=None,
                                              axis_lim=None,
                                              height=300,
                                              width=300):
    feat_data = copy.deepcopy(feat_data)
    feat_data = feat_data[feat_data["Feature"].isin(FEATURE_COLS)].copy()
    feat_data["sort_col"] = feat_data["Shapley Value"].abs()

    if top_x_feats is not None and feat_data.shape[0] > top_x_feats:
        feat_data = feat_data.sort_values("sort_col", ascending=False).head(top_x_feats).copy()

    if axis_lim is None:
        max_abs = np.nanmax(np.abs(feat_data["Shapley Value"].values))
        if max_abs == 0 or np.isnan(max_abs):
            max_abs = 1e-6
        axis_lim = [-max_abs * 1.15, max_abs * 1.15]

    sort_features = feat_data.sort_values("sort_col", ascending=False)["Feature"].tolist()

    bars = alt.Chart(feat_data).mark_bar(size=15, thickness=1).encode(
        y=alt.Y(
            "Feature:O",
            axis=alt.Axis(title="Feature", labelFontSize=12, titleFontSize=13, titleX=-55),
            sort=sort_features,
        ),
        x=alt.X(
            "Shapley Value:Q",
            axis=alt.Axis(grid=True, title="Shapley Value", labelFontSize=11, titleFontSize=13),
            scale=alt.Scale(domain=axis_lim),
        ),
        color=alt.condition(alt.datum["Shapley Value"] >= 0, alt.value("#d16f5b"), alt.value("#5f8fd6")),
        tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Shapley Value:Q", format=".4f")],
    )

    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#798184", strokeWidth=1).encode(x="x:Q")

    return (bars + zero_line).properties(
        width=width,
        height=height,
        title=alt.TitleParams(text=f"Local feature-level TimeSHAP | {sample_label}", anchor="start", fontSize=14, fontWeight="bold"),
    )


def plot_local_cell_level_final(cell_data: pd.DataFrame,
                                unified_feature_order: list,
                                unified_event_order: list,
                                sample_label="Uploaded sample",
                                axis_lim=None,
                                height=260,
                                width=320,
                                fontsize=11):
    cell_data = cell_data[
        cell_data["Feature"].isin(unified_feature_order) &
        cell_data["Event"].isin(unified_event_order)
    ].copy()

    if cell_data.empty:
        raise ValueError("cell_data is empty after filtering with unified orders.")

    if axis_lim is None:
        max_abs = np.nanmax(np.abs(cell_data["Shapley Value"].values))
        max_abs = max_abs if (max_abs > 0 and not np.isnan(max_abs)) else 1e-6
        axis_lim = [-max_abs, max_abs]

    c_range = ["#5f8fd6", "#99c3fb", "#f5f5f5", "#ffaa92", "#d16f5b"]
    cell_data["rounded"] = cell_data["Shapley Value"].apply(lambda x: round(float(x), 3))
    cell_data["rounded_str"] = cell_data["Shapley Value"].apply(
        lambda x: "0.000" if round(float(x), 3) == 0 else f"{float(x):.3f}"
    )

    base = alt.Chart(cell_data).encode(
        y=alt.Y(
            "Feature:O",
            sort=unified_feature_order,
            axis=alt.Axis(domain=False, labelFontSize=fontsize, title=None),
        )
    )

    rect = base.mark_rect().encode(
        x=alt.X(
            "Event:O",
            sort=unified_event_order,
            axis=alt.Axis(orient="top", title="Shapley Value", titleFontSize=13,
                          titleY=height + 20, labelAngle=30, labelFontSize=fontsize),
        ),
        color=alt.Color(
            "rounded:Q",
            title=None,
            legend=alt.Legend(gradientLength=height, gradientThickness=10, orient="right", labelFontSize=fontsize),
            scale=alt.Scale(domain=axis_lim, range=c_range),
        ),
        tooltip=[alt.Tooltip("Event:N"), alt.Tooltip("Feature:N"), alt.Tooltip("Shapley Value:Q", format=".4f")],
    )

    text = base.mark_text(align="right", baseline="middle", dx=18, fontSize=fontsize, color="#798184").encode(
        x=alt.X("Event:O", sort=unified_event_order,
                axis=alt.Axis(orient="top", title=None, domain=False, labelAngle=30, labelFontSize=fontsize)),
        text="rounded_str:N",
    )

    return alt.layer(rect, text, data=cell_data).properties(
        width=width,
        height=height,
        title=alt.TitleParams(text=f"Local cell-level TimeSHAP | {sample_label}", anchor="start", fontSize=14, fontWeight="bold"),
    )

# ─────────────────────────────────────────────
# 6. Page layout
# ─────────────────────────────────────────────
st.markdown("<h1 class='sci-title'>ACL Peak Force Prediction — CNN-LSTM-Attention</h1>", unsafe_allow_html=True)
st.markdown("<p class='sci-subtitle'>Time-series biomechanical input · Deep learning regression · local TimeSHAP interpretability</p>", unsafe_allow_html=True)

if not assets_ok:
    st.stop()

col_left, col_right = st.columns([1, 2.1], gap="large")

# ── LEFT ──────────────────────────────────────
with col_left:
    st.markdown("<div class='section-header'>📂 Upload Time-Series CSV</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='upload-box'>
    <b>Format:</b> {N_TIMEPOINTS} rows × {N_FEATURES} columns<br>
    <b>Required columns:</b><br>
    <code style='font-size:0.82rem'>{', '.join(FEATURE_COLS)}</code>
    </div>
    """, unsafe_allow_html=True)

    data_source = st.radio(
        "Data source",
        options=["Use Default Sample Data", "Upload My CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded = None
    if data_source == "Upload My CSV":
        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    def _load_and_validate(df_up):
        """Validate column names and row count, return (X_seq, error_msg)"""
        missing = [c for c in FEATURE_COLS if c not in df_up.columns]
        if missing:
            return None, f"Missing columns: {missing}"
        if df_up.shape[0] != N_TIMEPOINTS:
            return None, f"Expected {N_TIMEPOINTS} rows, got {df_up.shape[0]}"
        return df_up[FEATURE_COLS].values.astype(np.float32), None

    X_seq = None
    if data_source == "Use Default Sample Data":
        if os.path.exists(DEFAULT_CSV_PATH):
            try:
                df_default = pd.read_csv(DEFAULT_CSV_PATH)
                X_seq, err = _load_and_validate(df_default)
                if err:
                    st.error(f"Default dataset error: {err}")
                else:
                    st.success(f"✅ Default sample data loaded — shape {X_seq.shape}")
            except Exception as e:
                st.error(f"Error reading default CSV: {e}")
        else:
            st.warning(f"Default dataset file `{DEFAULT_CSV_PATH}` not found. Please place it in the same directory as app.py, or switch to Upload My CSV.")
    elif uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            X_seq, err = _load_and_validate(df_up)
            if err:
                st.error(err)
            else:
                st.success(f"✅ Loaded — shape {X_seq.shape}")
                # Data preview table removed
        except Exception as e:
            st.error(f"CSV error: {e}")

    run_btn = st.button("▶  Run Prediction + TimeSHAP", use_container_width=True,
                        type="primary", disabled=(X_seq is None))

    if run_btn and X_seq is not None:
        with st.spinner("Computing prediction + official TimeSHAP levels..."):
            pred_val = predict_single(X_seq)
            ts_results = run_official_timeshap(X_seq)
        st.session_state.update({
            "pred_val": pred_val,
            "timeshap_event_official": ts_results["event"],
            "timeshap_feature_official": ts_results["feature"],
            "timeshap_cell_official": ts_results["cell"],
            "unified_feature_order": ts_results["unified_feature_order"],
            "unified_event_order": ts_results["unified_event_order"],
            "baseline_note": ts_results["baseline_note"],
            "X_seq": X_seq,
        })

    if "pred_val" in st.session_state:
        pv = st.session_state["pred_val"]
        is_high = pv >= ACL_THRESHOLD
        s_color = "#C0392B" if is_high else "#1E8449"
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
                       f"Input: {N_TIMEPOINTS} frames × {N_FEATURES} joint features  |  "
                       "Output: peak ACL load (inverse-transformed)")
            st.caption(f"TimeSHAP baseline: {st.session_state.get('baseline_note', 'not available')}")

        df_exp = pd.DataFrame(st.session_state["X_seq"], columns=FEATURE_COLS)
        df_exp.insert(0, "frame_ms", TIME_AXIS)
        df_exp["pred_ACL_peak"] = pv
        df_exp["risk"] = s_label
        st.download_button("📥 Export Report CSV",
                           data=df_exp.to_csv(index=False).encode(),
                           file_name="acl_cnn_report.csv",
                           use_container_width=True)

# ── RIGHT: three TimeSHAP plots in one interface ───────────
with col_right:
    st.markdown("<div class='section-header'>🔍 Official-style Local TimeSHAP Interpretability</div>", unsafe_allow_html=True)

    if "timeshap_event_official" not in st.session_state:
        st.info("Upload a CSV and click **Run Prediction + Official TimeSHAP** to see the three TimeSHAP plots in the same interface.")
    else:
        st.markdown("""<div class='shap-desc'>
            These plots use the official TimeSHAP kernel route: event-level, feature-level, and recomputed local cell-level explanations.
            The event panel no longer includes temporal attention weights.
        </div>""", unsafe_allow_html=True)

        sample_label = "Uploaded sample"
        event_plot = plot_local_event_heatmap_official_style(
            st.session_state["timeshap_event_official"],
            sample_label=sample_label,
            height=360,
            width=95,
        )
        feature_plot = plot_local_feature_barplot_official_style(
            st.session_state["timeshap_feature_official"],
            sample_label=sample_label,
            top_x_feats=len(FEATURE_COLS),
            height=max(260, 28 * len(FEATURE_COLS)),
            width=300,
        )
        cell_plot = plot_local_cell_level_final(
            st.session_state["timeshap_cell_official"],
            unified_feature_order=st.session_state["unified_feature_order"],
            unified_event_order=st.session_state["unified_event_order"],
            sample_label=sample_label,
            height=260,
            width=320,
            fontsize=11,
        )

        combined_plot = alt.hconcat(event_plot, feature_plot, cell_plot).resolve_scale(color="independent").properties(
            title=alt.TitleParams(
                text=f"Local TimeSHAP explanation | Predicted ACL load = {st.session_state['pred_val']:.3f} ×BW",
                anchor="start",
                fontSize=16,
                fontWeight="bold",
            )
        )
        st.altair_chart(combined_plot, use_container_width=True)

# ─────────────────────────────────────────────
# 7. Footer
# ─────────────────────────────────────────────
st.markdown("""
<br><hr>
<div style='color:#95A5A6;font-size:0.78rem;font-family:Times New Roman;'>
CNN-LSTM-Attention ACL Load Predictor &nbsp;|&nbsp;
Official-style TimeSHAP local explanation &nbsp;|&nbsp;
Event-level attention-weight panel removed
</div>
""", unsafe_allow_html=True)
