"""
E-Nose Batch PCA + SVM App using MATLAB-style R0 normalization and features.

Input list file format, Excel or CSV:
    Column 1: data type, "train" or "valid"
    Column 2: path to e-nose CSV file
    Column 3: integer class label

Each e-nose CSV contains A/M phases and gas columns:
    State, OV, Alc, H2, AC1, NH3, AC2, VOC, LP, ...

For every A->M pair:
    1) Convert raw ADS1256 ADC values to sensor resistance.
    2) Estimate R0 from the final window of the previous A phase.
    3) Normalize M phase: xM = (R_M - R0) / R0.
    4) Extract transient and quasi-steady features.

If one CSV has multiple A->M pairs, the app represents that file by:
    first pair, last pair, specified pair, or average of all pairs.

Tab 1 is the live TCP logger copied from the real-time logger app.
Tab 2 plots loaded/logged CSV files in the same style as plotAMPhasePairs.m.
Tab 3 trains PCA + SVM and exports a complete classification model package.
Tab 3 can optionally drop zero-variance features and time/slope/integration-related features before PCA.
When more than three PCA dimensions are trained, Tab 3 provides three PC selector drop-down lists for 2D/3D plotting.
Tab 4 provides neural-network classification with configurable feature extraction, optional PCA, hidden layers, and activation function.
The package contains:
    - feature extraction parameters
    - A-M pair selection mode
    - feature column order
    - training medians for imputation
    - StandardScaler
    - PCA model and number of reduced features
    - SVM classification model

Tab 4 trains a neural network classifier and exports a complete NN model package.
Tab 5 loads the exported classification model, accepts one or more testing CSV
files, allows optional true-label entry in the file table, then predicts and
summarizes classification results.

Install:
    pip install PyQt5 pyqtgraph numpy pandas openpyxl scikit-learn matplotlib joblib

Run:
    python enose_pyqt_pca_svm_matlab_features_v7.py
"""

import csv
import os
import socket
import sys
import time
import math
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DEFAULT_HOST = "192.168.4.1"
DEFAULT_PORT = 5000
DEFAULT_GAS_COLUMNS = ["OV", "Alc", "H2", "AC1", "NH3", "AC2", "VOC", "LP"]
NON_GAS_COLUMNS = {
    "state", "phase", "temp", "temperature", "hud", "humid", "humidity",
    "airflow", "air_flow", "flow", "time", "timestamp", "timestamp_ms",
    "sample", "index",
}

FEATURE_NAMES = [
    "validFraction",
    "earlyMean",
    "earlyMedian",
    "initialSlope",
    "lateMean",
    "lateMedian",
    "lateStd",
    "lateSlope",
    "endValue",
    "maxValue",
    "minValue",
    "rangeValue",
    "absPeak",
    "timeToMax",
    "timeToMin",
    "timeToAbsPeak",
    "areaTotal",
    "areaAbs",
    "overshootPositive",
    "overshootNegative",
    "earlyToLateDiff",
]

FEATURE_DEFINITIONS = {
    "validFraction": "Fraction of finite normalized M-phase samples.",
    "earlyMean": "Mean of normalized response in the early M window.",
    "earlyMedian": "Median of normalized response in the early M window.",
    "initialSlope": "Linear slope of normalized response in the early M window.",
    "lateMean": "Mean of normalized response in the late M window.",
    "lateMedian": "Median of normalized response in the late M window.",
    "lateStd": "Standard deviation of normalized response in the late M window.",
    "lateSlope": "Linear slope of normalized response in the late M window.",
    "endValue": "Last valid normalized response value in M phase.",
    "maxValue": "Maximum normalized response during M phase.",
    "minValue": "Minimum normalized response during M phase.",
    "rangeValue": "maxValue - minValue during M phase.",
    "absPeak": "Maximum absolute normalized response during M phase.",
    "timeToMax": "Time to maximum normalized response.",
    "timeToMin": "Time to minimum normalized response.",
    "timeToAbsPeak": "Time to maximum absolute normalized response.",
    "areaTotal": "Signed area of normalized response, trapz(t, xM).",
    "areaAbs": "Absolute area of normalized response, trapz(t, abs(xM)).",
    "overshootPositive": "maxValue - lateMedian.",
    "overshootNegative": "lateMedian - minValue.",
    "earlyToLateDiff": "earlyMean - lateMedian.",
}


@dataclass
class PhaseSegment:
    phase: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class ProcessingSettings:
    ts: float = 0.5
    rref: float = 1000.0
    vs: float = 5.0
    vref_fs: float = 5.0
    adc_bits: int = 24
    adc_mode: str = "twoscomp"
    resistance_unit: str = "ohm"
    r0_method: str = "median"
    r0_win_sec: float = 5.0
    trim_percent: float = 20.0
    min_samples: int = 5
    early_sec: float = 5.0
    late_sec: float = 5.0
    pair_mode: str = "first"  # first, last, specified, average_all
    specified_pair: int = 1    # 1-based


class TcpClientWorker(QtCore.QThread):
    """TCP receiver thread for the ESP32-S3 e-nose logger."""

    line_received = QtCore.pyqtSignal(str)
    status_changed = QtCore.pyqtSignal(str)
    connected_changed = QtCore.pyqtSignal(bool)
    stream_ended = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, host: str, port: int, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = int(port)
        self._stop_requested = False
        self._sock: Optional[socket.socket] = None

    def stop(self):
        self._stop_requested = True
        try:
            if self._sock is not None:
                self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass

    def _emit_complete_lines(self, text_buffer: str) -> str:
        while "\n" in text_buffer:
            line, text_buffer = text_buffer.split("\n", 1)
            line = line.strip("\r")
            if line.strip():
                self.line_received.emit(line)
        return text_buffer

    def run(self):
        text_buffer = ""
        try:
            self.status_changed.emit(f"Connecting to {self.host}:{self.port} ...")
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)
            self._sock.connect((self.host, self.port))
            self._sock.settimeout(0.5)
            self.connected_changed.emit(True)
            self.status_changed.emit(f"Connected to {self.host}:{self.port}")

            while not self._stop_requested:
                try:
                    chunk = self._sock.recv(4096)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not self._stop_requested:
                        self.error_occurred.emit(f"Socket error: {exc}")
                    break

                if not chunk:
                    if text_buffer.strip():
                        self.line_received.emit(text_buffer.strip())
                    text_buffer = ""
                    self.stream_ended.emit("ESP32 closed the connection")
                    break

                text_buffer += chunk.decode("utf-8", errors="ignore")

                if "#END" in text_buffer:
                    before_end, _after_end = text_buffer.split("#END", 1)
                    before_end = self._emit_complete_lines(before_end)
                    if before_end.strip():
                        self.line_received.emit(before_end.strip())
                    text_buffer = ""
                    self.stream_ended.emit("Measurement completed (#END received)")
                    break

                text_buffer = self._emit_complete_lines(text_buffer)

        except Exception as exc:
            if not self._stop_requested:
                self.error_occurred.emit(str(exc))
        finally:
            try:
                if self._sock is not None:
                    self._sock.close()
            except OSError:
                pass
            self._sock = None
            self.connected_changed.emit(False)


class GasPlot:
    """One live pyqtgraph plot with A/M shaded regions."""

    def __init__(self, name: str, y_label: Optional[str] = None):
        self.name = name
        self.widget = pg.PlotWidget(title=name)
        self.widget.showGrid(x=True, y=True, alpha=0.25)
        self.widget.setLabel("bottom", "Sample")
        self.widget.setLabel("left", y_label if y_label else name)
        self.widget.setMouseEnabled(x=True, y=True)
        self.widget.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.curve = self.widget.plot([], [], pen=pg.mkPen(width=2))
        self.phase_regions: List[pg.LinearRegionItem] = []

    def add_phase_region(self, start_x: float, end_x: float, phase: str) -> pg.LinearRegionItem:
        if phase == "A":
            brush = QtGui.QBrush(QtGui.QColor(80, 160, 255, 35))
        elif phase == "M":
            brush = QtGui.QBrush(QtGui.QColor(255, 170, 30, 40))
        else:
            brush = QtGui.QBrush(QtGui.QColor(180, 180, 180, 25))
        region = pg.LinearRegionItem(values=(start_x, end_x), movable=False, brush=brush)
        region.setZValue(-100)
        for line in region.lines:
            line.setPen(pg.mkPen(None))
        self.widget.addItem(region)
        self.phase_regions.append(region)
        return region


# -------------------------------------------------------------------------
# Core signal processing: ADC -> resistance -> R0 -> normalized features
# -------------------------------------------------------------------------

def normalize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    state_col = find_state_column(out)
    if state_col is None:
        raise ValueError("CSV file does not contain a State or Phase column.")
    if state_col != "State":
        out.rename(columns={state_col: "State"}, inplace=True)
    out["State"] = out["State"].astype(str).str.strip().str.upper()
    for col in out.columns:
        if col != "State":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def find_state_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if str(col).strip().lower() in {"state", "phase"}:
            return str(col)
    return None


def detect_gas_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        norm = str(col).strip().lower().replace(" ", "_")
        if norm not in NON_GAS_COLUMNS and str(col) != "State":
            if pd.api.types.is_numeric_dtype(df[col]):
                cols.append(str(col))
    ordered = [name for name in DEFAULT_GAS_COLUMNS if name in cols]
    ordered += [name for name in cols if name not in ordered]
    return ordered


def phase_segments(phase: Sequence[str]) -> List[PhaseSegment]:
    if len(phase) == 0:
        return []
    states = [str(x).strip().upper() for x in phase]
    segments: List[PhaseSegment] = []
    current = states[0]
    start = 0
    for i in range(1, len(states)):
        if states[i] != current:
            segments.append(PhaseSegment(current, start, i - 1))
            current = states[i]
            start = i
    segments.append(PhaseSegment(current, start, len(states) - 1))
    return segments


def am_pair_segments(phase: Sequence[str]) -> List[Tuple[PhaseSegment, PhaseSegment]]:
    segs = phase_segments(phase)
    pairs: List[Tuple[PhaseSegment, PhaseSegment]] = []
    for i in range(len(segs) - 1):
        if segs[i].phase == "A" and segs[i + 1].phase == "M":
            pairs.append((segs[i], segs[i + 1]))
    return pairs


def twoscomp_to_signed(raw: np.ndarray, bits: int) -> np.ndarray:
    raw = raw.astype(float)
    out = raw.copy()
    threshold = 2 ** (bits - 1)
    full = 2 ** bits
    neg = out >= threshold
    out[neg] = out[neg] - full
    return out


def adc_to_voltage(adc: np.ndarray, settings: ProcessingSettings) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ADC code to voltage.

    For ADS1256 raw SPI output, use settings.adc_mode = "twoscomp".
    settings.vref_fs is the full-scale voltage magnitude. For ADS1256 with
    VREF=2.5 V and PGA=1, the input range is approximately ±5 V, so vref_fs=5.
    """
    raw = adc.astype(float)
    mode = str(settings.adc_mode).lower()
    bits = int(settings.adc_bits)

    if mode == "twoscomp":
        signed = twoscomp_to_signed(raw, bits)
        adc_max_pos = 2 ** (bits - 1) - 1
        v = signed / adc_max_pos * float(settings.vref_fs)
        invalid = (raw < 0) | (raw > (2 ** bits - 1))
    elif mode == "signed":
        adc_max_pos = 2 ** (bits - 1) - 1
        adc_min_neg = -(2 ** (bits - 1))
        v = raw / adc_max_pos * float(settings.vref_fs)
        invalid = (raw < adc_min_neg) | (raw > adc_max_pos)
    elif mode == "unsigned":
        adc_max = 2 ** bits - 1
        v = raw / adc_max * float(settings.vref_fs)
        invalid = (raw < 0) | (raw > adc_max)
    else:
        raise ValueError("ADC mode must be 'twoscomp', 'signed', or 'unsigned'.")

    return v, invalid


def adc_to_resistance_matrix(df: pd.DataFrame, gas_cols: List[str], settings: ProcessingSettings) -> Tuple[np.ndarray, np.ndarray]:
    adc = df[gas_cols].to_numpy(dtype=float)
    v_adc, invalid_adc = adc_to_voltage(adc, settings)
    r = settings.rref * (settings.vs / v_adc - 1.0)
    invalid = invalid_adc | (v_adc <= 0) | (v_adc > settings.vs) | (~np.isfinite(r)) | (r < 0)
    r[invalid] = np.nan
    if str(settings.resistance_unit).lower() in {"kohm", "kω", "kiloohm"}:
        r = r / 1000.0
    return r, v_adc


def finite_median(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def finite_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def finite_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if x.size >= 2 else float("nan")


def finite_slope(t: np.ndarray, x: np.ndarray, min_samples: int = 5) -> float:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(t) & np.isfinite(x)
    if valid.sum() < min_samples:
        return float("nan")
    tv = t[valid]
    xv = x[valid]
    if np.unique(tv).size < 2:
        return float("nan")
    try:
        return float(np.polyfit(tv, xv, 1)[0])
    except Exception:
        return float("nan")


def trimmed_mean(x: np.ndarray, trim_percent: float) -> float:
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x)])
    if x.size == 0:
        return float("nan")
    ntrim = int(math.floor((float(trim_percent) / 100.0) * x.size / 2.0))
    if 2 * ntrim >= x.size:
        return float(np.median(x))
    return float(np.mean(x[ntrim:x.size - ntrim]))


def robust_linear_huber(t: np.ndarray, y: np.ndarray, max_iter: int = 50) -> Tuple[float, float]:
    """Small Huber IRLS regression y = b0 + b1*t."""
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]
    if y.size < 2:
        return float("nan"), float("nan")
    t_mean = float(np.mean(t))
    tc = t - t_mean
    X = np.column_stack([np.ones_like(tc), tc])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except Exception:
        return float("nan"), float("nan")
    c = 1.345
    for _ in range(max_iter):
        r = y - X @ beta
        sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
        if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
            break
        u = r / sigma
        w = np.ones_like(u)
        idx = np.abs(u) > c
        w[idx] = c / np.abs(u[idx])
        W = np.sqrt(w)
        Xw = X * W[:, None]
        yw = y * W
        try:
            beta_new = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        except Exception:
            break
        if np.linalg.norm(beta_new - beta) < 1e-9 * (np.linalg.norm(beta) + np.finfo(float).eps):
            beta = beta_new
            break
        beta = beta_new
    b1 = float(beta[1])
    b0 = float(beta[0] - beta[1] * t_mean)
    return b0, b1


def estimate_r0_from_a(r_a: np.ndarray, ts: float, settings: ProcessingSettings) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate R0 and slope for each sensor from final A-phase window."""
    r_a = np.asarray(r_a, dtype=float)
    if r_a.ndim != 2:
        raise ValueError("A-phase resistance must be 2D: samples x sensors.")
    n_samples, n_sensors = r_a.shape
    nwin = min(n_samples, max(1, int(round(settings.r0_win_sec / ts))))
    start = n_samples - nwin
    idx = np.arange(start, n_samples)
    t_win = idx.astype(float) * ts
    t_pred = float(n_samples) * ts  # one sample after A local start = M boundary

    r0 = np.full(n_sensors, np.nan, dtype=float)
    slope = np.full(n_sensors, np.nan, dtype=float)
    slope_rel = np.full(n_sensors, np.nan, dtype=float)
    method = str(settings.r0_method).lower()

    for s in range(n_sensors):
        y = r_a[start:n_samples, s]
        valid = np.isfinite(y) & (y > 0)
        yv = y[valid]
        tv = t_win[valid]
        if yv.size < int(settings.min_samples):
            continue

        if method == "mean":
            r0_s = finite_mean(yv)
            slope_s = finite_slope(tv, yv, settings.min_samples)
        elif method == "median":
            r0_s = finite_median(yv)
            slope_s = finite_slope(tv, yv, settings.min_samples)
        elif method in {"trimmed", "trimmedmean"}:
            r0_s = trimmed_mean(yv, settings.trim_percent)
            slope_s = finite_slope(tv, yv, settings.min_samples)
        elif method == "linear":
            if np.unique(tv).size < 2:
                r0_s = float("nan")
                slope_s = float("nan")
            else:
                p = np.polyfit(tv, yv, 1)
                slope_s = float(p[0])
                r0_s = float(np.polyval(p, t_pred))
        elif method == "robustlinear":
            b0, b1 = robust_linear_huber(tv, yv)
            slope_s = b1
            r0_s = b0 + b1 * t_pred
        else:
            raise ValueError("R0 method must be median, mean, trimmedmean, linear, or robustlinear.")

        r0[s] = r0_s
        slope[s] = slope_s
        slope_rel[s] = slope_s / r0_s if np.isfinite(r0_s) and r0_s != 0 else np.nan

    return r0, slope, slope_rel


def compute_one_sensor_features(x: np.ndarray, t: np.ndarray, settings: ProcessingSettings) -> Dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    n = x.size
    out = {name: np.nan for name in FEATURE_NAMES}
    if n == 0:
        return out

    valid = np.isfinite(x)
    out["validFraction"] = float(valid.sum() / n)
    if valid.sum() < int(settings.min_samples):
        return out

    n_early = min(n, max(1, int(round(settings.early_sec / settings.ts))))
    n_late = min(n, max(1, int(round(settings.late_sec / settings.ts))))
    early = slice(0, n_early)
    late = slice(n - n_late, n)

    x_early = x[early]
    t_early = t[early]
    x_late = x[late]
    t_late = t[late]

    out["earlyMean"] = finite_mean(x_early)
    out["earlyMedian"] = finite_median(x_early)
    out["initialSlope"] = finite_slope(t_early, x_early, settings.min_samples)

    out["lateMean"] = finite_mean(x_late)
    out["lateMedian"] = finite_median(x_late)
    out["lateStd"] = finite_std(x_late)
    out["lateSlope"] = finite_slope(t_late, x_late, settings.min_samples)

    valid_idx = np.flatnonzero(valid)
    if valid_idx.size:
        out["endValue"] = float(x[valid_idx[-1]])
        x_valid = x[valid_idx]
        t_valid = t[valid_idx]
        max_i = int(np.argmax(x_valid))
        min_i = int(np.argmin(x_valid))
        abs_i = int(np.argmax(np.abs(x_valid)))
        out["maxValue"] = float(x_valid[max_i])
        out["minValue"] = float(x_valid[min_i])
        out["rangeValue"] = out["maxValue"] - out["minValue"]
        out["absPeak"] = float(abs(x_valid[abs_i]))
        out["timeToMax"] = float(t_valid[max_i])
        out["timeToMin"] = float(t_valid[min_i])
        out["timeToAbsPeak"] = float(t_valid[abs_i])
        if x_valid.size >= 2:
            out["areaTotal"] = float(np.trapz(x_valid, t_valid))
            out["areaAbs"] = float(np.trapz(np.abs(x_valid), t_valid))

    if np.isfinite(out["lateMedian"]):
        out["overshootPositive"] = out["maxValue"] - out["lateMedian"]
        out["overshootNegative"] = out["lateMedian"] - out["minValue"]
    if np.isfinite(out["earlyMean"]) and np.isfinite(out["lateMedian"]):
        out["earlyToLateDiff"] = out["earlyMean"] - out["lateMedian"]
    return out


def extract_pair_features_from_csv(csv_path: str, settings: ProcessingSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-pair features and a small metadata table for one CSV file."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df = normalize_input_dataframe(df)
    gas_cols = detect_gas_columns(df)
    if not gas_cols:
        raise ValueError(f"No gas sensor columns detected in {csv_path}")
    r, v_adc = adc_to_resistance_matrix(df, gas_cols, settings)
    phase = df["State"].astype(str).str.upper().to_numpy()
    pairs = am_pair_segments(phase)
    if not pairs:
        raise ValueError(f"No consecutive A->M phase pair found in {csv_path}")

    rows: List[Dict[str, object]] = []
    meta_rows: List[Dict[str, object]] = []
    for p_idx, (a_seg, m_seg) in enumerate(pairs, start=1):
        r_a = r[a_seg.start:a_seg.end + 1, :]
        r_m = r[m_seg.start:m_seg.end + 1, :]
        r0, r0_slope, r0_slope_rel = estimate_r0_from_a(r_a, settings.ts, settings)

        # Normalize M phase: xM = (R_M - R0) / R0
        x_m = (r_m - r0.reshape(1, -1)) / r0.reshape(1, -1)
        bad = (~np.isfinite(r0)) | (r0 <= 0)
        x_m[:, bad] = np.nan
        t_m = np.arange(r_m.shape[0], dtype=float) * settings.ts

        row: Dict[str, object] = {
            "PairNo": p_idx,
            "A_Start": a_seg.start + 1,
            "A_End": a_seg.end + 1,
            "M_Start": m_seg.start + 1,
            "M_End": m_seg.end + 1,
            "M_NumSamples": m_seg.length,
        }
        for s, gas in enumerate(gas_cols):
            safe = safe_name(gas)
            row[f"R0_{safe}"] = r0[s]
            row[f"R0Slope_{safe}"] = r0_slope[s]
            row[f"R0SlopeRelative_{safe}"] = r0_slope_rel[s]
            feats = compute_one_sensor_features(x_m[:, s], t_m, settings)
            for name in FEATURE_NAMES:
                row[f"{safe}_{name}"] = feats[name]

        rows.append(row)
        meta_rows.append({
            "PairNo": p_idx,
            "A_Start": a_seg.start + 1,
            "A_End": a_seg.end + 1,
            "M_Start": m_seg.start + 1,
            "M_End": m_seg.end + 1,
            "NumSensors": len(gas_cols),
            "GasColumns": ",".join(gas_cols),
        })

    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


def safe_name(name: str) -> str:
    text = str(name).strip()
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch == "_" else "_")
    s = "".join(out)
    if not s:
        s = "Sensor"
    if s[0].isdigit():
        s = "S_" + s
    return s


def represent_file_features(pair_df: pd.DataFrame, settings: ProcessingSettings) -> Tuple[pd.Series, str]:
    """Convert pair-level features into one file-level feature vector."""
    if pair_df.empty:
        raise ValueError("No pair features available.")
    mode = str(settings.pair_mode).lower()
    if mode == "first":
        row = pair_df.iloc[0].copy()
        return row, "first"
    if mode == "last":
        row = pair_df.iloc[-1].copy()
        return row, "last"
    if mode == "specified":
        idx = int(settings.specified_pair) - 1
        if idx < 0 or idx >= len(pair_df):
            raise ValueError(f"Specified A-M pair {settings.specified_pair} is outside available range 1 to {len(pair_df)}.")
        row = pair_df.iloc[idx].copy()
        return row, f"specified_{settings.specified_pair}"
    if mode in {"average_all", "average", "mean_all"}:
        numeric_cols = [c for c in pair_df.columns if pd.api.types.is_numeric_dtype(pair_df[c])]
        row = pair_df[numeric_cols].mean(axis=0, skipna=True)
        # Pair metadata is not meaningful after averaging; preserve number of pairs.
        row["PairNo"] = 0
        row["A_Start"] = pair_df["A_Start"].min()
        row["A_End"] = pair_df["A_End"].max()
        row["M_Start"] = pair_df["M_Start"].min()
        row["M_End"] = pair_df["M_End"].max()
        row["M_NumSamples"] = pair_df["M_NumSamples"].mean()
        return row, "average_all"
    raise ValueError("Pair mode must be first, last, specified, or average_all.")


def read_input_list(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, header=None)
    elif ext == ".csv":
        df = pd.read_csv(path, header=None)
    else:
        raise ValueError("Input list must be .xlsx, .xls, or .csv")
    if df.shape[1] < 3:
        raise ValueError("Input list must have at least 3 columns: train/valid, csv path, label.")
    df = df.iloc[:, :3].copy()
    df.columns = ["DataType", "FilePath", "Label"]
    # Skip a possible header row.
    first_type = str(df.iloc[0, 0]).strip().lower() if len(df) else ""
    if first_type in {"datatype", "data type", "type", "split"}:
        df = df.iloc[1:].reset_index(drop=True)
    df["DataType"] = df["DataType"].astype(str).str.strip().str.lower()
    df["FilePath"] = df["FilePath"].astype(str).str.strip()
    df["Label"] = pd.to_numeric(df["Label"], errors="raise").astype(int)
    if not df["DataType"].isin(["train", "valid", "validation", "val"]).all():
        raise ValueError("First column must contain only train or valid.")

    base = os.path.dirname(os.path.abspath(path))
    resolved = []
    for p in df["FilePath"]:
        if os.path.isabs(p):
            resolved.append(p)
        else:
            candidate = os.path.join(base, p)
            resolved.append(candidate if os.path.exists(candidate) else p)
    df["FilePath"] = resolved
    return df


def build_feature_table_from_input_list(list_path: str, settings: ProcessingSettings, log_callback=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spec = read_input_list(list_path)
    rows: List[pd.Series] = []
    pair_rows: List[pd.DataFrame] = []
    for i, rec in spec.iterrows():
        csv_path = str(rec["FilePath"])
        if log_callback:
            log_callback(f"Processing {i + 1}/{len(spec)}: {csv_path}")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        pair_df, _ = extract_pair_features_from_csv(csv_path, settings)
        selected, selected_desc = represent_file_features(pair_df, settings)
        selected = selected.copy()
        selected["DataType"] = str(rec["DataType"]).lower()
        selected["FilePath"] = csv_path
        selected["Label"] = int(rec["Label"])
        selected["NumPairs"] = int(len(pair_df))
        selected["PairSelection"] = selected_desc
        # Put metadata first by building later DataFrame and reordering.
        rows.append(selected)

        temp_pair_df = pair_df.copy()
        temp_pair_df.insert(0, "DataType", str(rec["DataType"]).lower())
        temp_pair_df.insert(1, "FilePath", csv_path)
        temp_pair_df.insert(2, "Label", int(rec["Label"]))
        pair_rows.append(temp_pair_df)

    feature_table = pd.DataFrame(rows)
    pair_table = pd.concat(pair_rows, ignore_index=True, sort=False) if pair_rows else pd.DataFrame()
    meta_cols = ["DataType", "FilePath", "Label", "NumPairs", "PairSelection", "PairNo", "A_Start", "A_End", "M_Start", "M_End", "M_NumSamples"]
    cols = [c for c in meta_cols if c in feature_table.columns] + [c for c in feature_table.columns if c not in meta_cols]
    feature_table = feature_table[cols]
    return feature_table, pair_table


def numeric_ml_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "DataType", "FilePath", "Label", "NumPairs", "PairSelection",
        "PairNo", "A_Start", "A_End", "M_Start", "M_End", "M_NumSamples",
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def is_time_related_feature_name(col: str) -> bool:
    """Return True for time, slope, and integration/area features.

    These features are useful when transient shape is important, but some
    classification experiments may intentionally remove them so that PCA/SVM
    depends mainly on response amplitude and steady-state features.

    Matched examples:
        OV_timeToMax, OV_timeToMin, OV_timeToAbsPeak
        OV_areaTotal, OV_areaAbs
        OV_initialSlope, OV_lateSlope
        R0Slope_OV, R0SlopeRelative_OV
    """
    name = str(col).lower()
    return (
        "timeto" in name
        or "area" in name
        or "slope" in name
    )


def clean_standardize_pca(
    feature_table: pd.DataFrame,
    n_components: int,
    drop_zero_variance: bool = True,
    drop_time_features: bool = False,
):
    data_type = feature_table["DataType"].astype(str).str.lower()
    train_mask = data_type.eq("train")
    valid_mask = data_type.isin(["valid", "validation", "val"])
    if not train_mask.any():
        raise ValueError("No training samples found in input list.")

    feature_cols = numeric_ml_columns(feature_table)
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    total_feature_count = len(feature_cols)

    # Optional feature-family dropout before imputation, standardization, and PCA.
    # This removes time-to-peak features, slope features, and integration/area
    # features from the candidate feature set.
    time_related_features = [c for c in feature_cols if is_time_related_feature_name(c)]
    dropped_time_related_features: List[str] = []
    if drop_time_features and time_related_features:
        dropped_time_related_features = time_related_features
        drop_set = set(time_related_features)
        feature_cols = [c for c in feature_cols if c not in drop_set]
        if not feature_cols:
            raise ValueError("All numeric features were removed by the time/integration feature dropout option.")

    feature_count_after_time_filter = len(feature_cols)

    # Keep all remaining numeric feature columns first. Missing/invalid values are
    # imputed from the training-set medians. Columns that are all missing in
    # training get a median value of zero. They become zero-variance features
    # and are dropped only when drop_zero_variance=True.
    X_all = feature_table[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    med = X_all.loc[train_mask].median(axis=0, skipna=True).fillna(0.0)
    X_all = X_all.fillna(med).fillna(0.0)

    X_train_raw = X_all.loc[train_mask].to_numpy(dtype=float)
    raw_std = np.std(X_train_raw, axis=0)
    zero_var_mask = raw_std <= np.finfo(float).eps
    zero_var_features = [c for c, is_zero in zip(feature_cols, zero_var_mask) if bool(is_zero)]
    dropped_zero_var_features: List[str] = []

    if drop_zero_variance and zero_var_features:
        keep_var = ~zero_var_mask
        dropped_zero_var_features = zero_var_features
        feature_cols = [c for c, keep in zip(feature_cols, keep_var) if bool(keep)]
        if not feature_cols:
            raise ValueError("All numeric features are zero-variance in the training data.")
        X_all = X_all.loc[:, feature_cols]
        med = med.loc[feature_cols]
        X_train_raw = X_all.loc[train_mask].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_all_scaled = scaler.transform(X_all.to_numpy(dtype=float))

    max_pc = min(X_train.shape[0], X_train.shape[1])
    if max_pc < 1:
        raise ValueError("Not enough training samples/features for PCA.")
    n_pc = max(1, min(int(n_components), max_pc))
    pca = PCA(n_components=n_pc)
    pca.fit(X_train)
    scores_all = pca.transform(X_all_scaled)

    return {
        "train_mask": train_mask.to_numpy(),
        "valid_mask": valid_mask.to_numpy(),
        "feature_cols": feature_cols,
        "total_feature_count": total_feature_count,
        "feature_count_after_time_filter": feature_count_after_time_filter,
        "time_related_features": time_related_features,
        "dropped_time_related_features": dropped_time_related_features,
        "drop_time_features": bool(drop_time_features),
        "zero_variance_features": zero_var_features,
        "dropped_zero_variance_features": dropped_zero_var_features,
        "drop_zero_variance": bool(drop_zero_variance),
        "medians": med,
        "scaler": scaler,
        "pca": pca,
        "scores_all": scores_all,
        "n_pc": n_pc,
    }


def clean_standardize_classifier_features(
    feature_table: pd.DataFrame,
    drop_zero_variance: bool = True,
    drop_time_features: bool = False,
):
    """Clean, impute, and standardize features without applying PCA.

    This is used by the neural-network tab when the user chooses to feed the
    NN with standardized original features instead of PCA-reduced features.
    The logic mirrors clean_standardize_pca so training and testing use the
    same feature-selection, imputation, and scaling rules.
    """
    data_type = feature_table["DataType"].astype(str).str.lower()
    train_mask = data_type.eq("train")
    valid_mask = data_type.isin(["valid", "validation", "val"])
    if not train_mask.any():
        raise ValueError("No training samples found in input list.")

    feature_cols = numeric_ml_columns(feature_table)
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    total_feature_count = len(feature_cols)
    time_related_features = [c for c in feature_cols if is_time_related_feature_name(c)]
    dropped_time_related_features: List[str] = []
    if drop_time_features and time_related_features:
        dropped_time_related_features = time_related_features
        drop_set = set(time_related_features)
        feature_cols = [c for c in feature_cols if c not in drop_set]
        if not feature_cols:
            raise ValueError("All numeric features were removed by the time/integration feature dropout option.")

    feature_count_after_time_filter = len(feature_cols)

    X_all = feature_table[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    med = X_all.loc[train_mask].median(axis=0, skipna=True).fillna(0.0)
    X_all = X_all.fillna(med).fillna(0.0)

    X_train_raw = X_all.loc[train_mask].to_numpy(dtype=float)
    raw_std = np.std(X_train_raw, axis=0)
    zero_var_mask = raw_std <= np.finfo(float).eps
    zero_var_features = [c for c, is_zero in zip(feature_cols, zero_var_mask) if bool(is_zero)]
    dropped_zero_var_features: List[str] = []

    if drop_zero_variance and zero_var_features:
        keep_var = ~zero_var_mask
        dropped_zero_var_features = zero_var_features
        feature_cols = [c for c, keep in zip(feature_cols, keep_var) if bool(keep)]
        if not feature_cols:
            raise ValueError("All numeric features are zero-variance in the training data.")
        X_all = X_all.loc[:, feature_cols]
        med = med.loc[feature_cols]
        X_train_raw = X_all.loc[train_mask].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_all_scaled = scaler.transform(X_all.to_numpy(dtype=float))

    return {
        "train_mask": train_mask.to_numpy(),
        "valid_mask": valid_mask.to_numpy(),
        "feature_cols": feature_cols,
        "total_feature_count": total_feature_count,
        "feature_count_after_time_filter": feature_count_after_time_filter,
        "time_related_features": time_related_features,
        "dropped_time_related_features": dropped_time_related_features,
        "drop_time_features": bool(drop_time_features),
        "zero_variance_features": zero_var_features,
        "dropped_zero_variance_features": dropped_zero_var_features,
        "drop_zero_variance": bool(drop_zero_variance),
        "medians": med,
        "scaler": scaler,
        "pca": None,
        "scores_all": X_all_scaled,
        "n_pc": int(X_all_scaled.shape[1]),
        "apply_pca": False,
    }


# -------------------------------------------------------------------------
# PyQt app
# -------------------------------------------------------------------------

class ENoseBatchPCASVMApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("E-Nose Batch PCA + SVM/NN using R0-normalized M features")
        self.resize(1480, 880)
        self.feature_table: Optional[pd.DataFrame] = None
        self.pair_feature_table: Optional[pd.DataFrame] = None
        self.prediction_table: Optional[pd.DataFrame] = None
        self.model_package: Optional[Dict[str, object]] = None
        self.nn_feature_table: Optional[pd.DataFrame] = None
        self.nn_pair_feature_table: Optional[pd.DataFrame] = None
        self.nn_prediction_table: Optional[pd.DataFrame] = None
        self.nn_model_package: Optional[Dict[str, object]] = None

        # Latest PCA score data for Tab 3 plot refresh.
        self.latest_train_scores: Optional[np.ndarray] = None
        self.latest_valid_scores: Optional[np.ndarray] = None
        self.latest_y_train: Optional[np.ndarray] = None
        self.latest_y_valid: Optional[np.ndarray] = None
        self.latest_explained: Optional[np.ndarray] = None

        # Live logger state
        self.worker: Optional[TcpClientWorker] = None
        self.live_log_file = None
        self.live_log_path = ""
        self.live_header: Optional[List[str]] = None
        self.live_gas_names: List[str] = DEFAULT_GAS_COLUMNS.copy()
        self.live_gas_indices: Dict[str, int] = {}
        self.live_sample_index = 0
        self.live_row_count = 0
        self.live_x_data: List[int] = []
        self.live_gas_data: Dict[str, List[float]] = {name: [] for name in self.live_gas_names}
        self.live_gas_plots: Dict[str, GasPlot] = {}
        self.live_current_phase: Optional[str] = None
        self.live_current_phase_start: Optional[int] = None
        self.live_active_regions: Dict[str, pg.LinearRegionItem] = {}
        self.live_max_phase_regions = 200

        # Plotting tab state
        self.last_plotted_csv_path = ""

        # Tab 2 testing/inference state
        self.loaded_model_package: Optional[Dict[str, object]] = None
        self.loaded_model_path: str = ""
        self.testing_feature_table: Optional[pd.DataFrame] = None
        self.testing_prediction_table: Optional[pd.DataFrame] = None

        self._build_ui()


    # ------------------------------------------------------------------
    # Tab 1: live TCP logger and live plots
    # ------------------------------------------------------------------
    def _build_live_tab(self):
        root = QtWidgets.QVBoxLayout(self.live_tab)

        control_box = QtWidgets.QGroupBox("Connection and logging")
        control_layout = QtWidgets.QGridLayout(control_box)
        self.live_host_edit = QtWidgets.QLineEdit(DEFAULT_HOST)
        self.live_port_spin = QtWidgets.QSpinBox()
        self.live_port_spin.setRange(1, 65535)
        self.live_port_spin.setValue(DEFAULT_PORT)
        default_out = datetime.now().strftime("enose_pyqt_log_%Y%m%d_%H%M%S.csv")
        self.live_out_edit = QtWidgets.QLineEdit(os.path.abspath(default_out))
        self.live_browse_btn = QtWidgets.QPushButton("Browse...")
        self.live_browse_btn.clicked.connect(self.browse_live_output_file)
        self.live_connect_btn = QtWidgets.QPushButton("Connect and log")
        self.live_connect_btn.clicked.connect(self.connect_to_esp32)
        self.live_disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self.live_disconnect_btn.clicked.connect(self.disconnect_from_esp32)
        self.live_disconnect_btn.setEnabled(False)
        self.live_clear_btn = QtWidgets.QPushButton("Clear plots/data")
        self.live_clear_btn.clicked.connect(self.clear_live_data)
        self.live_max_points_spin = QtWidgets.QSpinBox()
        self.live_max_points_spin.setRange(100, 200000)
        self.live_max_points_spin.setValue(5000)
        self.live_max_points_spin.setSingleStep(500)
        self.live_autoscroll_check = QtWidgets.QCheckBox("Auto-scroll X axis")
        self.live_autoscroll_check.setChecked(True)

        control_layout.addWidget(QtWidgets.QLabel("Host"), 0, 0)
        control_layout.addWidget(self.live_host_edit, 0, 1)
        control_layout.addWidget(QtWidgets.QLabel("Port"), 0, 2)
        control_layout.addWidget(self.live_port_spin, 0, 3)
        control_layout.addWidget(self.live_connect_btn, 0, 4)
        control_layout.addWidget(self.live_disconnect_btn, 0, 5)
        control_layout.addWidget(QtWidgets.QLabel("Output CSV"), 1, 0)
        control_layout.addWidget(self.live_out_edit, 1, 1, 1, 3)
        control_layout.addWidget(self.live_browse_btn, 1, 4)
        control_layout.addWidget(self.live_clear_btn, 1, 5)
        control_layout.addWidget(QtWidgets.QLabel("Max plotted samples"), 2, 0)
        control_layout.addWidget(self.live_max_points_spin, 2, 1)
        control_layout.addWidget(self.live_autoscroll_check, 2, 2)
        root.addWidget(control_box)

        status_box = QtWidgets.QGroupBox("Status")
        status_layout = QtWidgets.QHBoxLayout(status_box)
        self.live_status_label = QtWidgets.QLabel("Disconnected")
        self.live_rows_label = QtWidgets.QLabel("Rows: 0")
        self.live_phase_label = QtWidgets.QLabel("Phase: -")
        self.live_file_label = QtWidgets.QLabel("File: -")
        status_layout.addWidget(self.live_status_label, 2)
        status_layout.addWidget(self.live_rows_label, 1)
        status_layout.addWidget(self.live_phase_label, 1)
        status_layout.addWidget(self.live_file_label, 3)
        root.addWidget(status_box)

        legend = QtWidgets.QLabel("Y-axis: sensor resistance (Ω). Fixed conversion: Rref=1 kΩ, Vs=5 V, ADC full-scale=5 V, ADC bits=24, ADC mode=twoscomp. Background: A=blue, M=orange.")
        legend.setStyleSheet("font-weight: bold;")
        root.addWidget(legend)

        self.live_plot_area = QtWidgets.QScrollArea()
        self.live_plot_area.setWidgetResizable(True)
        self.live_plot_container = QtWidgets.QWidget()
        self.live_plot_grid = QtWidgets.QGridLayout(self.live_plot_container)
        self.live_plot_grid.setContentsMargins(4, 4, 4, 4)
        self.live_plot_grid.setSpacing(8)
        self.live_plot_area.setWidget(self.live_plot_container)
        root.addWidget(self.live_plot_area, 1)
        self._build_logger_plots(self.live_gas_names)

        self.live_plot_timer = QtCore.QTimer(self)
        self.live_plot_timer.setInterval(100)
        self.live_plot_timer.timeout.connect(self.update_live_plots)
        self.live_plot_timer.start()

    def _build_logger_plots(self, gas_names: List[str]):
        while self.live_plot_grid.count():
            item = self.live_plot_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.live_gas_plots.clear()
        for i, name in enumerate(gas_names):
            gas_plot = GasPlot(name, y_label="Resistance (Ω)")
            self.live_gas_plots[name] = gas_plot
            self.live_plot_grid.addWidget(gas_plot.widget, i // 2, i % 2)

    def browse_live_output_file(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select output CSV file", self.live_out_edit.text(),
            "CSV files (*.csv);;All files (*.*)")
        if path:
            self.live_out_edit.setText(path)

    def connect_to_esp32(self):
        if self.worker is not None:
            return
        self.clear_live_data()
        host = self.live_host_edit.text().strip()
        port = int(self.live_port_spin.value())
        out_path = self.live_out_edit.text().strip()
        if not out_path:
            out_path = os.path.abspath(datetime.now().strftime("enose_pyqt_log_%Y%m%d_%H%M%S.csv"))
            self.live_out_edit.setText(out_path)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            self.live_log_file = open(out_path, "w", newline="", encoding="utf-8")
            self.live_log_path = out_path
            self.live_file_label.setText(f"File: {out_path}")
            self.last_plotted_csv_path = out_path
            if hasattr(self, "plot_file_edit"):
                self.plot_file_edit.setText(out_path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "File error", f"Cannot open output file:\n{exc}")
            self.live_log_file = None
            return

        self.worker = TcpClientWorker(host, port, self)
        self.worker.line_received.connect(self.handle_live_line)
        self.worker.status_changed.connect(self.set_live_status)
        self.worker.connected_changed.connect(self.handle_live_connected_changed)
        self.worker.stream_ended.connect(self.handle_live_stream_ended)
        self.worker.error_occurred.connect(self.handle_live_error)
        self.worker.finished.connect(self.handle_live_worker_finished)
        self.worker.start()
        self.live_connect_btn.setEnabled(False)
        self.live_disconnect_btn.setEnabled(True)
        self.live_host_edit.setEnabled(False)
        self.live_port_spin.setEnabled(False)
        self.live_out_edit.setEnabled(False)
        self.live_browse_btn.setEnabled(False)

    def disconnect_from_esp32(self):
        if self.worker is not None:
            self.set_live_status("Disconnecting...")
            self.worker.stop()
            self.worker.wait(2000)
        self.close_live_log_file()
        self.reset_live_connection_buttons()

    def handle_live_connected_changed(self, connected: bool):
        if connected:
            self.set_live_status("Connected. Waiting for CSV data...")
        else:
            if self.worker is not None:
                self.set_live_status("Disconnected")

    def handle_live_stream_ended(self, reason: str):
        self.set_live_status(reason)
        self.close_live_log_file()

    def handle_live_error(self, message: str):
        self.set_live_status(f"Error: {message}")
        QtWidgets.QMessageBox.warning(self, "TCP logger error", message)
        self.close_live_log_file()

    def handle_live_worker_finished(self):
        self.worker = None
        self.close_live_log_file()
        self.reset_live_connection_buttons()

    def reset_live_connection_buttons(self):
        self.live_connect_btn.setEnabled(True)
        self.live_disconnect_btn.setEnabled(False)
        self.live_host_edit.setEnabled(True)
        self.live_port_spin.setEnabled(True)
        self.live_out_edit.setEnabled(True)
        self.live_browse_btn.setEnabled(True)

    def close_live_log_file(self):
        if self.live_log_file is not None:
            try:
                self.live_log_file.flush()
                self.live_log_file.close()
            except OSError:
                pass
            self.live_log_file = None

    def set_live_status(self, message: str):
        self.live_status_label.setText(str(message))

    def handle_live_line(self, line: str):
        clean_line = line.strip()
        if not clean_line or clean_line == "#END":
            return
        if self.live_log_file is not None:
            self.live_log_file.write(clean_line + "\n")
            if self.live_row_count % 10 == 0:
                self.live_log_file.flush()
        try:
            row = next(csv.reader([clean_line], skipinitialspace=True))
        except Exception:
            return
        if not row:
            return
        first = row[0].strip()
        if first.lower() == "state":
            self.live_header = [item.strip() for item in row]
            self.configure_live_gas_columns_from_header()
            return
        if self.live_header is None:
            self.live_header = ["State"] + DEFAULT_GAS_COLUMNS + ["Temp", "Hud", "AirFlow"]
            self.configure_live_gas_columns_from_header()
        if len(row) < 2:
            return
        phase = first.upper()
        self.live_phase_label.setText(f"Phase: {phase}")
        values: Dict[str, float] = {}
        for gas_name, idx in self.live_gas_indices.items():
            if idx < len(row):
                values[gas_name] = self._to_float_or_nan(row[idx].strip())
        if values:
            self.add_live_sample(phase, values)
        self.live_row_count += 1
        self.live_rows_label.setText(f"Rows: {self.live_row_count}")

    @staticmethod
    def _to_float_or_nan(value: object) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    def configure_live_gas_columns_from_header(self):
        if self.live_header is None:
            return
        gas_indices = {}
        for idx, col_name in enumerate(self.live_header):
            normalized = col_name.strip().lower().replace(" ", "_")
            if normalized not in NON_GAS_COLUMNS and idx != 0:
                gas_indices[col_name.strip()] = idx
        ordered = [name for name in DEFAULT_GAS_COLUMNS if name in gas_indices]
        ordered += [name for name in gas_indices.keys() if name not in ordered]
        if ordered and ordered != self.live_gas_names:
            self.live_gas_names = ordered
            self.live_gas_data = {name: [] for name in self.live_gas_names}
            self._build_logger_plots(self.live_gas_names)
            self.live_current_phase = None
            self.live_current_phase_start = None
            self.live_active_regions.clear()
        self.live_gas_indices = {name: gas_indices[name] for name in ordered}
        self.set_live_status("Header received. Gas columns: " + ", ".join(self.live_gas_names))

    def _convert_live_adc_values_to_resistance(self, values: Dict[str, float]) -> Dict[str, float]:
        """Convert one live ADC row to resistance using fixed logger settings.

        Fixed live-plot conversion requested for the logger tab:
            Rref = 1000 ohm, Vs = 5 V, ADC full-scale = 5 V,
            ADC bits = 24, ADC mode = twoscomp.
        The raw ADC values are still written to the log CSV; only the live plot
        y-data are converted to resistance.
        """
        if not values:
            return {}
        gas_names = list(values.keys())
        adc = np.asarray([[values.get(g, np.nan) for g in gas_names]], dtype=float)
        settings = ProcessingSettings(
            rref=1000.0,
            vs=5.0,
            vref_fs=5.0,
            adc_bits=24,
            adc_mode="twoscomp",
            resistance_unit="ohm",
        )
        v_adc, invalid_adc = adc_to_voltage(adc, settings)
        r = settings.rref * (settings.vs / v_adc - 1.0)
        invalid = invalid_adc | (v_adc <= 0) | (v_adc > settings.vs) | (~np.isfinite(r)) | (r < 0)
        r[invalid] = np.nan
        return {g: float(r[0, i]) for i, g in enumerate(gas_names)}

    def add_live_sample(self, phase: str, values: Dict[str, float]):
        x = self.live_sample_index
        self.live_sample_index += 1
        self.live_x_data.append(x)
        resistance_values = self._convert_live_adc_values_to_resistance(values)
        for name in self.live_gas_names:
            self.live_gas_data.setdefault(name, []).append(resistance_values.get(name, float("nan")))
        self.update_live_phase_regions(phase, x)
        self.enforce_live_max_points()

    def update_live_phase_regions(self, phase: str, x: int):
        if phase not in ("A", "M"):
            phase = "?"
        if self.live_current_phase != phase:
            self.live_current_phase = phase
            self.live_current_phase_start = x
            self.live_active_regions.clear()
            start = x - 0.5
            end = x + 0.5
            for name, gas_plot in self.live_gas_plots.items():
                region = gas_plot.add_phase_region(start, end, phase)
                self.live_active_regions[name] = region
                if len(gas_plot.phase_regions) > self.live_max_phase_regions:
                    old = gas_plot.phase_regions.pop(0)
                    try:
                        gas_plot.widget.removeItem(old)
                    except Exception:
                        pass
        start = (self.live_current_phase_start if self.live_current_phase_start is not None else x) - 0.5
        end = x + 0.5
        for region in self.live_active_regions.values():
            try:
                region.setRegion((start, end))
            except Exception:
                pass

    def enforce_live_max_points(self):
        max_points = int(self.live_max_points_spin.value())
        excess = len(self.live_x_data) - max_points
        if excess <= 0:
            return
        del self.live_x_data[:excess]
        for name in list(self.live_gas_data.keys()):
            del self.live_gas_data[name][:excess]

    def update_live_plots(self):
        if not self.live_x_data:
            return
        for name, gas_plot in self.live_gas_plots.items():
            y = self.live_gas_data.get(name, [])
            gas_plot.curve.setData(self.live_x_data, y)
            if self.live_autoscroll_check.isChecked():
                max_points = int(self.live_max_points_spin.value())
                x_max = self.live_x_data[-1]
                x_min = max(0, x_max - max_points)
                gas_plot.widget.setXRange(x_min, x_max + 1, padding=0.01)

    def clear_live_data(self):
        self.live_sample_index = 0
        self.live_row_count = 0
        self.live_header = None
        self.live_gas_names = DEFAULT_GAS_COLUMNS.copy()
        self.live_gas_indices = {}
        self.live_x_data.clear()
        self.live_gas_data = {name: [] for name in self.live_gas_names}
        self.live_current_phase = None
        self.live_current_phase_start = None
        self.live_active_regions.clear()
        self.live_rows_label.setText("Rows: 0")
        self.live_phase_label.setText("Phase: -")
        self._build_logger_plots(self.live_gas_names)

    # ------------------------------------------------------------------
    # Tab 2: signal plotting similar to plotAMPhasePairs.m
    # ------------------------------------------------------------------
    def _build_signal_plot_tab(self):
        root = QtWidgets.QHBoxLayout(self.plot_tab)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_widget = QtWidgets.QWidget()
        left_scroll.setWidget(left_widget)
        left_scroll.setMinimumWidth(300)
        left = QtWidgets.QVBoxLayout(left_widget)

        file_box = QtWidgets.QGroupBox("CSV file")
        file_grid = QtWidgets.QGridLayout(file_box)
        self.plot_file_edit = QtWidgets.QLineEdit()
        self.plot_browse_btn = QtWidgets.QPushButton("Browse...")
        self.plot_browse_btn.clicked.connect(self.browse_signal_plot_file)
        self.plot_use_logged_btn = QtWidgets.QPushButton("Use last logged file")
        self.plot_use_logged_btn.clicked.connect(self.use_last_logged_file_for_plot)
        file_grid.addWidget(QtWidgets.QLabel("CSV file"), 0, 0)
        file_grid.addWidget(self.plot_file_edit, 0, 1)
        file_grid.addWidget(self.plot_browse_btn, 0, 2)
        file_grid.addWidget(self.plot_use_logged_btn, 1, 1, 1, 2)
        left.addWidget(file_box)

        plot_box = QtWidgets.QGroupBox("Plot options")
        plot_grid = QtWidgets.QGridLayout(plot_box)
        self.plot_pair_mode_combo = QtWidgets.QComboBox()
        self.plot_pair_mode_combo.addItems(["all", "first", "last", "specified"])
        self.plot_pair_mode_combo.setCurrentText("all")
        self.plot_pair_index_spin = QtWidgets.QSpinBox()
        self.plot_pair_index_spin.setRange(1, 9999)
        self.plot_pair_index_spin.setValue(1)
        self.plot_mode_combo = QtWidgets.QComboBox()
        self.plot_mode_combo.addItems(["adc", "resistance"])
        self.plot_mode_combo.setCurrentText("resistance")
        self.plot_now_btn = QtWidgets.QPushButton("Plot signal")
        self.plot_now_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.plot_now_btn.clicked.connect(self.plot_signal_file)
        for r, (label, widget) in enumerate([
            ("A-M pair", self.plot_pair_mode_combo),
            ("Specified pair", self.plot_pair_index_spin),
            ("Y-axis mode", self.plot_mode_combo),
        ]):
            plot_grid.addWidget(QtWidgets.QLabel(label), r, 0)
            plot_grid.addWidget(widget, r, 1)
        plot_grid.addWidget(self.plot_now_btn, 3, 0, 1, 2)
        left.addWidget(plot_box)

        conv_box = QtWidgets.QGroupBox("ADC/resistance settings")
        conv_grid = QtWidgets.QGridLayout(conv_box)
        self.plot_ts_spin = self.double_spin(0.5, 0.001, 3600, 3)
        self.plot_rref_spin = self.double_spin(1000, 0.000001, 1e9, 3)
        self.plot_vs_spin = self.double_spin(5, 0.000001, 100, 4)
        self.plot_vref_spin = self.double_spin(5, 0.000001, 100, 4)
        self.plot_bits_spin = QtWidgets.QSpinBox()
        self.plot_bits_spin.setRange(1, 32)
        self.plot_bits_spin.setValue(24)
        self.plot_adc_mode_combo = QtWidgets.QComboBox()
        self.plot_adc_mode_combo.addItems(["twoscomp", "signed", "unsigned"])
        self.plot_adc_mode_combo.setCurrentText("twoscomp")
        for r, (label, widget) in enumerate([
            ("Ts (s)", self.plot_ts_spin),
            ("Rref (ohm)", self.plot_rref_spin),
            ("Vs (V)", self.plot_vs_spin),
            ("ADC full-scale (V)", self.plot_vref_spin),
            ("ADC bits", self.plot_bits_spin),
            ("ADC mode", self.plot_adc_mode_combo),
        ]):
            conv_grid.addWidget(QtWidgets.QLabel(label), r, 0)
            conv_grid.addWidget(widget, r, 1)
        left.addWidget(conv_box)

        self.plot_status_text = QtWidgets.QTextEdit()
        self.plot_status_text.setReadOnly(True)
        self.plot_status_text.setMinimumHeight(130)
        self.plot_status_text.setText("Load a CSV file and click Plot signal.")
        left.addWidget(self.plot_status_text)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        self.signal_figure = Figure(figsize=(10, 7))
        self.signal_canvas = FigureCanvas(self.signal_figure)
        right.addWidget(self.signal_canvas, 1)
        self.signal_table = QtWidgets.QTableWidget()
        self.signal_table.setMaximumHeight(190)
        right.addWidget(self._table_box("Detected A-M pairs", self.signal_table))

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setSizes([370, 1050])
        root.addWidget(splitter, 1)

    def browse_signal_plot_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select e-nose CSV file", os.getcwd(),
            "CSV files (*.csv);;All files (*.*)")
        if path:
            self.plot_file_edit.setText(path)
            self.last_plotted_csv_path = path

    def use_last_logged_file_for_plot(self):
        path = self.live_log_path or self.last_plotted_csv_path
        if path:
            self.plot_file_edit.setText(path)
        else:
            QtWidgets.QMessageBox.information(self, "No logged file", "No logged CSV file is available yet.")

    def current_plot_settings(self) -> ProcessingSettings:
        return ProcessingSettings(
            ts=float(self.plot_ts_spin.value()),
            rref=float(self.plot_rref_spin.value()),
            vs=float(self.plot_vs_spin.value()),
            vref_fs=float(self.plot_vref_spin.value()),
            adc_bits=int(self.plot_bits_spin.value()),
            adc_mode=str(self.plot_adc_mode_combo.currentText()),
            early_sec=5.0,
            late_sec=5.0,
        )

    def plot_signal_file(self):
        try:
            csv_path = self.plot_file_edit.text().strip()
            if not csv_path or not os.path.isfile(csv_path):
                QtWidgets.QMessageBox.warning(self, "CSV file", "Please select a valid e-nose CSV file.")
                return
            settings = self.current_plot_settings()
            df = pd.read_csv(csv_path, skipinitialspace=True)
            df = normalize_input_dataframe(df)
            gas_cols = detect_gas_columns(df)
            if not gas_cols:
                raise ValueError("No gas sensor columns were detected in the CSV file.")
            pairs = am_pair_segments(df["State"].to_numpy())
            if not pairs:
                raise ValueError("No consecutive A-M phase pair was found.")

            pair_mode = str(self.plot_pair_mode_combo.currentText()).lower()
            if pair_mode == "all":
                selected_indices = list(range(len(pairs)))
            elif pair_mode == "first":
                selected_indices = [0]
            elif pair_mode == "last":
                selected_indices = [len(pairs) - 1]
            else:
                idx = int(self.plot_pair_index_spin.value()) - 1
                if idx < 0 or idx >= len(pairs):
                    raise ValueError(f"Specified pair is outside available range 1 to {len(pairs)}.")
                selected_indices = [idx]

            pair_meta = []
            for i, (a_seg, m_seg) in enumerate(pairs, start=1):
                pair_meta.append({
                    "PairNo": i,
                    "A_Start": a_seg.start + 1,
                    "A_End": a_seg.end + 1,
                    "M_Start": m_seg.start + 1,
                    "M_End": m_seg.end + 1,
                    "A_Samples": a_seg.length,
                    "M_Samples": m_seg.length,
                })
            self.show_dataframe(pd.DataFrame(pair_meta), self.signal_table, max_rows=200, max_cols=20)

            plot_mode = str(self.plot_mode_combo.currentText()).lower()
            y_data_by_sensor, phase_info, time_vector = self._prepare_signal_plot_data(
                df, gas_cols, pairs, selected_indices, settings, plot_mode)
            self._draw_signal_plot(gas_cols, y_data_by_sensor, phase_info, time_vector, plot_mode)
            self.plot_status_text.setText(
                f"Plotted {len(selected_indices)} A-M pair(s) from {os.path.basename(csv_path)}.\n"
                f"Detected pairs: {len(pairs)}. Mode: {plot_mode}. ADC mode: {settings.adc_mode}."
            )
        except Exception as exc:
            self.plot_status_text.setText(traceback.format_exc())
            QtWidgets.QMessageBox.critical(self, "Signal plot error", str(exc))

    def _prepare_signal_plot_data(self, df: pd.DataFrame, gas_cols: List[str], pairs, selected_indices, settings: ProcessingSettings, plot_mode: str):
        phase_info = []
        sample_offset = 0
        y_parts = {gas: [] for gas in gas_cols}
        for idx in selected_indices:
            a_seg, m_seg = pairs[idx]
            pair_slice = np.r_[np.arange(a_seg.start, a_seg.end + 1), np.arange(m_seg.start, m_seg.end + 1)]
            na = a_seg.length
            nm = m_seg.length
            if plot_mode == "resistance":
                sub_df = df.iloc[pair_slice].reset_index(drop=True)
                rmat, _ = adc_to_resistance_matrix(sub_df, gas_cols, settings)
                for s, gas in enumerate(gas_cols):
                    y_parts[gas].extend(rmat[:, s].tolist())
            else:
                for gas in gas_cols:
                    y_parts[gas].extend(df.iloc[pair_slice][gas].to_numpy(dtype=float).tolist())
            phase_info.append({"pairNo": idx + 1, "phase": "A", "tStart": sample_offset * settings.ts, "tEnd": (sample_offset + na - 1) * settings.ts})
            phase_info.append({"pairNo": idx + 1, "phase": "M", "tStart": (sample_offset + na) * settings.ts, "tEnd": (sample_offset + na + nm - 1) * settings.ts})
            sample_offset += na + nm
        n = sample_offset
        t = np.arange(n, dtype=float) * settings.ts
        y_data = {gas: np.asarray(vals, dtype=float) for gas, vals in y_parts.items()}
        return y_data, phase_info, t

    def _draw_signal_plot(self, gas_cols: List[str], y_data_by_sensor: Dict[str, np.ndarray], phase_info: List[Dict[str, object]], t: np.ndarray, plot_mode: str):
        self.signal_figure.clear()
        axes = self.signal_figure.subplots(4, 2)
        axes = np.asarray(axes).reshape(-1)
        odor_titles = {
            "OV": "Organic vapor (OV)",
            "Alc": "Alcohol (Alc)",
            "H2": "Hydrogen (H$_2$)",
            "AC1": "AC1",
            "NH3": "Ammonia (NH$_3$)",
            "AC2": "AC2",
            "VOC": "Volatile organic compounds (VOC)",
            "LP": "Liquefied petroleum gas (LP)",
        }
        a_bg = (0.70, 0.90, 1.00)
        m_bg = (1.00, 0.85, 0.65)
        a_text = (0.05, 0.38, 0.60)
        m_text = (0.75, 0.35, 0.05)
        ylabel = "Resistance (Ω)" if plot_mode == "resistance" else "ADC value"
        title_mode = "Resistance" if plot_mode == "resistance" else "ADC Value"

        for i, gas in enumerate(gas_cols[:8]):
            ax = axes[i]
            y = y_data_by_sensor[gas]
            finite = y[np.isfinite(y)]
            if finite.size == 0:
                y_min, y_max = 0.0, 1.0
            else:
                y_min, y_max = float(np.min(finite)), float(np.max(finite))
                margin = max(abs(y_min) * 0.05, 1.0) if y_min == y_max else 0.05 * (y_max - y_min)
                y_min -= margin
                y_max += margin

            for info in phase_info:
                bg = a_bg if info["phase"] == "A" else m_bg
                label_color = a_text if info["phase"] == "A" else m_text
                ax.axvspan(float(info["tStart"]), float(info["tEnd"]), color=bg, alpha=0.25, linewidth=0)
                ax.axvline(float(info["tStart"]), color="0.4", linestyle=":", linewidth=0.8)
                x_text = 0.5 * (float(info["tStart"]) + float(info["tEnd"]))
                y_text = y_min + 0.06 * (y_max - y_min)
                ax.text(x_text, y_text, str(info["phase"]), ha="center", va="bottom", weight="bold", fontsize=9, color=label_color)
            if phase_info:
                ax.axvline(float(phase_info[-1]["tEnd"]), color="0.4", linestyle=":", linewidth=0.8)
            ax.plot(t, y, color="black", linewidth=1.1)
            ax.set_title(odor_titles.get(gas, gas), fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(y_min, y_max)
            if t.size:
                ax.set_xlim(float(t[0]), float(t[-1]) if t.size > 1 else float(t[0] + 1))
        for j in range(len(gas_cols[:8]), len(axes)):
            axes[j].set_axis_off()
        self.signal_figure.suptitle(f"E-nose Response ({title_mode})")
        self.signal_figure.tight_layout(rect=[0, 0.02, 1, 0.96])
        self.signal_canvas.draw()

    def _build_ui(self):
        self.tabs = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabs)

        self.live_tab = QtWidgets.QWidget()
        self.plot_tab = QtWidgets.QWidget()
        self.train_tab = QtWidgets.QWidget()
        self.nn_tab = QtWidgets.QWidget()
        self.test_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.live_tab, "1. Live logger")
        self.tabs.addTab(self.plot_tab, "2. Signal plot")
        self.tabs.addTab(self.train_tab, "3. Training / model export")
        self.tabs.addTab(self.nn_tab, "4. NN training / model export")
        self.tabs.addTab(self.test_tab, "5. Testing / model inference")

        self._build_live_tab()
        self._build_signal_plot_tab()

        root = QtWidgets.QHBoxLayout(self.train_tab)

        left = QtWidgets.QScrollArea()
        left.setWidgetResizable(True)
        left_widget = QtWidgets.QWidget()
        left.setWidget(left_widget)
        left.setMinimumWidth(320)
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        # Input file group
        input_box = QtWidgets.QGroupBox("Input Excel/CSV list")
        input_grid = QtWidgets.QGridLayout(input_box)
        self.input_path_edit = QtWidgets.QLineEdit()
        self.browse_btn = QtWidgets.QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_input_file)
        input_grid.addWidget(QtWidgets.QLabel("List file"), 0, 0)
        input_grid.addWidget(self.input_path_edit, 0, 1)
        input_grid.addWidget(self.browse_btn, 0, 2)
        input_note = QtWidgets.QLabel("Columns: train/valid | CSV file path | integer label")
        input_note.setWordWrap(True)
        input_grid.addWidget(input_note, 1, 0, 1, 3)
        left_layout.addWidget(input_box)

        # Signal processing group
        proc_box = QtWidgets.QGroupBox("Signal processing")
        proc = QtWidgets.QGridLayout(proc_box)
        row = 0
        self.ts_spin = self.double_spin(0.5, 0.001, 3600, 3)
        self.rref_spin = self.double_spin(1000, 0.000001, 1e9, 3)
        self.vs_spin = self.double_spin(5, 0.000001, 100, 4)
        self.vref_spin = self.double_spin(5, 0.000001, 100, 4)
        self.bits_spin = QtWidgets.QSpinBox()
        self.bits_spin.setRange(1, 32)
        self.bits_spin.setValue(24)
        self.adc_mode_combo = QtWidgets.QComboBox()
        self.adc_mode_combo.addItems(["twoscomp", "signed", "unsigned"])
        self.adc_mode_combo.setCurrentText("twoscomp")
        self.r0_method_combo = QtWidgets.QComboBox()
        self.r0_method_combo.addItems(["median", "mean", "trimmedmean", "linear", "robustlinear"])
        self.r0_method_combo.setCurrentText("median")
        self.r0_win_spin = self.double_spin(5, 0.001, 3600, 2)
        self.early_spin = self.double_spin(5, 0.001, 3600, 2)
        self.late_spin = self.double_spin(5, 0.001, 3600, 2)
        self.trim_spin = self.double_spin(20, 0, 99.9, 1)
        self.min_samples_spin = QtWidgets.QSpinBox()
        self.min_samples_spin.setRange(1, 100000)
        self.min_samples_spin.setValue(5)

        for label, widget in [
            ("Ts (s)", self.ts_spin),
            ("Rref (ohm)", self.rref_spin),
            ("Vs (V)", self.vs_spin),
            ("ADC full-scale (V)", self.vref_spin),
            ("ADC bits", self.bits_spin),
            ("ADC mode", self.adc_mode_combo),
            ("R0 method", self.r0_method_combo),
            ("R0 window (s)", self.r0_win_spin),
            ("Early window (s)", self.early_spin),
            ("Late window (s)", self.late_spin),
            ("Trim percent", self.trim_spin),
            ("Min samples", self.min_samples_spin),
        ]:
            proc.addWidget(QtWidgets.QLabel(label), row, 0)
            proc.addWidget(widget, row, 1)
            row += 1
        left_layout.addWidget(proc_box)

        # Pair representation group
        pair_box = QtWidgets.QGroupBox("A-M pair representation per file")
        pair_grid = QtWidgets.QGridLayout(pair_box)
        self.pair_mode_combo = QtWidgets.QComboBox()
        self.pair_mode_combo.addItems(["first", "last", "specified", "average_all"])
        self.pair_mode_combo.setCurrentText("first")
        self.pair_index_spin = QtWidgets.QSpinBox()
        self.pair_index_spin.setRange(1, 9999)
        self.pair_index_spin.setValue(1)
        pair_grid.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        pair_grid.addWidget(self.pair_mode_combo, 0, 1)
        pair_grid.addWidget(QtWidgets.QLabel("Specified pair"), 1, 0)
        pair_grid.addWidget(self.pair_index_spin, 1, 1)
        note = QtWidgets.QLabel("When one CSV has multiple A-M cycles, this option converts pair-level features into one file-level feature row before PCA/SVM.")
        note.setWordWrap(True)
        pair_grid.addWidget(note, 2, 0, 1, 2)
        left_layout.addWidget(pair_box)

        # PCA and SVM group
        model_box = QtWidgets.QGroupBox("PCA and SVM")
        model = QtWidgets.QGridLayout(model_box)
        self.pca_dim_spin = QtWidgets.QSpinBox()
        self.pca_dim_spin.setRange(1, 100)
        self.pca_dim_spin.setValue(2)
        self.svm_kernel_combo = QtWidgets.QComboBox()
        self.svm_kernel_combo.addItems(["rbf", "linear", "poly", "sigmoid"])
        self.svm_kernel_combo.setCurrentText("rbf")
        self.c_spin = self.double_spin(1.0, 0.000001, 1e9, 6)
        self.gamma_combo = QtWidgets.QComboBox()
        self.gamma_combo.addItems(["scale", "auto", "custom"])
        self.gamma_value_spin = self.double_spin(0.1, 0.000001, 1e9, 6)
        self.degree_spin = QtWidgets.QSpinBox()
        self.degree_spin.setRange(1, 10)
        self.degree_spin.setValue(3)
        self.class_weight_check = QtWidgets.QCheckBox("class_weight='balanced'")
        self.drop_zero_var_check = QtWidgets.QCheckBox("Drop zero-variance features before PCA")
        self.drop_zero_var_check.setChecked(True)
        self.drop_time_feature_check = QtWidgets.QCheckBox("Drop time/slope/integration features before PCA")
        self.drop_time_feature_check.setChecked(False)
        self.drop_time_feature_check.setToolTip(
            "Drops timeToMax, timeToMin, timeToAbsPeak, areaTotal, areaAbs, "
            "initialSlope, lateSlope, R0Slope, and R0SlopeRelative features."
        )

        self.pca_plot_selector_label = QtWidgets.QLabel("PCA plot PCs")
        self.pca_plot_selector_widget = QtWidgets.QWidget()
        selector_layout = QtWidgets.QHBoxLayout(self.pca_plot_selector_widget)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(4)
        self.pca_plot_pc1_combo = QtWidgets.QComboBox()
        self.pca_plot_pc2_combo = QtWidgets.QComboBox()
        self.pca_plot_pc3_combo = QtWidgets.QComboBox()
        for combo in self.pca_plot_combos():
            combo.addItem("N/A")
            combo.currentIndexChanged.connect(self.refresh_training_pca_plot)
            selector_layout.addWidget(combo)
        self.pca_dim_spin.valueChanged.connect(self.update_pca_plot_selector_visibility)
        r = 0
        for label, widget in [
            ("PCA dimensions", self.pca_dim_spin),
            ("SVM kernel", self.svm_kernel_combo),
            ("C", self.c_spin),
            ("Gamma", self.gamma_combo),
            ("Gamma custom", self.gamma_value_spin),
            ("Poly degree", self.degree_spin),
        ]:
            model.addWidget(QtWidgets.QLabel(label), r, 0)
            model.addWidget(widget, r, 1)
            r += 1
        model.addWidget(self.class_weight_check, r, 0, 1, 2)
        r += 1
        model.addWidget(self.drop_zero_var_check, r, 0, 1, 2)
        r += 1
        model.addWidget(self.drop_time_feature_check, r, 0, 1, 2)
        r += 1
        model.addWidget(self.pca_plot_selector_label, r, 0)
        model.addWidget(self.pca_plot_selector_widget, r, 1)
        self.set_pca_plot_selector_visible(False)
        left_layout.addWidget(model_box)

        self.run_btn = QtWidgets.QPushButton("Run feature extraction + PCA + SVM")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.run_btn.clicked.connect(self.run_pipeline)
        left_layout.addWidget(self.run_btn)

        export_row = QtWidgets.QHBoxLayout()
        self.export_features_btn = QtWidgets.QPushButton("Export features")
        self.export_features_btn.clicked.connect(self.export_features)
        self.export_features_btn.setEnabled(False)
        self.export_predictions_btn = QtWidgets.QPushButton("Export predictions")
        self.export_predictions_btn.clicked.connect(self.export_predictions)
        self.export_predictions_btn.setEnabled(False)
        self.save_model_btn = QtWidgets.QPushButton("Export model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        export_row.addWidget(self.export_features_btn)
        export_row.addWidget(self.export_predictions_btn)
        export_row.addWidget(self.save_model_btn)
        left_layout.addLayout(export_row)

        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(170)
        self.status_text.setText("Ready.")
        left_layout.addWidget(self.status_text)
        left_layout.addStretch(1)

        # Right side: plots and tables
        right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top)
        self.figure = Figure(figsize=(8, 4.7))
        self.canvas = FigureCanvas(self.figure)
        top_layout.addWidget(self.canvas)
        self.result_text = QtWidgets.QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        top_layout.addWidget(self.result_text)
        right_split.addWidget(top)

        tables = QtWidgets.QTabWidget()
        self.feature_table_widget = QtWidgets.QTableWidget()
        self.prediction_table_widget = QtWidgets.QTableWidget()
        self.pair_table_widget = QtWidgets.QTableWidget()
        tables.addTab(self.feature_table_widget, "File-level features")
        tables.addTab(self.prediction_table_widget, "Predictions")
        tables.addTab(self.pair_table_widget, "Pair-level features")
        right_split.addWidget(tables)
        right_split.setSizes([520, 300])

        train_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        train_splitter.addWidget(left)
        train_splitter.addWidget(right_split)
        train_splitter.setSizes([390, 1050])
        root.addWidget(train_splitter, 1)

        self._build_nn_tab()
        self._build_testing_tab()

    def _build_nn_tab(self):
        root = QtWidgets.QHBoxLayout(self.nn_tab)

        left = QtWidgets.QScrollArea()
        left.setWidgetResizable(True)
        left_widget = QtWidgets.QWidget()
        left.setWidget(left_widget)
        left.setMinimumWidth(360)
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        input_box = QtWidgets.QGroupBox("Input Excel/CSV list")
        input_grid = QtWidgets.QGridLayout(input_box)
        self.nn_input_path_edit = QtWidgets.QLineEdit()
        self.nn_browse_btn = QtWidgets.QPushButton("Browse...")
        self.nn_browse_btn.clicked.connect(self.nn_browse_input_file)
        input_grid.addWidget(QtWidgets.QLabel("List file"), 0, 0)
        input_grid.addWidget(self.nn_input_path_edit, 0, 1)
        input_grid.addWidget(self.nn_browse_btn, 0, 2)
        input_note = QtWidgets.QLabel("Columns: train/valid | CSV file path | integer label")
        input_note.setWordWrap(True)
        input_grid.addWidget(input_note, 1, 0, 1, 3)
        left_layout.addWidget(input_box)

        proc_box = QtWidgets.QGroupBox("Signal processing")
        proc = QtWidgets.QGridLayout(proc_box)
        row = 0
        self.nn_ts_spin = self.double_spin(0.5, 0.001, 3600, 3)
        self.nn_rref_spin = self.double_spin(1000, 0.000001, 1e9, 3)
        self.nn_vs_spin = self.double_spin(5, 0.000001, 100, 4)
        self.nn_vref_spin = self.double_spin(5, 0.000001, 100, 4)
        self.nn_bits_spin = QtWidgets.QSpinBox()
        self.nn_bits_spin.setRange(1, 32)
        self.nn_bits_spin.setValue(24)
        self.nn_adc_mode_combo = QtWidgets.QComboBox()
        self.nn_adc_mode_combo.addItems(["twoscomp", "signed", "unsigned"])
        self.nn_adc_mode_combo.setCurrentText("twoscomp")
        self.nn_r0_method_combo = QtWidgets.QComboBox()
        self.nn_r0_method_combo.addItems(["median", "mean", "trimmedmean", "linear", "robustlinear"])
        self.nn_r0_method_combo.setCurrentText("median")
        self.nn_r0_win_spin = self.double_spin(5, 0.001, 3600, 2)
        self.nn_early_spin = self.double_spin(5, 0.001, 3600, 2)
        self.nn_late_spin = self.double_spin(5, 0.001, 3600, 2)
        self.nn_trim_spin = self.double_spin(20, 0, 99.9, 1)
        self.nn_min_samples_spin = QtWidgets.QSpinBox()
        self.nn_min_samples_spin.setRange(1, 100000)
        self.nn_min_samples_spin.setValue(5)
        for label, widget in [
            ("Ts (s)", self.nn_ts_spin),
            ("Rref (ohm)", self.nn_rref_spin),
            ("Vs (V)", self.nn_vs_spin),
            ("ADC full-scale (V)", self.nn_vref_spin),
            ("ADC bits", self.nn_bits_spin),
            ("ADC mode", self.nn_adc_mode_combo),
            ("R0 method", self.nn_r0_method_combo),
            ("R0 window (s)", self.nn_r0_win_spin),
            ("Early window (s)", self.nn_early_spin),
            ("Late window (s)", self.nn_late_spin),
            ("Trim percent", self.nn_trim_spin),
            ("Min samples", self.nn_min_samples_spin),
        ]:
            proc.addWidget(QtWidgets.QLabel(label), row, 0)
            proc.addWidget(widget, row, 1)
            row += 1
        left_layout.addWidget(proc_box)

        pair_box = QtWidgets.QGroupBox("A-M pair representation per file")
        pair_grid = QtWidgets.QGridLayout(pair_box)
        self.nn_pair_mode_combo = QtWidgets.QComboBox()
        self.nn_pair_mode_combo.addItems(["first", "last", "specified", "average_all"])
        self.nn_pair_mode_combo.setCurrentText("first")
        self.nn_pair_index_spin = QtWidgets.QSpinBox()
        self.nn_pair_index_spin.setRange(1, 9999)
        self.nn_pair_index_spin.setValue(1)
        pair_grid.addWidget(QtWidgets.QLabel("Mode"), 0, 0)
        pair_grid.addWidget(self.nn_pair_mode_combo, 0, 1)
        pair_grid.addWidget(QtWidgets.QLabel("Specified pair"), 1, 0)
        pair_grid.addWidget(self.nn_pair_index_spin, 1, 1)
        pair_note = QtWidgets.QLabel("This uses the same file-level representation concept as the SVM training tab.")
        pair_note.setWordWrap(True)
        pair_grid.addWidget(pair_note, 2, 0, 1, 2)
        left_layout.addWidget(pair_box)

        prep_box = QtWidgets.QGroupBox("Feature preprocessing and optional PCA")
        prep = QtWidgets.QGridLayout(prep_box)
        self.nn_apply_pca_check = QtWidgets.QCheckBox("Apply PCA before neural network")
        self.nn_apply_pca_check.setChecked(True)
        self.nn_pca_dim_spin = QtWidgets.QSpinBox()
        self.nn_pca_dim_spin.setRange(1, 200)
        self.nn_pca_dim_spin.setValue(2)
        self.nn_drop_zero_var_check = QtWidgets.QCheckBox("Drop zero-variance features before training")
        self.nn_drop_zero_var_check.setChecked(True)
        self.nn_drop_time_feature_check = QtWidgets.QCheckBox("Drop time/slope/integration features before training")
        self.nn_drop_time_feature_check.setChecked(False)
        prep.addWidget(self.nn_apply_pca_check, 0, 0, 1, 2)
        prep.addWidget(QtWidgets.QLabel("PCA reduced features"), 1, 0)
        prep.addWidget(self.nn_pca_dim_spin, 1, 1)
        prep.addWidget(self.nn_drop_zero_var_check, 2, 0, 1, 2)
        prep.addWidget(self.nn_drop_time_feature_check, 3, 0, 1, 2)
        left_layout.addWidget(prep_box)

        nn_box = QtWidgets.QGroupBox("Neural network parameters")
        nn_grid = QtWidgets.QGridLayout(nn_box)
        self.nn_hidden_layers_spin = QtWidgets.QSpinBox()
        self.nn_hidden_layers_spin.setRange(1, 10)
        self.nn_hidden_layers_spin.setValue(2)
        self.nn_neurons_spin = QtWidgets.QSpinBox()
        self.nn_neurons_spin.setRange(1, 1000)
        self.nn_neurons_spin.setValue(32)
        self.nn_activation_combo = QtWidgets.QComboBox()
        self.nn_activation_combo.addItems(["relu", "tanh", "logistic", "identity"])
        self.nn_activation_combo.setCurrentText("relu")
        self.nn_solver_combo = QtWidgets.QComboBox()
        self.nn_solver_combo.addItems(["adam", "lbfgs", "sgd"])
        self.nn_solver_combo.setCurrentText("adam")
        self.nn_alpha_spin = self.double_spin(0.0001, 0.0, 100.0, 6)
        self.nn_lr_spin = self.double_spin(0.001, 0.000001, 10.0, 6)
        self.nn_max_iter_spin = QtWidgets.QSpinBox()
        self.nn_max_iter_spin.setRange(10, 100000)
        self.nn_max_iter_spin.setValue(1000)
        self.nn_random_state_spin = QtWidgets.QSpinBox()
        self.nn_random_state_spin.setRange(0, 999999)
        self.nn_random_state_spin.setValue(42)
        for r, (label, widget) in enumerate([
            ("Hidden layers", self.nn_hidden_layers_spin),
            ("Neurons/layer", self.nn_neurons_spin),
            ("Activation", self.nn_activation_combo),
            ("Solver", self.nn_solver_combo),
            ("L2 alpha", self.nn_alpha_spin),
            ("Learning rate", self.nn_lr_spin),
            ("Max iterations", self.nn_max_iter_spin),
            ("Random state", self.nn_random_state_spin),
        ]):
            nn_grid.addWidget(QtWidgets.QLabel(label), r, 0)
            nn_grid.addWidget(widget, r, 1)
        left_layout.addWidget(nn_box)

        self.nn_run_btn = QtWidgets.QPushButton("Run feature extraction + PCA/standardization + NN")
        self.nn_run_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.nn_run_btn.clicked.connect(self.run_nn_pipeline)
        left_layout.addWidget(self.nn_run_btn)

        export_row = QtWidgets.QHBoxLayout()
        self.nn_export_features_btn = QtWidgets.QPushButton("Export features")
        self.nn_export_features_btn.clicked.connect(self.export_nn_features)
        self.nn_export_features_btn.setEnabled(False)
        self.nn_export_predictions_btn = QtWidgets.QPushButton("Export predictions")
        self.nn_export_predictions_btn.clicked.connect(self.export_nn_predictions)
        self.nn_export_predictions_btn.setEnabled(False)
        self.nn_save_model_btn = QtWidgets.QPushButton("Export NN model")
        self.nn_save_model_btn.clicked.connect(self.save_nn_model)
        self.nn_save_model_btn.setEnabled(False)
        export_row.addWidget(self.nn_export_features_btn)
        export_row.addWidget(self.nn_export_predictions_btn)
        export_row.addWidget(self.nn_save_model_btn)
        left_layout.addLayout(export_row)

        self.nn_status_text = QtWidgets.QTextEdit()
        self.nn_status_text.setReadOnly(True)
        self.nn_status_text.setMinimumHeight(170)
        self.nn_status_text.setText("Ready.")
        left_layout.addWidget(self.nn_status_text)
        left_layout.addStretch(1)

        right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top)
        self.nn_figure = Figure(figsize=(8, 4.7))
        self.nn_canvas = FigureCanvas(self.nn_figure)
        top_layout.addWidget(self.nn_canvas)
        self.nn_result_text = QtWidgets.QTextEdit()
        self.nn_result_text.setReadOnly(True)
        self.nn_result_text.setMaximumHeight(170)
        top_layout.addWidget(self.nn_result_text)
        right_split.addWidget(top)

        tables = QtWidgets.QTabWidget()
        self.nn_feature_table_widget = QtWidgets.QTableWidget()
        self.nn_prediction_table_widget = QtWidgets.QTableWidget()
        self.nn_pair_table_widget = QtWidgets.QTableWidget()
        tables.addTab(self.nn_feature_table_widget, "File-level features")
        tables.addTab(self.nn_prediction_table_widget, "Predictions")
        tables.addTab(self.nn_pair_table_widget, "Pair-level features")
        right_split.addWidget(tables)
        right_split.setSizes([520, 300])

        nn_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        nn_splitter.addWidget(left)
        nn_splitter.addWidget(right_split)
        nn_splitter.setSizes([410, 1040])
        root.addWidget(nn_splitter, 1)

    def nn_browse_input_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select input Excel/CSV list for neural-network training",
            os.getcwd(),
            "Input list (*.xlsx *.xls *.csv);;Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self.nn_input_path_edit.setText(path)

    def current_nn_settings(self) -> ProcessingSettings:
        return ProcessingSettings(
            ts=float(self.nn_ts_spin.value()),
            rref=float(self.nn_rref_spin.value()),
            vs=float(self.nn_vs_spin.value()),
            vref_fs=float(self.nn_vref_spin.value()),
            adc_bits=int(self.nn_bits_spin.value()),
            adc_mode=str(self.nn_adc_mode_combo.currentText()),
            r0_method=str(self.nn_r0_method_combo.currentText()),
            r0_win_sec=float(self.nn_r0_win_spin.value()),
            trim_percent=float(self.nn_trim_spin.value()),
            min_samples=int(self.nn_min_samples_spin.value()),
            early_sec=float(self.nn_early_spin.value()),
            late_sec=float(self.nn_late_spin.value()),
            pair_mode=str(self.nn_pair_mode_combo.currentText()),
            specified_pair=int(self.nn_pair_index_spin.value()),
        )

    def nn_log(self, message: str):
        self.nn_status_text.append(str(message))
        QtWidgets.QApplication.processEvents()

    def run_nn_pipeline(self):
        try:
            self.nn_status_text.setText("Starting neural-network pipeline...")
            self.nn_result_text.clear()
            self.nn_figure.clear()
            self.nn_canvas.draw()
            input_path = self.nn_input_path_edit.text().strip()
            if not input_path or not os.path.isfile(input_path):
                QtWidgets.QMessageBox.warning(self, "Input file", "Please select a valid Excel/CSV input list.")
                return

            settings = self.current_nn_settings()
            feature_table, pair_table = build_feature_table_from_input_list(input_path, settings, self.nn_log)
            self.nn_feature_table = feature_table
            self.nn_pair_feature_table = pair_table
            self.show_dataframe(feature_table, self.nn_feature_table_widget, max_rows=300, max_cols=120)
            self.show_dataframe(pair_table, self.nn_pair_table_widget, max_rows=300, max_cols=120)
            self.nn_log(f"Extracted one feature row per file: {len(feature_table)} rows.")

            use_pca = bool(self.nn_apply_pca_check.isChecked())
            drop_zero_variance = bool(self.nn_drop_zero_var_check.isChecked())
            drop_time_features = bool(self.nn_drop_time_feature_check.isChecked())
            if use_pca:
                prep_pack = clean_standardize_pca(
                    feature_table,
                    int(self.nn_pca_dim_spin.value()),
                    drop_zero_variance=drop_zero_variance,
                    drop_time_features=drop_time_features,
                )
                prep_pack["apply_pca"] = True
            else:
                prep_pack = clean_standardize_classifier_features(
                    feature_table,
                    drop_zero_variance=drop_zero_variance,
                    drop_time_features=drop_time_features,
                )

            dropped_time = prep_pack.get("dropped_time_related_features", [])
            detected_time = prep_pack.get("time_related_features", [])
            dropped_zero = prep_pack.get("dropped_zero_variance_features", [])
            detected_zero = prep_pack.get("zero_variance_features", [])
            total_feature_count = prep_pack.get("total_feature_count", len(prep_pack.get("feature_cols", [])))
            if drop_time_features:
                self.nn_log(
                    f"Time/slope/integration feature dropout: ON. Dropped {len(dropped_time)} "
                    f"feature(s) from {total_feature_count} candidate numeric feature(s)."
                )
                if dropped_time:
                    self.nn_log("Dropped time/slope/integration features:")
                    self.nn_log("  " + "\n  ".join(map(str, dropped_time)))
            else:
                self.nn_log(
                    f"Time/slope/integration feature dropout: OFF. Kept {len(detected_time)} "
                    f"time/slope/integration-related feature(s)."
                )
            if drop_zero_variance:
                self.nn_log(
                    f"Zero-variance feature dropout: ON. Dropped {len(dropped_zero)} "
                    f"feature(s) from {total_feature_count} candidate numeric feature(s)."
                )
                if dropped_zero:
                    self.nn_log("Dropped zero-variance features:")
                    self.nn_log("  " + "\n  ".join(map(str, dropped_zero)))
            else:
                self.nn_log(
                    f"Zero-variance feature dropout: OFF. Using all {len(prep_pack['feature_cols'])} "
                    f"candidate numeric feature(s). Detected zero-variance features kept: {len(detected_zero)}."
                )

            X_all_model = prep_pack["scores_all"]
            train_mask = prep_pack["train_mask"]
            valid_mask = prep_pack["valid_mask"]
            y = feature_table["Label"].to_numpy()
            y_train = y[train_mask]
            y_valid = y[valid_mask]
            X_train = X_all_model[train_mask, :]
            X_valid = X_all_model[valid_mask, :]

            if len(np.unique(y_train)) < 2:
                raise ValueError("Training data must contain at least two classes for neural-network training.")

            nn = self.train_neural_network(X_train, y_train)
            pred_train = nn.predict(X_train)
            pred_valid = nn.predict(X_valid) if X_valid.size else np.array([], dtype=int)
            train_acc = accuracy_score(y_train, pred_train) * 100
            valid_acc = accuracy_score(y_valid, pred_valid) * 100 if pred_valid.size else np.nan

            prediction_rows = []
            train_indices = np.flatnonzero(train_mask)
            valid_indices = np.flatnonzero(valid_mask)
            for local_i, idx in enumerate(train_indices):
                prediction_rows.append({
                    "DataType": feature_table.iloc[idx]["DataType"],
                    "FilePath": feature_table.iloc[idx]["FilePath"],
                    "TrueLabel": int(y_train[local_i]),
                    "PredictedLabel": int(pred_train[local_i]),
                    "Correct": bool(pred_train[local_i] == y_train[local_i]),
                    "PairSelection": feature_table.iloc[idx].get("PairSelection", ""),
                    "NumPairs": feature_table.iloc[idx].get("NumPairs", np.nan),
                })
            for local_i, idx in enumerate(valid_indices):
                prediction_rows.append({
                    "DataType": feature_table.iloc[idx]["DataType"],
                    "FilePath": feature_table.iloc[idx]["FilePath"],
                    "TrueLabel": int(y_valid[local_i]),
                    "PredictedLabel": int(pred_valid[local_i]),
                    "Correct": bool(pred_valid[local_i] == y_valid[local_i]),
                    "PairSelection": feature_table.iloc[idx].get("PairSelection", ""),
                    "NumPairs": feature_table.iloc[idx].get("NumPairs", np.nan),
                })
            pred_df = pd.DataFrame(prediction_rows)
            self.nn_prediction_table = pred_df
            self.show_dataframe(pred_df, self.nn_prediction_table_widget, max_rows=300, max_cols=80)

            self.plot_nn_scores(X_train, y_train, X_valid, y_valid, prep_pack)

            hidden_layers = int(self.nn_hidden_layers_spin.value())
            neurons = int(self.nn_neurons_spin.value())
            report = []
            report.append("Neural-network training finished.\n")
            report.append(f"Files: {len(feature_table)}")
            report.append(f"Training samples: {int(train_mask.sum())}")
            report.append(f"Validation samples: {int(valid_mask.sum())}")
            report.append(f"Candidate numeric features: {total_feature_count}")
            report.append(f"Features used after dropout: {len(prep_pack['feature_cols'])}")
            report.append(f"PCA applied: {'Yes' if use_pca else 'No'}")
            if use_pca:
                report.append(f"PCA reduced features fed to NN: {prep_pack['n_pc']}")
                report.append(f"Explained variance of used PCs: {100*np.sum(prep_pack['pca'].explained_variance_ratio_):.2f}%")
            else:
                report.append(f"Standardized original features fed to NN: {X_train.shape[1]}")
            report.append(f"Hidden layers: {hidden_layers}")
            report.append(f"Neurons per layer: {neurons}")
            report.append(f"Activation: {self.nn_activation_combo.currentText()}")
            report.append(f"Solver: {self.nn_solver_combo.currentText()}")
            report.append(f"Training accuracy: {train_acc:.2f}%")
            if np.isfinite(valid_acc):
                report.append(f"Validation accuracy: {valid_acc:.2f}%")
            report.append("\nNN classification report, training:")
            report.append(classification_report(y_train, pred_train, zero_division=0))
            if pred_valid.size:
                report.append("NN classification report, validation:")
                report.append(classification_report(y_valid, pred_valid, zero_division=0))
            self.nn_result_text.setText("\n".join(report))

            nn_params = {
                "hidden_layer_sizes": tuple([neurons] * hidden_layers),
                "hidden_layers": hidden_layers,
                "neurons_per_layer": neurons,
                "activation": self.nn_activation_combo.currentText(),
                "solver": self.nn_solver_combo.currentText(),
                "alpha": float(self.nn_alpha_spin.value()),
                "learning_rate_init": float(self.nn_lr_spin.value()),
                "max_iter": int(self.nn_max_iter_spin.value()),
                "random_state": int(self.nn_random_state_spin.value()),
            }

            self.nn_model_package = {
                "model_type": "enose_pca_nn_r0_feature_model",
                "classifier_type": "neural_network_mlp",
                "app_version": "matlab_features_v7",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "settings": settings.__dict__,
                "feature_columns": list(prep_pack["feature_cols"]),
                "total_candidate_feature_count": int(total_feature_count),
                "drop_time_features": bool(prep_pack.get("drop_time_features", False)),
                "time_related_features": list(map(str, prep_pack.get("time_related_features", []))),
                "dropped_time_related_features": list(map(str, prep_pack.get("dropped_time_related_features", []))),
                "drop_zero_variance": bool(prep_pack.get("drop_zero_variance", False)),
                "zero_variance_features": list(map(str, prep_pack.get("zero_variance_features", []))),
                "dropped_zero_variance_features": list(map(str, prep_pack.get("dropped_zero_variance_features", []))),
                "medians": prep_pack["medians"],
                "scaler": prep_pack["scaler"],
                "apply_pca": bool(use_pca),
                "pca": prep_pack.get("pca", None),
                "n_reduced_features": int(prep_pack["n_pc"]),
                "requested_pca_dimensions": int(self.nn_pca_dim_spin.value()) if use_pca else 0,
                "explained_variance_ratio": list(map(float, prep_pack["pca"].explained_variance_ratio_)) if use_pca and prep_pack.get("pca") is not None else [],
                "classifier": nn,
                "nn": nn,
                "nn_params": nn_params,
                "class_labels": list(map(int, nn.classes_)) if np.issubdtype(np.asarray(nn.classes_).dtype, np.integer) else list(map(str, nn.classes_)),
                "feature_definitions": FEATURE_DEFINITIONS,
                "training_files": feature_table[["DataType", "FilePath", "Label", "NumPairs", "PairSelection"]].to_dict(orient="records"),
            }
            self.nn_export_features_btn.setEnabled(True)
            self.nn_export_predictions_btn.setEnabled(True)
            self.nn_save_model_btn.setEnabled(True)
            self.nn_log("Neural-network pipeline finished successfully.")
        except Exception as exc:
            tb = traceback.format_exc()
            self.nn_log(tb)
            QtWidgets.QMessageBox.critical(self, "Neural-network training error", str(exc))

    def train_neural_network(self, X: np.ndarray, y: np.ndarray) -> MLPClassifier:
        hidden_layers = int(self.nn_hidden_layers_spin.value())
        neurons = int(self.nn_neurons_spin.value())
        hidden_layer_sizes = tuple([neurons] * hidden_layers)
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=self.nn_activation_combo.currentText(),
            solver=self.nn_solver_combo.currentText(),
            alpha=float(self.nn_alpha_spin.value()),
            learning_rate_init=float(self.nn_lr_spin.value()),
            max_iter=int(self.nn_max_iter_spin.value()),
            random_state=int(self.nn_random_state_spin.value()),
            early_stopping=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X, y)
        return clf

    def plot_nn_scores(self, X_train, y_train, X_valid, y_valid, prep_pack: Dict[str, object]):
        self.nn_figure.clear()
        n_dim = X_train.shape[1]
        apply_pca = bool(prep_pack.get("apply_pca", False))
        if n_dim < 2:
            ax = self.nn_figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Plot requires at least 2 model-input features. Current = {n_dim}.", ha="center", va="center")
            ax.set_axis_off()
            self.nn_canvas.draw()
            return
        plot_dim = 2 if n_dim == 2 else 3
        component_indices = list(range(plot_dim))
        train_plot = X_train[:, component_indices]
        valid_plot = X_valid[:, component_indices] if X_valid.size else np.empty((0, plot_dim))
        labels = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_valid)])) if valid_plot.size else np.unique(y_train)
        colors = plt_colors(len(labels))
        label_to_color = {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}

        if plot_dim == 2:
            ax_train = self.nn_figure.add_subplot(1, 2, 1)
            ax_valid = self.nn_figure.add_subplot(1, 2, 2)
        else:
            ax_train = self.nn_figure.add_subplot(1, 2, 1, projection="3d")
            ax_valid = self.nn_figure.add_subplot(1, 2, 2, projection="3d")

        explained = prep_pack["pca"].explained_variance_ratio_ if apply_pca and prep_pack.get("pca") is not None else []
        def axis_label(i: int) -> str:
            if apply_pca:
                if i < len(explained):
                    return f"PC{i+1} ({100*float(explained[i]):.1f}%)"
                return f"PC{i+1}"
            return f"Input feature {i+1}"

        for ax, scores, y, title in [(ax_train, train_plot, y_train, "NN training data"), (ax_valid, valid_plot, y_valid, "NN validation data")]:
            if scores.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(title)
                continue
            for lab in labels:
                idx = np.asarray(y) == lab
                if not np.any(idx):
                    continue
                if plot_dim == 2:
                    ax.scatter(scores[idx, 0], scores[idx, 1], s=45, color=label_to_color[lab], label=f"Class {lab}", alpha=0.85)
                else:
                    ax.scatter(scores[idx, 0], scores[idx, 1], scores[idx, 2], s=45, color=label_to_color[lab], label=f"Class {lab}", alpha=0.85)
            ax.set_xlabel(axis_label(0))
            ax.set_ylabel(axis_label(1))
            if plot_dim == 3:
                ax.set_zlabel(axis_label(2))
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        self.nn_figure.tight_layout()
        self.nn_canvas.draw()

    def export_nn_features(self):
        if self.nn_feature_table is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export NN file-level features", os.getcwd(), "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                self.nn_feature_table.to_csv(path, index=False)
            else:
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    self.nn_feature_table.to_excel(writer, sheet_name="file_level_features", index=False)
                    if self.nn_pair_feature_table is not None:
                        self.nn_pair_feature_table.to_excel(writer, sheet_name="pair_level_features", index=False)
                    pd.DataFrame([{"Feature": k, "Definition": v} for k, v in FEATURE_DEFINITIONS.items()]).to_excel(writer, sheet_name="feature_definitions", index=False)
            self.nn_log(f"NN features exported: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))

    def export_nn_predictions(self):
        if self.nn_prediction_table is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export NN predictions", os.getcwd(), "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                self.nn_prediction_table.to_csv(path, index=False)
            else:
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    self.nn_prediction_table.to_excel(writer, sheet_name="predictions", index=False)
                    if self.nn_feature_table is not None:
                        self.nn_feature_table.to_excel(writer, sheet_name="file_level_features", index=False)
            self.nn_log(f"NN predictions exported: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))

    def save_nn_model(self):
        if self.nn_model_package is None:
            return
        default_name = datetime.now().strftime("enose_nn_classification_model_%Y%m%d_%H%M%S.joblib")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export neural-network model package", os.path.join(os.getcwd(), default_name), "Joblib (*.joblib)")
        if not path:
            return
        try:
            joblib.dump(self.nn_model_package, path)
            self.nn_log(f"Neural-network model exported: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save NN model error", str(exc))

    def _build_testing_tab(self):
        root = QtWidgets.QHBoxLayout(self.test_tab)

        left = QtWidgets.QVBoxLayout()

        model_box = QtWidgets.QGroupBox("Load exported classification model")
        model_grid = QtWidgets.QGridLayout(model_box)
        self.test_model_path_edit = QtWidgets.QLineEdit()
        self.test_model_path_edit.setReadOnly(True)
        self.load_model_btn = QtWidgets.QPushButton("Load model...")
        self.load_model_btn.clicked.connect(self.load_model_for_testing)
        self.test_model_info_label = QtWidgets.QLabel("No model loaded.")
        self.test_model_info_label.setWordWrap(True)
        model_grid.addWidget(QtWidgets.QLabel("Model file"), 0, 0)
        model_grid.addWidget(self.test_model_path_edit, 0, 1)
        model_grid.addWidget(self.load_model_btn, 0, 2)
        model_grid.addWidget(self.test_model_info_label, 1, 0, 1, 3)
        left.addWidget(model_box)

        files_box = QtWidgets.QGroupBox("Testing CSV files")
        files_grid = QtWidgets.QGridLayout(files_box)
        self.test_file_table = QtWidgets.QTableWidget()
        self.test_file_table.setColumnCount(3)
        self.test_file_table.setHorizontalHeaderLabels(["CSV file", "True label (optional)", "Status"])
        self.test_file_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.test_file_table.horizontalHeader().setStretchLastSection(True)
        self.test_file_table.setMinimumHeight(260)

        self.add_test_files_btn = QtWidgets.QPushButton("Add testing files...")
        self.add_test_files_btn.clicked.connect(self.add_testing_files)
        self.remove_test_files_btn = QtWidgets.QPushButton("Remove selected")
        self.remove_test_files_btn.clicked.connect(self.remove_selected_testing_files)
        self.clear_test_files_btn = QtWidgets.QPushButton("Clear files")
        self.clear_test_files_btn.clicked.connect(self.clear_testing_files)
        self.classify_test_btn = QtWidgets.QPushButton("Classify testing files")
        self.classify_test_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.classify_test_btn.clicked.connect(self.classify_testing_files)
        self.export_test_predictions_btn = QtWidgets.QPushButton("Export testing results")
        self.export_test_predictions_btn.clicked.connect(self.export_testing_predictions)
        self.export_test_predictions_btn.setEnabled(False)

        files_grid.addWidget(self.test_file_table, 0, 0, 6, 1)
        files_grid.addWidget(self.add_test_files_btn, 0, 1)
        files_grid.addWidget(self.remove_test_files_btn, 1, 1)
        files_grid.addWidget(self.clear_test_files_btn, 2, 1)
        files_grid.addWidget(self.classify_test_btn, 3, 1)
        files_grid.addWidget(self.export_test_predictions_btn, 4, 1)
        file_note = QtWidgets.QLabel("After adding files, you may type the true label in the second column. Leave it blank when the true class is unknown.")
        file_note.setWordWrap(True)
        files_grid.addWidget(file_note, 5, 1)
        left.addWidget(files_box)

        self.test_summary_text = QtWidgets.QTextEdit()
        self.test_summary_text.setReadOnly(True)
        self.test_summary_text.setMinimumHeight(190)
        self.test_summary_text.setText("Load a model and add testing CSV files.")
        left.addWidget(self.test_summary_text)

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_widget)
        left_scroll.setMinimumWidth(360)

        right_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)
        self.test_figure = Figure(figsize=(8, 4.5))
        self.test_canvas = FigureCanvas(self.test_figure)
        plot_layout.addWidget(self.test_canvas)
        right_split.addWidget(plot_widget)

        test_tables = QtWidgets.QTabWidget()
        self.test_prediction_table_widget = QtWidgets.QTableWidget()
        self.test_feature_table_widget = QtWidgets.QTableWidget()
        test_tables.addTab(self.test_prediction_table_widget, "Testing predictions")
        test_tables.addTab(self.test_feature_table_widget, "Testing file features")
        right_split.addWidget(test_tables)
        right_split.setSizes([430, 360])

        test_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        test_splitter.addWidget(left_scroll)
        test_splitter.addWidget(right_split)
        test_splitter.setSizes([480, 950])
        root.addWidget(test_splitter, 1)


    def _table_box(self, title: str, table: QtWidgets.QTableWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(False)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(table)
        return box

    @staticmethod
    def double_spin(value: float, minimum: float, maximum: float, decimals: int) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSingleStep(0.1)
        return spin

    def browse_input_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select input Excel/CSV list",
            os.getcwd(),
            "Input list (*.xlsx *.xls *.csv);;Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*.*)",
        )
        if path:
            self.input_path_edit.setText(path)

    def current_settings(self) -> ProcessingSettings:
        return ProcessingSettings(
            ts=float(self.ts_spin.value()),
            rref=float(self.rref_spin.value()),
            vs=float(self.vs_spin.value()),
            vref_fs=float(self.vref_spin.value()),
            adc_bits=int(self.bits_spin.value()),
            adc_mode=str(self.adc_mode_combo.currentText()),
            r0_method=str(self.r0_method_combo.currentText()),
            r0_win_sec=float(self.r0_win_spin.value()),
            trim_percent=float(self.trim_spin.value()),
            min_samples=int(self.min_samples_spin.value()),
            early_sec=float(self.early_spin.value()),
            late_sec=float(self.late_spin.value()),
            pair_mode=str(self.pair_mode_combo.currentText()),
            specified_pair=int(self.pair_index_spin.value()),
        )

    def log(self, message: str):
        self.status_text.append(str(message))
        QtWidgets.QApplication.processEvents()

    def run_pipeline(self):
        try:
            self.status_text.setText("Starting pipeline...")
            self.result_text.clear()
            self.figure.clear()
            self.canvas.draw()
            input_path = self.input_path_edit.text().strip()
            if not input_path or not os.path.isfile(input_path):
                QtWidgets.QMessageBox.warning(self, "Input file", "Please select a valid Excel/CSV input list.")
                return
            settings = self.current_settings()
            feature_table, pair_table = build_feature_table_from_input_list(input_path, settings, self.log)
            self.feature_table = feature_table
            self.pair_feature_table = pair_table
            self.show_dataframe(feature_table, self.feature_table_widget, max_rows=300, max_cols=120)
            self.show_dataframe(pair_table, self.pair_table_widget, max_rows=300, max_cols=120)
            self.log(f"Extracted one feature row per file: {len(feature_table)} rows.")

            drop_zero_variance = bool(self.drop_zero_var_check.isChecked())
            drop_time_features = bool(self.drop_time_feature_check.isChecked())
            pca_pack = clean_standardize_pca(
                feature_table,
                int(self.pca_dim_spin.value()),
                drop_zero_variance=drop_zero_variance,
                drop_time_features=drop_time_features,
            )
            dropped_time = pca_pack.get("dropped_time_related_features", [])
            detected_time = pca_pack.get("time_related_features", [])
            dropped_zero = pca_pack.get("dropped_zero_variance_features", [])
            detected_zero = pca_pack.get("zero_variance_features", [])
            total_feature_count = pca_pack.get("total_feature_count", len(pca_pack.get("feature_cols", [])))
            if drop_time_features:
                self.log(
                    f"Time/slope/integration feature dropout: ON. Dropped {len(dropped_time)} "
                    f"feature(s) from {total_feature_count} candidate numeric feature(s)."
                )
                if dropped_time:
                    self.log("Dropped time/slope/integration features:")
                    self.log("  " + "\n  ".join(map(str, dropped_time)))
            else:
                self.log(
                    f"Time/slope/integration feature dropout: OFF. Kept {len(detected_time)} "
                    f"time/slope/integration-related feature(s)."
                )
            if drop_zero_variance:
                self.log(
                    f"Zero-variance feature dropout: ON. Dropped {len(dropped_zero)} "
                    f"feature(s) from {total_feature_count} candidate numeric feature(s)."
                )
                if dropped_zero:
                    self.log("Dropped zero-variance features:")
                    self.log("  " + "\n  ".join(map(str, dropped_zero)))
            else:
                self.log(
                    f"Zero-variance feature dropout: OFF. Using all {len(pca_pack['feature_cols'])} "
                    f"candidate numeric feature(s). Detected zero-variance features kept: {len(detected_zero)}."
                )
            scores = pca_pack["scores_all"]
            train_mask = pca_pack["train_mask"]
            valid_mask = pca_pack["valid_mask"]
            y = feature_table["Label"].to_numpy()
            y_train = y[train_mask]
            y_valid = y[valid_mask]
            scores_train = scores[train_mask, :]
            scores_valid = scores[valid_mask, :]

            if len(np.unique(y_train)) < 2:
                raise ValueError("Training data must contain at least two classes for SVM.")
            svm = self.train_svm(scores_train, y_train)
            pred_train = svm.predict(scores_train)
            pred_valid = svm.predict(scores_valid) if scores_valid.size else np.array([], dtype=int)
            train_acc = accuracy_score(y_train, pred_train) * 100
            valid_acc = accuracy_score(y_valid, pred_valid) * 100 if pred_valid.size else np.nan

            prediction_rows = []
            train_indices = np.flatnonzero(train_mask)
            valid_indices = np.flatnonzero(valid_mask)
            for local_i, idx in enumerate(train_indices):
                prediction_rows.append({
                    "DataType": feature_table.iloc[idx]["DataType"],
                    "FilePath": feature_table.iloc[idx]["FilePath"],
                    "TrueLabel": int(y_train[local_i]),
                    "PredictedLabel": int(pred_train[local_i]),
                    "Correct": bool(pred_train[local_i] == y_train[local_i]),
                    "PairSelection": feature_table.iloc[idx].get("PairSelection", ""),
                    "NumPairs": feature_table.iloc[idx].get("NumPairs", np.nan),
                })
            for local_i, idx in enumerate(valid_indices):
                prediction_rows.append({
                    "DataType": feature_table.iloc[idx]["DataType"],
                    "FilePath": feature_table.iloc[idx]["FilePath"],
                    "TrueLabel": int(y_valid[local_i]),
                    "PredictedLabel": int(pred_valid[local_i]),
                    "Correct": bool(pred_valid[local_i] == y_valid[local_i]),
                    "PairSelection": feature_table.iloc[idx].get("PairSelection", ""),
                    "NumPairs": feature_table.iloc[idx].get("NumPairs", np.nan),
                })
            pred_df = pd.DataFrame(prediction_rows)
            self.prediction_table = pred_df
            self.show_dataframe(pred_df, self.prediction_table_widget, max_rows=300, max_cols=80)

            self.latest_train_scores = scores_train
            self.latest_valid_scores = scores_valid
            self.latest_y_train = y_train
            self.latest_y_valid = y_valid
            self.latest_explained = pca_pack["pca"].explained_variance_ratio_
            self.populate_pca_plot_component_combos(pca_pack["n_pc"])
            self.plot_scores(scores_train, y_train, scores_valid, y_valid, self.latest_explained)

            report = []
            report.append("Finished.\n")
            report.append(f"Files: {len(feature_table)}")
            report.append(f"Training samples: {int(train_mask.sum())}")
            report.append(f"Validation samples: {int(valid_mask.sum())}")
            report.append(f"Candidate numeric features: {pca_pack.get('total_feature_count', len(pca_pack['feature_cols']))}")
            if pca_pack.get("drop_time_features", False):
                report.append(f"Time/slope/integration features dropped: {len(pca_pack.get('dropped_time_related_features', []))}")
                if pca_pack.get("dropped_time_related_features", []):
                    report.append("Dropped time/slope/integration features: " + ", ".join(map(str, pca_pack.get("dropped_time_related_features", []))))
            else:
                report.append("Time/slope/integration feature dropout: OFF; these features were kept.")
            report.append(f"Features used for PCA: {len(pca_pack['feature_cols'])}")
            if pca_pack.get("drop_zero_variance", False):
                report.append(f"Zero-variance features dropped: {len(pca_pack.get('dropped_zero_variance_features', []))}")
                if pca_pack.get("dropped_zero_variance_features", []):
                    report.append("Dropped features: " + ", ".join(map(str, pca_pack.get("dropped_zero_variance_features", []))))
            else:
                report.append("Zero-variance feature dropout: OFF; all candidate numeric features were kept.")
            report.append(f"PCA dimensions used: {pca_pack['n_pc']}")
            report.append(f"Explained variance of used PCs: {100*np.sum(pca_pack['pca'].explained_variance_ratio_):.2f}%")
            report.append(f"Training accuracy: {train_acc:.2f}%")
            if np.isfinite(valid_acc):
                report.append(f"Validation accuracy: {valid_acc:.2f}%")
            report.append("\nSVM classification report, training:")
            report.append(classification_report(y_train, pred_train, zero_division=0))
            if pred_valid.size:
                report.append("SVM classification report, validation:")
                report.append(classification_report(y_valid, pred_valid, zero_division=0))
            self.result_text.setText("\n".join(report))

            self.model_package = {
                "model_type": "enose_pca_svm_r0_feature_model",
                "app_version": "matlab_features_v7",
                "created_at": datetime.now().isoformat(timespec="seconds"),

                # All parameters used before feature extraction and before PCA/SVM.
                "settings": settings.__dict__,

                # Exact feature processing state. These must be reused for testing data.
                "feature_columns": list(pca_pack["feature_cols"]),
                "total_candidate_feature_count": int(pca_pack.get("total_feature_count", len(pca_pack["feature_cols"]))),
                "drop_time_features": bool(pca_pack.get("drop_time_features", False)),
                "time_related_features": list(map(str, pca_pack.get("time_related_features", []))),
                "dropped_time_related_features": list(map(str, pca_pack.get("dropped_time_related_features", []))),
                "drop_zero_variance": bool(pca_pack.get("drop_zero_variance", False)),
                "zero_variance_features": list(map(str, pca_pack.get("zero_variance_features", []))),
                "dropped_zero_variance_features": list(map(str, pca_pack.get("dropped_zero_variance_features", []))),
                "medians": pca_pack["medians"],
                "scaler": pca_pack["scaler"],

                # PCA reduction model.
                "apply_pca": True,
                "pca": pca_pack["pca"],
                "n_reduced_features": int(pca_pack["n_pc"]),
                "requested_pca_dimensions": int(self.pca_dim_spin.value()),
                "explained_variance_ratio": list(map(float, pca_pack["pca"].explained_variance_ratio_)),

                # Classification model.
                "classifier_type": "svm",
                "classifier": svm,
                "svm": svm,
                "svm_params": svm.get_params(),
                "class_labels": list(map(int, svm.classes_)) if np.issubdtype(np.asarray(svm.classes_).dtype, np.integer) else list(map(str, svm.classes_)),

                "feature_definitions": FEATURE_DEFINITIONS,
                "training_files": feature_table[["DataType", "FilePath", "Label", "NumPairs", "PairSelection"]].to_dict(orient="records"),
            }
            self.export_features_btn.setEnabled(True)
            self.export_predictions_btn.setEnabled(True)
            self.save_model_btn.setEnabled(True)
            self.log("Pipeline finished successfully.")
        except Exception as exc:
            tb = traceback.format_exc()
            self.log(tb)
            QtWidgets.QMessageBox.critical(self, "Pipeline error", str(exc))

    def train_svm(self, X: np.ndarray, y: np.ndarray) -> SVC:
        gamma_mode = self.gamma_combo.currentText()
        gamma = float(self.gamma_value_spin.value()) if gamma_mode == "custom" else gamma_mode
        class_weight = "balanced" if self.class_weight_check.isChecked() else None
        clf = SVC(
            kernel=self.svm_kernel_combo.currentText(),
            C=float(self.c_spin.value()),
            gamma=gamma,
            degree=int(self.degree_spin.value()),
            class_weight=class_weight,
            probability=True,
        )
        clf.fit(X, y)
        return clf

    def pca_plot_combos(self):
        return [self.pca_plot_pc1_combo, self.pca_plot_pc2_combo, self.pca_plot_pc3_combo]

    def set_pca_plot_selector_visible(self, visible: bool):
        self.pca_plot_selector_label.setVisible(bool(visible))
        self.pca_plot_selector_widget.setVisible(bool(visible))

    def update_pca_plot_selector_visibility(self):
        if self.latest_train_scores is not None:
            n_pc = int(self.latest_train_scores.shape[1])
            self.set_pca_plot_selector_visible(n_pc > 3)
            return

        n_pc = int(self.pca_dim_spin.value())
        if n_pc > 3:
            # Before a model is trained, populate the selectors from the requested
            # PCA dimension so the controls appear immediately and clearly.
            self.populate_pca_plot_component_combos(n_pc)
        else:
            self.set_pca_plot_selector_visible(False)

    def populate_pca_plot_component_combos(self, n_pc: int):
        n_pc = int(n_pc)
        items = ["N/A"] + [f"PC{i}" for i in range(1, n_pc + 1)]
        defaults = ["PC1", "PC2", "PC3"]
        for combo, default in zip(self.pca_plot_combos(), defaults):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.setCurrentText(default if default in items else "N/A")
            combo.blockSignals(False)
        self.set_pca_plot_selector_visible(n_pc > 3)

    def selected_pca_plot_components(self, n_pc: int) -> Tuple[List[int], str]:
        """Return selected PCA component zero-based indices and an optional warning."""
        if n_pc <= 3:
            return list(range(n_pc)), ""

        selected: List[int] = []
        for combo in self.pca_plot_combos():
            text = combo.currentText().strip().upper()
            if text in {"", "N/A", "NONE"}:
                continue
            if text.startswith("PC"):
                try:
                    idx = int(text[2:]) - 1
                except ValueError:
                    continue
                if 0 <= idx < n_pc:
                    selected.append(idx)

        if len(selected) != len(set(selected)):
            return [], "Please select different PCA components. Duplicate PCs are not allowed."

        if len(selected) not in (2, 3):
            return [], "Select exactly two PCA components for 2D plotting, or three PCA components for 3D plotting. Use N/A to discard a selector."

        return selected, ""

    def refresh_training_pca_plot(self):
        if self.latest_train_scores is None or self.latest_y_train is None or self.latest_explained is None:
            return
        valid_scores = self.latest_valid_scores if self.latest_valid_scores is not None else np.empty((0, 0))
        y_valid = self.latest_y_valid if self.latest_y_valid is not None else np.array([])
        self.plot_scores(
            self.latest_train_scores,
            self.latest_y_train,
            valid_scores,
            y_valid,
            self.latest_explained,
        )

    def plot_scores(self, train_scores, y_train, valid_scores, y_valid, explained, component_indices: Optional[List[int]] = None):
        self.figure.clear()
        n_pc = train_scores.shape[1]

        if component_indices is None:
            component_indices, warning = self.selected_pca_plot_components(n_pc)
        else:
            component_indices = list(component_indices)
            warning = ""

        plot_dim = len(component_indices)

        if warning or plot_dim not in (2, 3):
            ax = self.figure.add_subplot(111)
            message = warning if warning else "PCA plot requires exactly two or three selected components."
            if n_pc > 3:
                message += "\nUse the PCA plot PC drop-down lists to select PC components."
            else:
                message += f"\nCurrent PCA dimension = {n_pc}."
            ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
            ax.set_axis_off()
            self.canvas.draw()
            return

        train_plot = train_scores[:, component_indices]
        if valid_scores.size:
            valid_plot = valid_scores[:, component_indices]
        else:
            valid_plot = np.empty((0, plot_dim))

        labels = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_valid)])) if valid_plot.size else np.unique(y_train)
        colors = plt_colors(len(labels))
        label_to_color = {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}

        if plot_dim == 2:
            ax_train = self.figure.add_subplot(1, 2, 1)
            ax_valid = self.figure.add_subplot(1, 2, 2)
        else:
            ax_train = self.figure.add_subplot(1, 2, 1, projection="3d")
            ax_valid = self.figure.add_subplot(1, 2, 2, projection="3d")

        self._plot_one_score_axis(ax_train, train_plot, y_train, labels, label_to_color, plot_dim, "Training data", explained, component_indices)
        self._plot_one_score_axis(ax_valid, valid_plot, y_valid, labels, label_to_color, plot_dim, "Validation data", explained, component_indices)
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_one_score_axis(self, ax, scores, y, labels, label_to_color, plot_dim, title, explained, component_indices):
        if scores.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            return
        for lab in labels:
            idx = np.asarray(y) == lab
            if not np.any(idx):
                continue
            if plot_dim == 2:
                ax.scatter(scores[idx, 0], scores[idx, 1], s=45, color=label_to_color[lab], label=f"Class {lab}", alpha=0.85)
            else:
                ax.scatter(scores[idx, 0], scores[idx, 1], scores[idx, 2], s=45, color=label_to_color[lab], label=f"Class {lab}", alpha=0.85)

        def axis_label(local_axis: int) -> str:
            pc_index = component_indices[local_axis]
            if pc_index < len(explained):
                return f"PC{pc_index + 1} ({explained[pc_index] * 100:.1f}%)"
            return f"PC{pc_index + 1}"

        ax.set_xlabel(axis_label(0))
        ax.set_ylabel(axis_label(1))
        if plot_dim == 3:
            ax.set_zlabel(axis_label(2))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    def export_features(self):
        if self.feature_table is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export file-level features", os.getcwd(), "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                self.feature_table.to_csv(path, index=False)
            else:
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    self.feature_table.to_excel(writer, sheet_name="file_level_features", index=False)
                    if self.pair_feature_table is not None:
                        self.pair_feature_table.to_excel(writer, sheet_name="pair_level_features", index=False)
                    pd.DataFrame([{"Feature": k, "Definition": v} for k, v in FEATURE_DEFINITIONS.items()]).to_excel(writer, sheet_name="feature_definitions", index=False)
            self.log(f"Features exported: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))

    def export_predictions(self):
        if self.prediction_table is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export predictions", os.getcwd(), "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                self.prediction_table.to_csv(path, index=False)
            else:
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    self.prediction_table.to_excel(writer, sheet_name="predictions", index=False)
                    if self.feature_table is not None:
                        self.feature_table.to_excel(writer, sheet_name="file_level_features", index=False)
            self.log(f"Predictions exported: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))

    def save_model(self):
        if self.model_package is None:
            return
        default_name = datetime.now().strftime("enose_classification_model_%Y%m%d_%H%M%S.joblib")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export classification model package", os.path.join(os.getcwd(), default_name), "Joblib (*.joblib)")
        if not path:
            return
        try:
            joblib.dump(self.model_package, path)
            self.log(f"Classification model exported: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save model error", str(exc))


    # ------------------------------------------------------------------
    # Tab 2: load exported model and classify new testing CSV files
    # ------------------------------------------------------------------
    def load_model_for_testing(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load exported classification model",
            os.getcwd(),
            "Joblib (*.joblib);;All files (*.*)",
        )
        if not path:
            return
        try:
            package = joblib.load(path)
            self.validate_model_package(package)
            self.loaded_model_package = package
            self.loaded_model_path = path
            self.test_model_path_edit.setText(path)
            pca_obj = package.get("pca", None)
            n_pc = int(package.get("n_reduced_features", getattr(pca_obj, "n_components_", 0)))
            settings = package.get("settings", {})
            pair_mode = settings.get("pair_mode", "") if isinstance(settings, dict) else ""
            explained = package.get("explained_variance_ratio", []) or []
            explained_text = ", ".join([f"PC{i+1}={100*float(v):.2f}%" for i, v in enumerate(explained[:5])])
            if explained_text:
                explained_text = " | " + explained_text
            classifier_type = package.get("classifier_type", "svm" if "svm" in package else "classifier")
            pca_state = "PCA" if bool(package.get("apply_pca", True)) and package.get("pca", None) is not None else "No PCA"
            self.test_model_info_label.setText(
                f"Loaded: {os.path.basename(path)} | classifier={classifier_type} | {pca_state} | "
                f"model-input features={n_pc} | original features={len(package.get('feature_columns', []))} | "
                f"pair mode={pair_mode}{explained_text}"
            )
            self.test_summary_text.setText("Model loaded. Add testing CSV files and press Classify testing files.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load model error", str(exc))
            self.test_summary_text.setText(f"Load model error: {exc}")

    @staticmethod
    def validate_model_package(package: Dict[str, object]):
        required = ["settings", "feature_columns", "medians", "scaler"]
        missing = [key for key in required if key not in package]
        if missing:
            raise ValueError("Invalid model package. Missing: " + ", ".join(missing))
        if not any(key in package for key in ["classifier", "svm", "nn"]):
            raise ValueError("Invalid model package. Missing classifier model: expected classifier, svm, or nn.")
        apply_pca = bool(package.get("apply_pca", True))
        if apply_pca and package.get("pca", None) is None:
            raise ValueError("Invalid model package. PCA is enabled but the PCA model is missing.")

    @staticmethod
    def get_model_classifier(package: Dict[str, object]):
        if "classifier" in package:
            return package["classifier"]
        if "svm" in package:
            return package["svm"]
        if "nn" in package:
            return package["nn"]
        raise ValueError("Model package does not contain a classifier.")

    def add_testing_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select testing e-nose CSV files",
            os.getcwd(),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not paths:
            return
        for path in paths:
            row = self.test_file_table.rowCount()
            self.test_file_table.insertRow(row)
            file_item = QtWidgets.QTableWidgetItem(path)
            file_item.setFlags(file_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.test_file_table.setItem(row, 0, file_item)
            self.test_file_table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
            status_item = QtWidgets.QTableWidgetItem("Waiting")
            status_item.setFlags(status_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.test_file_table.setItem(row, 2, status_item)
        self.test_file_table.resizeColumnsToContents()

    def remove_selected_testing_files(self):
        rows = sorted({idx.row() for idx in self.test_file_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.test_file_table.removeRow(row)

    def clear_testing_files(self):
        self.test_file_table.setRowCount(0)
        self.test_prediction_table_widget.clear()
        self.test_prediction_table_widget.setRowCount(0)
        self.test_prediction_table_widget.setColumnCount(0)
        self.test_feature_table_widget.clear()
        self.test_feature_table_widget.setRowCount(0)
        self.test_feature_table_widget.setColumnCount(0)
        self.test_figure.clear()
        self.test_canvas.draw()
        self.testing_feature_table = None
        self.testing_prediction_table = None
        self.export_test_predictions_btn.setEnabled(False)
        self.test_summary_text.setText("Testing file list cleared.")

    def testing_file_records_from_table(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        for row in range(self.test_file_table.rowCount()):
            file_item = self.test_file_table.item(row, 0)
            label_item = self.test_file_table.item(row, 1)
            if file_item is None:
                continue
            path = file_item.text().strip()
            label_text = label_item.text().strip() if label_item is not None else ""
            label_value: object = np.nan
            if label_text:
                try:
                    label_value = int(float(label_text))
                except Exception:
                    label_value = label_text
            records.append({"row": row, "FilePath": path, "TrueLabel": label_value, "LabelText": label_text})
        return records

    def set_testing_row_status(self, row: int, status: str):
        item = self.test_file_table.item(row, 2)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.test_file_table.setItem(row, 2, item)
        item.setText(status)
        QtWidgets.QApplication.processEvents()

    def model_settings(self, package: Dict[str, object]) -> ProcessingSettings:
        raw = package.get("settings", {})
        if not isinstance(raw, dict):
            raise ValueError("Model package does not contain a valid settings dictionary.")
        valid_names = set(ProcessingSettings.__dataclass_fields__.keys())
        kwargs = {key: raw[key] for key in raw.keys() if key in valid_names}
        return ProcessingSettings(**kwargs)

    def build_testing_feature_table(self, records: List[Dict[str, object]], settings: ProcessingSettings) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rows: List[pd.Series] = []
        pair_rows: List[pd.DataFrame] = []
        for i, rec in enumerate(records):
            csv_path = str(rec["FilePath"])
            row_index = int(rec["row"])
            self.set_testing_row_status(row_index, "Processing")
            if not os.path.isfile(csv_path):
                self.set_testing_row_status(row_index, "File not found")
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            pair_df, _ = extract_pair_features_from_csv(csv_path, settings)
            selected, selected_desc = represent_file_features(pair_df, settings)
            selected = selected.copy()
            selected["DataType"] = "test"
            selected["FilePath"] = csv_path
            selected["TrueLabel"] = rec["TrueLabel"]
            selected["NumPairs"] = int(len(pair_df))
            selected["PairSelection"] = selected_desc
            rows.append(selected)

            temp_pair_df = pair_df.copy()
            temp_pair_df.insert(0, "DataType", "test")
            temp_pair_df.insert(1, "FilePath", csv_path)
            temp_pair_df.insert(2, "TrueLabel", rec["TrueLabel"])
            pair_rows.append(temp_pair_df)
            self.set_testing_row_status(row_index, "Feature extracted")

        feature_table = pd.DataFrame(rows)
        pair_table = pd.concat(pair_rows, ignore_index=True, sort=False) if pair_rows else pd.DataFrame()
        meta_cols = ["DataType", "FilePath", "TrueLabel", "NumPairs", "PairSelection", "PairNo", "A_Start", "A_End", "M_Start", "M_End", "M_NumSamples"]
        cols = [c for c in meta_cols if c in feature_table.columns] + [c for c in feature_table.columns if c not in meta_cols]
        return feature_table[cols], pair_table

    def transform_features_with_loaded_model(self, feature_table: pd.DataFrame, package: Dict[str, object]) -> np.ndarray:
        feature_columns = list(package["feature_columns"])
        medians = package.get("medians", {})
        if isinstance(medians, pd.Series):
            med_series = medians.copy()
        elif isinstance(medians, dict):
            med_series = pd.Series(medians, dtype=float)
        else:
            med_series = pd.Series(dtype=float)

        X = pd.DataFrame(index=feature_table.index)
        for col in feature_columns:
            if col in feature_table.columns:
                X[col] = pd.to_numeric(feature_table[col], errors="coerce")
            else:
                X[col] = np.nan
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in feature_columns:
            fill_value = med_series.get(col, 0.0)
            try:
                fill_value = float(fill_value)
            except Exception:
                fill_value = 0.0
            X[col] = X[col].fillna(fill_value)
        X = X.fillna(0.0)
        X_scaled = package["scaler"].transform(X.to_numpy(dtype=float))
        apply_pca = bool(package.get("apply_pca", True))
        if apply_pca and package.get("pca", None) is not None:
            scores = package["pca"].transform(X_scaled)
            n_pc = int(package.get("n_reduced_features", scores.shape[1]))
            return scores[:, :n_pc]
        return X_scaled

    def classify_testing_files(self):
        try:
            if self.loaded_model_package is None:
                QtWidgets.QMessageBox.warning(self, "No model", "Please load an exported classification model first.")
                return
            records = self.testing_file_records_from_table()
            if not records:
                QtWidgets.QMessageBox.warning(self, "No testing files", "Please add one or more testing CSV files.")
                return

            package = self.loaded_model_package
            self.validate_model_package(package)
            settings = self.model_settings(package)
            feature_table, pair_table = self.build_testing_feature_table(records, settings)
            scores = self.transform_features_with_loaded_model(feature_table, package)
            classifier = self.get_model_classifier(package)
            pred = classifier.predict(scores)

            result = feature_table[["FilePath", "TrueLabel", "NumPairs", "PairSelection"]].copy()
            result["PredictedLabel"] = pred
            has_label = result["TrueLabel"].apply(lambda v: not (pd.isna(v) if not isinstance(v, str) else v == ""))
            result["Correct"] = ""
            if has_label.any():
                result.loc[has_label, "Correct"] = (
                    result.loc[has_label, "TrueLabel"].astype(str).to_numpy()
                    == result.loc[has_label, "PredictedLabel"].astype(str).to_numpy()
                )

            if hasattr(classifier, "predict_proba"):
                try:
                    proba = classifier.predict_proba(scores)
                    result["Confidence"] = np.max(proba, axis=1)
                except Exception:
                    pass

            score_prefix = "PC" if bool(package.get("apply_pca", True)) and package.get("pca", None) is not None else "Z"
            for i in range(scores.shape[1]):
                result[f"{score_prefix}{i+1}"] = scores[:, i]

            for rec in records:
                self.set_testing_row_status(int(rec["row"]), "Classified")

            self.testing_feature_table = feature_table
            self.testing_prediction_table = result
            self.show_dataframe(result, self.test_prediction_table_widget, max_rows=500, max_cols=100)
            self.show_dataframe(feature_table, self.test_feature_table_widget, max_rows=500, max_cols=120)
            self.export_test_predictions_btn.setEnabled(True)
            self.plot_testing_scores(scores, result["PredictedLabel"].to_numpy(), package)
            self.summarize_testing_results(result)
        except Exception as exc:
            tb = traceback.format_exc()
            self.test_summary_text.setText(tb)
            QtWidgets.QMessageBox.critical(self, "Testing classification error", str(exc))

    def summarize_testing_results(self, result: pd.DataFrame):
        lines: List[str] = []
        lines.append("Testing classification finished.\n")
        lines.append(f"Testing files: {len(result)}")
        lines.append("Predicted class counts:")
        counts = result["PredictedLabel"].astype(str).value_counts().sort_index()
        for label, count in counts.items():
            lines.append(f"  Class {label}: {count}")

        has_label = result["TrueLabel"].apply(lambda v: not (pd.isna(v) if not isinstance(v, str) else v == ""))
        if has_label.any():
            true = result.loc[has_label, "TrueLabel"].astype(str)
            pred = result.loc[has_label, "PredictedLabel"].astype(str)
            acc = accuracy_score(true, pred) * 100.0
            lines.append(f"\nAccuracy on files with entered true label: {acc:.2f}%")
            lines.append("\nClassification report:")
            lines.append(classification_report(true, pred, zero_division=0))
            labels = sorted(set(true) | set(pred))
            cm = confusion_matrix(true, pred, labels=labels)
            cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
            lines.append("Confusion matrix:")
            lines.append(cm_df.to_string())
        else:
            lines.append("\nNo true labels were entered, so accuracy was not computed.")

        self.test_summary_text.setText("\n".join(lines))

    def plot_testing_scores(self, scores: np.ndarray, pred_labels: np.ndarray, package: Dict[str, object]):
        self.test_figure.clear()
        n_pc = scores.shape[1]
        explained = package.get("explained_variance_ratio", []) or []
        if n_pc not in (2, 3):
            ax = self.test_figure.add_subplot(111)
            feature_name = "PCA" if bool(package.get("apply_pca", True)) and package.get("pca", None) is not None else "model-input"
            ax.text(0.5, 0.5, f"Testing {feature_name} plot is shown only for 2D or 3D. Current dimension = {n_pc}.",
                    ha="center", va="center")
            ax.set_axis_off()
            self.test_canvas.draw()
            return
        labels = np.unique(pred_labels)
        colors = plt_colors(len(labels))
        label_to_color = {lab: colors[i % len(colors)] for i, lab in enumerate(labels)}
        if n_pc == 2:
            ax = self.test_figure.add_subplot(111)
        else:
            ax = self.test_figure.add_subplot(111, projection="3d")
        for lab in labels:
            idx = np.asarray(pred_labels) == lab
            if n_pc == 2:
                ax.scatter(scores[idx, 0], scores[idx, 1], s=55, color=label_to_color[lab], label=f"Pred {lab}", alpha=0.9)
            else:
                ax.scatter(scores[idx, 0], scores[idx, 1], scores[idx, 2], s=55, color=label_to_color[lab], label=f"Pred {lab}", alpha=0.9)
        def pc_label(i: int) -> str:
            if bool(package.get("apply_pca", True)) and package.get("pca", None) is not None:
                if i < len(explained):
                    return f"PC{i+1} ({100*float(explained[i]):.1f}%)"
                return f"PC{i+1}"
            return f"Model feature {i+1}"
        ax.set_xlabel(pc_label(0))
        if n_pc >= 2:
            ax.set_ylabel(pc_label(1))
        if n_pc == 3:
            ax.set_zlabel(pc_label(2))
        ax.set_title("Testing PCA scores colored by predicted class")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        self.test_figure.tight_layout()
        self.test_canvas.draw()

    def export_testing_predictions(self):
        if self.testing_prediction_table is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export testing predictions", os.getcwd(), "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                self.testing_prediction_table.to_csv(path, index=False)
            else:
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    self.testing_prediction_table.to_excel(writer, sheet_name="testing_predictions", index=False)
                    if self.testing_feature_table is not None:
                        self.testing_feature_table.to_excel(writer, sheet_name="testing_file_features", index=False)
            self.test_summary_text.append(f"\nTesting results exported: {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export testing results error", str(exc))

    def show_dataframe(self, df: pd.DataFrame, table: QtWidgets.QTableWidget, max_rows: int = 200, max_cols: int = 120):
        preview = df.iloc[:max_rows, :max_cols].copy()
        table.clear()
        table.setRowCount(len(preview))
        table.setColumnCount(len(preview.columns))
        table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for r in range(len(preview)):
            for c, col in enumerate(preview.columns):
                value = preview.iloc[r, c]
                if isinstance(value, float) or isinstance(value, np.floating):
                    text = "" if not np.isfinite(value) else f"{value:.6g}"
                else:
                    text = str(value)
                table.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        table.resizeColumnsToContents()


def plt_colors(n: int):
    # Matplotlib default tab colors without importing pyplot.
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]
    return base[:max(n, 1)]



    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.disconnect_from_esp32()
        except Exception:
            pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("E-Nose Batch PCA SVM NN")
    win = ENoseBatchPCASVMApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
