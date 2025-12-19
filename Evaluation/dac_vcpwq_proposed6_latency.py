#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate & Compare:
  • Proposed AR+RVQ models from a sweep (auto-detect rvqB{books}_K{embed}/best.pth)
  • DAC 24kHz pretrained at multiple n_quantizers (rate scalable)
  • VC-PWQ (using decoded and compressed files from VC-PWQ codec)

Metrics (for DAC / Proposed / VC-PWQ):
  - ST-SIM (mel-cosine in [0,1]) at 24 kHz
  - PSNR_global_raw (dB):
        * raw tactile at original SR
        * per-file scale for DAC/Proposed: ref/scale -> encode/decode -> * scale
        * align by cross-correlation at original SR
        * PSNR = 10 log10(peak_global^2 / MSE(ref, est))

Bitrate:
  - Proposed: kbps = tokens/sec (24k DAC encoder) * books_used * log2(embed) / 1000
  - DAC 24k:  kbps = tokens/sec * n_q * log2(codebook_size) / 1000
  - VC-PWQ:   kbps = total_compressed_bits / total_signal_duration / 1000

Compression ratio:
  - DAC / Proposed: CR = 48 / kbps  (vs 3 kHz 16-bit PCM, 48 kbps)
  - VC-PWQ:        CR = total_orig_bytes / total_comp_bytes (empirical)

Latency (this script adds):
  - Buffer delay (ms):
        * DAC / Proposed:  buffer_ms = 1000 / tokens_per_second (measured)
        * VC-PWQ:          buffer_ms = 512 / 2.8kHz ≈ 182.9 ms (from paper)
  - Encoding delay (ms):
        * DAC:             measured by timing encode() on 1 s dummy audio
        * Proposed:        measured by timing encode_latents() on 1 s dummy audio+tactile
        * VC-PWQ:          not reported in paper → NaN
  - Decoding delay (ms):
        * DAC:             measured by timing decode()
        * Proposed:        measured by timing T_DEC() from latent codes
        * VC-PWQ:          not reported in paper → NaN

Outputs:
  - combined JSON summary (DAC, Proposed, VC-PWQ) including latency
  - Plots (all 3 models):
      • STSIM vs Bitrate
      • PSNR vs Bitrate
      • STSIM vs Compression Ratio
      • PSNR vs Compression Ratio
      • Compression Ratio vs Bitrate
      • 10 example tactile waveforms (original vs reconstructed, proposed)
"""

import os, json, glob, math, random, warnings, time
from pathlib import Path
warnings.filterwarnings("once", category=UserWarning)

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import dac  # pip install descript-audio-codec

# ===================== USER CONFIG =====================

# AUDIO_DIR: audio modality (original SR, resampled to EVAL_SR internally)
# TACT_DIR:  vibrotactile modality (original ~3 kHz signals)
AUDIO_DIR   = r"/home/student/studentdata/WAV_Files_raw"
TACT_DIR    = r"/home/student/studentdata/Vibrotactile_Files_Raw"

# Root of your training sweep (as produced by the sweep script)
SWEEP_ROOT  = r"/home/student/studentdata/SWEEP_ALLPRED_AR_RVQ"

OUT_DIR     = os.path.join(SWEEP_ROOT, "eval_vs_dac24_with_vcpwq_rawPSNR_latency")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_SR  = 24000
ORIG_3K  = 3000  # used only for baseline 48 kbps
SEED     = 7
USE_AMP  = torch.cuda.is_available()

def _sync():
    """Synchronize CUDA (if available) so timing is correct."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# DAC 24k settings
DAC_MODEL_TYPE = "24khz"
DAC_NQ_LIST    = [1, 2, 3, 4, 8]

# 3 kHz mono 16-bit PCM baseline
PCM_KBPS_TACT_ORIG = ORIG_3K * 16.0 / 1000.0  # = 48 kbps

# Alignment (± samples @ original tactile SR)
MAX_ALIGN_SHIFT = 200

# Plot Y-range for ST-SIM
Y_STSIM_MIN, Y_STSIM_MAX = 0.80, 1.00

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ===================== VC-PWQ CONFIG =====================
VC_DEC_DIR  = r"/home/student/studentdata/dac_eval_3khz_fixed/VC-PWQ/build/source/testprogram/data_decoded"
VC_COMP_DIR = r"/home/student/studentdata/dac_eval_3khz_fixed/VC-PWQ/build/source/testprogram/data_compressed"

VC_CONFIGS = [
    {"label": "VC bl512 b8",   "b": 8},
    {"label": "VC bl512 b12",  "b": 12},
    {"label": "VC bl512 b16",  "b": 16},
    {"label": "VC bl512 b20",  "b": 20},
    {"label": "VC bl512 b24",  "b": 24},
    {"label": "VC bl512 b48",  "b": 48}
]

# VC-PWQ buffer delay from paper:
# block length = 512, sampling rate = 2.8 kHz ⇒ 512 / 2800 s ≈ 182.9 ms
VC_FS_STD_HZ   = 2800.0
VC_BLOCK_LEN   = 512
VC_BUFFER_MS   = 1000.0 * VC_BLOCK_LEN / VC_FS_STD_HZ  # ≈ 182.9 ms

# ===================== BASIC IO & HELPERS =====================

def list_wavs(dirpath):
    return {Path(p).stem: p for p in glob.glob(os.path.join(dirpath, "*.wav"))}

def list_pairs(audio_dir, tact_dir):
    A = list_wavs(audio_dir)
    T = list_wavs(tact_dir)
    stems = sorted(set(A) & set(T))
    pairs = [(A[s], T[s], s) for s in stems]
    print(f"[Pairs] Found {len(pairs)} audio/tactile pairs.")
    return pairs

def load_wav_raw(path):
    """
    Load WAV without extra normalization (sf scales PCM_16 to [-1,1]).
    Returns: tensor [1,T], sr
    """
    data, sr = sf.read(path, always_2d=True)
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    w = torch.from_numpy(data).t().contiguous()[:1, :]  # [1,T]
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return w, int(sr)

@torch.no_grad()
def resample_to(x_1T, sr_in, sr_out):
    if sr_in == sr_out:
        return x_1T
    x_1T = x_1T.to(torch.float32)
    res = torchaudio.transforms.Resample(orig_freq=sr_in, new_freq=sr_out).to(x_1T.device)
    return res(x_1T.contiguous())

def crop_match(a_1T, b_1T):
    T = min(a_1T.shape[-1], b_1T.shape[-1])
    return a_1T[..., :T], b_1T[..., :T]

# ===================== ALIGNMENT + PSNR (GLOBAL PEAK) =====================

def align_by_xcorr(ref_1T, est_1T, max_shift=MAX_ALIGN_SHIFT):
    """
    Align est to ref by maximizing cross-correlation over integer shifts
    at the ORIGINAL tactile sample rate.
    Inputs: ref_1T, est_1T: [1,T]
    Returns: ref_aligned[1,T'], est_aligned[1,T'], best_shift
    """
    r = ref_1T.squeeze(0).to(torch.float32)
    e = est_1T.squeeze(0).to(torch.float32)
    best_s, best_corr = 0, -1e18

    for s in range(-max_shift, max_shift + 1):
        if s < 0:
            r_seg = r[-s:]
            e_seg = e[: r_seg.numel()]
        elif s > 0:
            r_seg = r[:-s]
            e_seg = e[s : s + r_seg.numel()]
        else:
            r_seg = r
            e_seg = e[: r_seg.numel()]
        if r_seg.numel() == 0 or e_seg.numel() == 0:
            continue
        c = torch.sum(r_seg * e_seg)
        if c > best_corr:
            best_corr, best_s = c, s

    s = best_s
    if s < 0:
        r_a = r[-s:]
        e_a = e[: r_a.numel()]
    elif s > 0:
        r_a = r[:-s]
        e_a = e[s : s + r_a.numel()]
    else:
        r_a = r
        e_a = e[: r.numel()]

    return r_a.unsqueeze(0), e_a.unsqueeze(0), best_s

def psnr_global_peak_db(ref, est, peak, eps=1e-12):
    """
    PSNR using GLOBAL peak (same for all files):
        PSNR = 10 log10(peak^2 / MSE(ref, est))
    ref, est: [1,T]
    """
    ref = ref.reshape(-1).to(torch.float32)
    est = est.reshape(-1).to(torch.float32)
    mse = torch.mean((ref - est) ** 2) + eps
    peak = max(float(peak), eps)
    return float(10.0 * torch.log10((peak * peak) / mse).cpu())

def compute_global_peak(wav_paths):
    """
    Max abs amplitude over ALL tactile wav files.
    """
    max_val = 0.0
    for p in wav_paths:
        w, _ = load_wav_raw(p)
        m = float(w.abs().max().cpu())
        if m > max_val:
            max_val = m
    return max_val if max_val > 0.0 else 1.0

# ===================== ST-SIM (24 kHz) =====================

_MEL_CACHE = {}

def _mel_mag(x_1T, sr=EVAL_SR, n_fft=512, hop=128, n_mels=64):
    """
    x_1T: [B,1,T]
    """
    if x_1T.dim() == 3:
        x = x_1T[:, 0, :]
    else:
        x = x_1T
    with torch.cuda.amp.autocast(enabled=False):
        x = x.to(torch.float32)
        window = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=window, center=True, return_complex=True)
        mag = spec.abs().clamp_min_(1e-8)
        key = (x.device.type, sr, n_fft, hop, n_mels)
        mel = _MEL_CACHE.get(key)
        if mel is None:
            mel = torchaudio.transforms.MelScale(
                n_mels=n_mels, sample_rate=sr, n_stft=n_fft//2 + 1,
                f_min=0.0, f_max=sr*0.5, norm=None, mel_scale="htk"
            ).to(x.device)
            _MEL_CACHE[key] = mel
        M = mel(mag)
        M = M / M.amax(dim=(1, 2), keepdim=True).clamp_min_(1e-8)
    return M

@torch.no_grad()
def stsim_batch(ref_1T, est_1T):
    """
    ref_1T, est_1T: [B,1,T] at 24 kHz.
    """
    Mref = _mel_mag(ref_1T); Mest = _mel_mag(est_1T)
    Tf = max(Mref.shape[-1], Mest.shape[-1])
    if Mref.shape[-1] != Tf:
        Mref = F.interpolate(Mref, size=Tf, mode="linear", align_corners=False)
    if Mest.shape[-1] != Tf:
        Mest = F.interpolate(Mest, size=Tf, mode="linear", align_corners=False)
    num = (Mref * Mest).sum(dim=1)
    den = (Mref.norm(dim=1) * Mest.norm(dim=1)).clamp_min(1e-8)
    cos_t = (num / den).clamp(-1, 1)
    val = 0.5 * (cos_t.mean(dim=-1) + 1.0)  # [B]
    return [float(v.item()) for v in val]

# ===================== DAC 24k Helpers =====================

@torch.no_grad()
def probe_tokens_per_sec(dac_model, sr_target: int):
    x = torch.zeros(1, 1, sr_target, device=DEVICE, dtype=torch.float32)
    with torch.cuda.amp.autocast(enabled=False):
        z = dac_model.encoder(x)
    return float(z.shape[-1])

def get_n_books_and_bins(quantizer):
    n_books = None; bins = None
    if hasattr(quantizer, "n_q"):   n_books = int(quantizer.n_q)
    if hasattr(quantizer, "bins"):  bins    = int(quantizer.bins)
    if (n_books is not None) and (bins is not None):
        return n_books, bins
    books = 0
    for n, p in quantizer.named_parameters():
        if "codebook" in n.lower() or "embed" in n.lower():
            if p.dim() == 2:
                books += 1
                bins = p.size(0) if bins is None else max(bins, p.size(0))
    if n_books is None: n_books = books if books > 0 else 8
    if bins    is None: bins    = 1024
    return int(n_books), int(bins)

@torch.no_grad()
def measure_dac_latency(dac_model, sr_native: int, n_q: int, repeats: int = 10):
    """
    Measure average encoding and decoding latency (ms) of DAC for 1 s audio
    at sr_native, with given n_quantizers.
    """
    x = torch.zeros(1, 1, sr_native, device=DEVICE, dtype=torch.float32)

    # Warmup
    for _ in range(3):
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            z, *_ = dac_model.encode(x, n_quantizers=int(n_q))
            _ = dac_model.decode(z)
    _sync()

    enc_times = []
    dec_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            z, *_ = dac_model.encode(x, n_quantizers=int(n_q))
        _sync()
        enc_times.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            _ = dac_model.decode(z)
        _sync()
        dec_times.append((time.perf_counter() - start) * 1000.0)

    enc_ms = float(np.mean(enc_times))
    dec_ms = float(np.mean(dec_times))
    return enc_ms, dec_ms

# ===================== Proposed Model (Eval wrapper) =====================

CODE_DIM = 96
AR_CHUNK_TOK = 16

class PosEnc1D(nn.Module):
    def __init__(self, c, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, c)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, c, 2) * (-math.log(10000.0)/c))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(-1)
        return x + self.pe[:T, :].T.unsqueeze(0).to(x.dtype)

class TokenNorm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)
    def forward(self, z):
        zt = z.permute(0, 2, 1)
        zt = self.ln(zt)
        return zt.permute(0, 2, 1)

class CrossPredictor(nn.Module):
    def __init__(self, c, heads=8, mlp_mul=2, dropout=0.1):
        super().__init__()
        self.pos = PosEnc1D(c)
        self.h = heads
        self.dh = c // heads
        assert c % heads == 0
        self.ln_q = nn.LayerNorm(c)
        self.ln_kv = nn.LayerNorm(c)
        self.q_proj = nn.Linear(c, c, False)
        self.k_proj = nn.Linear(c, c, False)
        self.v_proj = nn.Linear(c, c, False)
        self.out = nn.Linear(c, c, False)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, mlp_mul * c),
            nn.GELU(),
            nn.Linear(mlp_mul * c, c)
        )

    def _split(self, x):
        B, T, C = x.shape
        return x.view(B, T, self.h, self.dh).permute(0, 2, 1, 3)

    def _merge(self, x):
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)

    def forward(self, zt_prev, za):
        q  = self.pos(zt_prev).permute(0, 2, 1)
        kv = self.pos(za).permute(0, 2, 1)
        q  = self.ln_q(q)
        kv = self.ln_kv(kv)

        Q = self._split(self.q_proj(q))
        K = self._split(self.k_proj(kv))
        V = self._split(self.v_proj(kv))

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dh)
        ctx  = (attn.softmax(dim=-1) @ V)

        y = self.out(self.drop(self._merge(ctx)))
        y = y + q
        y = y + self.ffn(y)
        return y.permute(0, 2, 1)

class ResidualVQEMA(nn.Module):
    def __init__(self, dim: int, n_books: int, n_embed: int):
        super().__init__()
        self.books = nn.ParameterList([
            nn.Parameter(torch.randn(n_embed, dim) / math.sqrt(dim))
            for _ in range(n_books)
        ])

    @staticmethod
    def _nearest_l2(x, emb):
        return (x @ emb.t() - 0.5 * (emb * emb).sum(dim=1).unsqueeze(0)).argmax(dim=1)

    def forward(self, z, n_books_use=None):
        if n_books_use is None:
            n_books_use = len(self.books)
        n_books_use = min(n_books_use, len(self.books))
        B, D, T = z.shape
        x = z.permute(0, 2, 1).reshape(B * T, D)
        residual = x
        q_sum = torch.zeros_like(x)
        for cb in self.books[:n_books_use]:
            emb = cb.detach().to(z.dtype).to(z.device)
            idx = self._nearest_l2(residual, emb)
            q   = F.embedding(idx, emb)
            q_sum = q_sum + (q - residual).detach() + residual
            residual = residual - q
        return q_sum.view(B, T, D).permute(0, 2, 1).contiguous()

class ProposedEval(nn.Module):
    def __init__(self, A_ENC, A_QUANT, T_ENC, T_DEC, c_lat, rvq_books, rvq_embed):
        super().__init__()
        self.A_ENC, self.A_QUANT, self.T_ENC, self.T_DEC = A_ENC, A_QUANT, T_ENC, T_DEC
        for m in [self.A_ENC, self.A_QUANT, self.T_ENC, self.T_DEC]:
            for p in m.parameters():
                p.requires_grad_(False)
        self.predict   = CrossPredictor(c=c_lat)
        self.tokennorm = TokenNorm(c_lat)
        self.scale     = nn.Parameter(torch.tensor(0.08))
        self.proj_down = nn.Conv1d(c_lat, CODE_DIM, 1)
        self.proj_up   = nn.Conv1d(CODE_DIM, c_lat, 1)
        self.vq        = ResidualVQEMA(dim=CODE_DIM, n_books=rvq_books, n_embed=rvq_embed)

    @torch.no_grad()
    def encode_latents(self, a_1T, t_1T, books_use=None):
        """
        Encoder part: from 24k audio+tactile to quantized latent z_run.
        a_1T, t_1T: [B,1,T] at 24 kHz, already scaled.
        """
        za = self.A_ENC(a_1T)
        qa, *_ = self.A_QUANT(za)
        zt = self.T_ENC(t_1T)
        B, C, Tlat = zt.shape
        z_run = torch.zeros_like(zt)
        for s in range(0, Tlat, AR_CHUNK_TOK):
            e = min(Tlat, s + AR_CHUNK_TOK)
            zt_prev = torch.zeros(B, C, e - s, device=zt.device, dtype=zt.dtype)
            if s == 0:
                zt_prev[..., 1:] = z_run[..., s:e-1]
            else:
                zt_prev[...]     = z_run[..., s-1:e-1]
            qa_chunk = qa[..., s:e]
            z_pred   = self.predict(zt_prev, qa_chunk)
            r        = (zt[..., s:e] - z_pred.detach())
            rN       = torch.tanh(self.tokennorm(r))
            scale    = self.scale.clamp(5e-3, 0.5)
            rD       = self.proj_down(scale * rN)
            qD       = self.vq(rD, n_books_use=books_use)
            z_hat    = self.proj_up(qD) + z_pred
            z_run[..., s:e] = z_hat
        return z_run

    @torch.no_grad()
    def forward_eval(self, a_1T, t_1T, books_use=None):
        """
        Full encode+decode: returns decoded tactile [B,1,T] at 24 kHz.
        """
        z_run = self.encode_latents(a_1T, t_1T, books_use=books_use)
        y = self.T_DEC(z_run)
        return y

@torch.no_grad()
def measure_proposed_latency(net: ProposedEval, sr_native: int, rvq_books: int, repeats: int = 10):
    """
    Measure average encoding and decoding latency (ms) of Proposed model for 1 s
    audio+tactile at sr_native.

    Encoding:  encode_latents()
    Decoding:  T_DEC() from latent codes
    """
    a = torch.zeros(1, 1, sr_native, device=DEVICE, dtype=torch.float32)
    t = torch.zeros(1, 1, sr_native, device=DEVICE, dtype=torch.float32)

    # Warmup
    for _ in range(3):
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            z_run = net.encode_latents(a, t, books_use=rvq_books)
            _ = net.T_DEC(z_run)
    _sync()

    enc_times = []
    dec_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            z_run = net.encode_latents(a, t, books_use=rvq_books)
        _sync()
        enc_times.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            _ = net.T_DEC(z_run)
        _sync()
        dec_times.append((time.perf_counter() - start) * 1000.0)

    enc_ms = float(np.mean(enc_times))
    dec_ms = float(np.mean(dec_times))
    return enc_ms, dec_ms

def build_backbones_for_eval():
    da = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    dt = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    A_ENC, A_QUANT = da.encoder, da.quantizer
    T_ENC, T_DEC   = dt.encoder, dt.decoder
    dummy = torch.zeros(1, 1, EVAL_SR, device=DEVICE)
    with torch.cuda.amp.autocast(enabled=False):
        C = A_ENC(dummy).size(1)
    return A_ENC, A_QUANT, T_ENC, T_DEC, C

# ===================== EVAL: DAC 24k (file-wise, raw PSNR) =====================

@torch.no_grad()
def eval_dac24(pairs, n_q_list, peak_global):
    mdl = dac.DAC.load(dac.utils.download(DAC_MODEL_TYPE)).to(DEVICE).eval()
    native_sr = 24000
    tps = probe_tokens_per_sec(mdl, native_sr)
    _, bins = get_n_books_and_bins(mdl.quantizer)
    bits_per_code = math.log2(bins)
    buffer_ms = 1000.0 / tps if tps > 0 else float("nan")

    out = {}
    for n_q in n_q_list:
        print(f"[DAC24] n_q={n_q}")
        st_vals, ps_vals = [], []

        # Measure latency once per n_q (1 s dummy signal)
        enc_ms, dec_ms = measure_dac_latency(mdl, sr_native=native_sr, n_q=n_q)

        for _, t_path, stem in pairs:
            # Original tactile
            ref_1T, sr_ref = load_wav_raw(t_path)    # [1,T_ref]
            ref_1T = ref_1T.to(DEVICE)

            # Per-file scale
            scale = max(float(ref_1T.abs().max().cpu()), 1e-8)

            # To 24 kHz, scaled
            x_native = resample_to(ref_1T / scale, sr_ref, native_sr).to(DEVICE)

            # Encode / decode
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                z, *_ = mdl.encode(x_native.unsqueeze(0), n_quantizers=int(n_q))
                y_native = mdl.decode(z)[0]  # [1,T24]

            # Back to original SR, restore amplitude
            est_1T = resample_to(y_native, native_sr, sr_ref).to(DEVICE) * scale

            # Crop & align at ORIGINAL SR
            ref_c, est_c = crop_match(ref_1T, est_1T)
            ref_a, est_a, _ = align_by_xcorr(ref_c, est_c, MAX_ALIGN_SHIFT)
            ref_a, est_a = crop_match(ref_a, est_a)

            # PSNR with GLOBAL PEAK
            psnr = psnr_global_peak_db(ref_a, est_a, peak_global)
            ps_vals.append(psnr)

            # ST-SIM: resample aligned signals to 24 kHz
            ref_24 = resample_to(ref_a, sr_ref, EVAL_SR).to(DEVICE)
            est_24 = resample_to(est_a, sr_ref, EVAL_SR).to(DEVICE)
            ref_24b = ref_24.unsqueeze(0)  # [1,1,T]
            est_24b = est_24.unsqueeze(0)
            st = stsim_batch(ref_24b, est_24b)[0]
            st_vals.append(st)

        arr_s = np.array(st_vals, dtype=np.float64)
        arr_p = np.array(ps_vals, dtype=np.float64)
        n = int(arr_s.size)

        st_m = float(arr_s.mean()); st_ci = 1.96 * float(arr_s.std(ddof=0)) / max(1, math.sqrt(n))
        ps_m = float(arr_p.mean()); ps_ci = 1.96 * float(arr_p.std(ddof=0)) / max(1, math.sqrt(n))

        kbps = (tps * n_q * bits_per_code) / 1000.0
        cr   = PCM_KBPS_TACT_ORIG / kbps if kbps > 0 else float('inf')

        out[int(n_q)] = {
            "stsim_mean": st_m, "stsim_ci95": st_ci,
            "psnr_mean":  ps_m, "psnr_ci95":  ps_ci,
            "kbps": float(kbps), "compression_ratio": float(cr),
            "n": n, "tps": float(tps), "bins": int(bins),
            "encoding_delay_ms": float(enc_ms),
            "decoding_delay_ms": float(dec_ms),
            "buffer_delay_ms": float(buffer_ms),
        }

    return out

# ===================== EVAL: Proposed runs (file-wise, raw PSNR) =====================

@torch.no_grad()
def eval_proposed_runs(pairs, sweep_root: str, peak_global: float):
    # discover runs like rvqB{books}_K{embed}
    candidates = sorted(
        [p for p in glob.glob(os.path.join(sweep_root, "rvqB*_K*")) if os.path.isdir(p)]
    )
    if not candidates:
        raise RuntimeError(f"No runs found under {sweep_root} (expected rvqB*_K*/).")

    A_ENC, A_QUANT, T_ENC, T_DEC, C = build_backbones_for_eval()

    da24 = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    tps = probe_tokens_per_sec(da24, EVAL_SR)
    buffer_ms = 1000.0 / tps if tps > 0 else float("nan")

    results = []
    for run in candidates:
        meta_path = os.path.join(run, "meta.json")
        best_path = os.path.join(run, "best.pth")
        if not os.path.isfile(best_path):
            print(f"[Skip] no best.pth in {run}")
            continue

        rvq_books = None; rvq_embed = None
        stem = Path(run).name

        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                rvq_books = int(meta.get("rvq_books", 0)) or None
                rvq_embed = int(meta.get("rvq_embed", 0)) or None
            except Exception:
                pass

        if rvq_books is None or rvq_embed is None:
            try:
                partB = stem.split("_")[0]
                partK = stem.split("_")[1]
                rvq_books = int(partB.replace("rvqB", ""))
                rvq_embed = int(partK.replace("K", ""))
            except Exception:
                raise RuntimeError(f"Cannot infer (books, embed) for run: {run}")

        print(f"[Proposed] {stem} | books={rvq_books}, embed={rvq_embed}")
        net = ProposedEval(A_ENC, A_QUANT, T_ENC, T_DEC, c_lat=C,
                           rvq_books=rvq_books, rvq_embed=rvq_embed).to(DEVICE)
        sd = torch.load(best_path, map_location="cpu")
        missing, unexpected = net.load_state_dict(sd["model"], strict=False)
        if missing or unexpected:
            print(f"  (state-dict mismatch tolerated) missing={len(missing)} unexpected={len(unexpected)}")
        net.eval()

        # Measure latency once per run (1 s dummy audio+tactile)
        enc_ms, dec_ms = measure_proposed_latency(net, sr_native=EVAL_SR, rvq_books=rvq_books)

        st_vals, ps_vals = [], []

        for a_path, t_path, pair_stem in pairs:
            # raw audio & tactile
            a_raw, sr_a = load_wav_raw(a_path)   # [1,Ta]
            t_raw, sr_t = load_wav_raw(t_path)   # [1,Tt]
            a_raw = a_raw.to(DEVICE)
            t_raw = t_raw.to(DEVICE)

            # per-file scale from tactile
            scale = max(float(t_raw.abs().max().cpu()), 1e-8)

            # to 24k, scaled
            a_24 = resample_to(a_raw / scale, sr_a, EVAL_SR).to(DEVICE)
            t_24 = resample_to(t_raw / scale, sr_t, EVAL_SR).to(DEVICE)
            a_24b = a_24.unsqueeze(0)  # [1,1,T]
            t_24b = t_24.unsqueeze(0)

            # forward
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                y_scaled_24b = net.forward_eval(a_24b, t_24b, books_use=rvq_books)  # [1,1,T]

            # restore amplitude
            y_24b = y_scaled_24b * scale
            y_24  = y_24b.squeeze(0)  # [1,T]

            # back to original tactile SR
            est_1T = resample_to(y_24, EVAL_SR, sr_t).to(DEVICE)

            # crop & align at ORIGINAL SR
            ref_c, est_c = crop_match(t_raw, est_1T)
            ref_a, est_a, _ = align_by_xcorr(ref_c, est_c, MAX_ALIGN_SHIFT)
            ref_a, est_a = crop_match(ref_a, est_a)

            # PSNR with GLOBAL PEAK
            psnr = psnr_global_peak_db(ref_a, est_a, peak_global)
            ps_vals.append(psnr)

            # ST-SIM on aligned signals resampled to 24k
            ref_24 = resample_to(ref_a, sr_t, EVAL_SR).to(DEVICE)
            est_24 = resample_to(est_a, sr_t, EVAL_SR).to(DEVICE)
            ref_24b = ref_24.unsqueeze(0)
            est_24b = est_24.unsqueeze(0)
            st = stsim_batch(ref_24b, est_24b)[0]
            st_vals.append(st)

        arr_s = np.array(st_vals, dtype=np.float64)
        arr_p = np.array(ps_vals, dtype=np.float64)
        n     = int(arr_s.size)

        st_m, st_ci = float(arr_s.mean()), 1.96 * float(arr_s.std(ddof=0)) / max(1, math.sqrt(n))
        ps_m, ps_ci = float(arr_p.mean()), 1.96 * float(arr_p.std(ddof=0)) / max(1, math.sqrt(n))

        bits_per_code = math.log2(rvq_embed)
        kbps = (tps * rvq_books * bits_per_code) / 1000.0
        cr   = PCM_KBPS_TACT_ORIG / kbps if kbps > 0 else float('inf')

        results.append({
            "run": stem, "path": run,
            "books": int(rvq_books), "embed": int(rvq_embed),
            "bits_per_code": float(bits_per_code),
            "tps": float(tps),
            "kbps": float(kbps), "compression_ratio": float(cr),
            "n": n,
            "stsim_mean": st_m, "stsim_ci95": st_ci,
            "psnr_mean": ps_m, "psnr_ci95": ps_ci,
            "encoding_delay_ms": float(enc_ms),
            "decoding_delay_ms": float(dec_ms),
            "buffer_delay_ms": float(buffer_ms),
        })

    return results

# ===================== VC-PWQ EVAL (raw PSNR logic) =====================

@torch.no_grad()
def eval_vc_pwq(vc_dec_dir, vc_comp_dir, tact_dir, vc_configs, peak_global):
    """
    Evaluate VC-PWQ on tactile files using the SAME PSNR logic as DAC/Proposed:
      - raw tactile vs VC-decoded
      - align_by_xcorr at original SR
      - PSNR_global_peak_db with shared GLOBAL peak.

    ST-SIM is computed on aligned waveforms resampled to 24 kHz.

    Latency:
      - buffer_delay_ms fixed from paper (182.9 ms for block length 512 @ 2.8 kHz)
      - encoding/decoding delay not reported → NaN
    """
    results = []

    for cfg in vc_configs:
        label = cfg["label"]
        b_val = int(cfg["b"])
        print(f"[VC-PWQ] Evaluating {label} (b={b_val})")

        st_vals, ps_vals = [], []
        total_comp_bytes = 0
        total_orig_bytes = 0
        total_time_sec   = 0.0
        used_pairs       = 0

        tact_paths = sorted(glob.glob(os.path.join(tact_dir, "*.wav")))
        for t_path in tact_paths:
            t_name = Path(t_path).stem

            # decoded VC-PWQ
            dec_pattern = os.path.join(vc_dec_dir, f"*{t_name}*_{b_val}.wav")
            dec_candidates = glob.glob(dec_pattern)
            if not dec_candidates:
                continue
            dec_path = sorted(dec_candidates)[0]

            # original tactile and decoded at ORIGINAL SR
            ref_1T, sr_ref = load_wav_raw(t_path)
            est_1T, sr_dec = load_wav_raw(dec_path)
            ref_1T = ref_1T.to(DEVICE)
            est_1T = est_1T.to(DEVICE)

            est_res = resample_to(est_1T, sr_dec, sr_ref)

            # crop & align at ORIGINAL SR
            ref_c, est_c = crop_match(ref_1T, est_res)
            ref_a, est_a, _ = align_by_xcorr(ref_c, est_c, MAX_ALIGN_SHIFT)
            ref_a, est_a = crop_match(ref_a, est_a)

            # PSNR with GLOBAL PEAK
            psnr = psnr_global_peak_db(ref_a, est_a, peak_global)
            ps_vals.append(psnr)

            # ST-SIM at 24k on aligned signals
            ref_24 = resample_to(ref_a, sr_ref, EVAL_SR).to(DEVICE)
            est_24 = resample_to(est_a, sr_ref, EVAL_SR).to(DEVICE)
            ref_24b = ref_24.unsqueeze(0)
            est_24b = est_24.unsqueeze(0)
            st = stsim_batch(ref_24b, est_24b)[0]
            st_vals.append(st)

            # compressed file size
            comp_pattern = os.path.join(vc_comp_dir, f"*{t_name}*_{b_val}.binary")
            comp_candidates = glob.glob(comp_pattern)
            if comp_candidates:
                comp_path = sorted(comp_candidates)[0]
                comp_bytes = os.path.getsize(comp_path)
                total_comp_bytes += comp_bytes

                # original size
                orig_bytes = os.path.getsize(t_path)
                total_orig_bytes += orig_bytes

                # duration from original tactile
                data_t, sr_t2 = sf.read(t_path)
                n_samples = data_t.shape[0] if data_t.ndim > 1 else len(data_t)
                duration = n_samples / float(sr_t2)
                total_time_sec += duration

            used_pairs += 1

        if used_pairs == 0:
            print(f"  [VC-PWQ] WARNING: no matching pairs found for b={b_val}")
            continue

        arr_s = np.array(st_vals, dtype=np.float64)
        arr_p = np.array(ps_vals, dtype=np.float64)
        n = int(arr_s.size)

        st_m = float(arr_s.mean())
        st_ci = 1.96 * float(arr_s.std(ddof=0)) / max(1, math.sqrt(n))
        ps_m = float(arr_p.mean())
        ps_ci = 1.96 * float(arr_p.std(ddof=0)) / max(1, math.sqrt(n))

        if total_comp_bytes > 0 and total_time_sec > 0:
            total_comp_bits = total_comp_bytes * 8.0
            bitrate_bps = total_comp_bits / total_time_sec
            kbps = bitrate_bps / 1000.0
            cr = (total_orig_bytes * 1.0) / (total_comp_bytes * 1.0)
        else:
            kbps = float("nan")
            cr   = float("nan")

        print(f"  pairs={used_pairs}, nseg={n}")
        print(f"  STSIM_mean={st_m:.4f}, STSIM_CI={st_ci:.4f}")
        print(f"  PSNR_mean ={ps_m:.2f} dB, PSNR_CI={ps_ci:.2f}")
        print(f"  kbps={kbps:.3f}, CR={cr:.2f}x")

        # Encoding/decoding delay not reported in the paper → NaN
        enc_ms = float("nan")
        dec_ms = float("nan")
        buffer_ms = float(VC_BUFFER_MS)

        results.append({
            "label": label,
            "b": int(b_val),
            "kbps": float(kbps),
            "compression_ratio": float(cr),
            "stsim_mean": st_m,
            "stsim_ci95": st_ci,
            "psnr_mean": ps_m,
            "psnr_ci95": ps_ci,
            "n": int(n),
            "pairs": int(used_pairs),
            "encoding_delay_ms": enc_ms,
            "decoding_delay_ms": dec_ms,
            "buffer_delay_ms": buffer_ms,
        })

    return results

# ===================== PLOTTING HELPERS =====================

def _group_by_embed(rows):
    groups = {}
    for r in rows:
        k = int(r["embed"])
        groups.setdefault(k, []).append(r)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda x: x["kbps"])
    return groups

def _errfill(x, y, ci, label, marker, color):
    x = np.asarray(x); y = np.asarray(y); ci = np.asarray(ci)
    plt.plot(x, y, marker + "-", lw=2.0, ms=0, label=label, color=color)
    plt.scatter(x, y, s=36, zorder=3, color=color)
    plt.fill_between(x, y-ci, y+ci, alpha=0.20, color=color)

@torch.no_grad()
def plot_proposed_examples(best_run, pairs, num_examples, out_dir):
    """
    Plot `num_examples` examples of original vs reconstructed tactile signals
    for the given best proposed run.

    For each example we save TWO figures:
      - proposed_example_XX.png        (waveform, original vs reconstructed)
      - proposed_example_XX_mel.png    (log-mel spectrograms, original vs reconstructed)
    """
    run_path   = best_run["path"]
    run_name   = best_run["run"]
    rvq_books  = int(best_run["books"])
    rvq_embed  = int(best_run["embed"])
    best_ckpt  = os.path.join(run_path, "best.pth")

    print(f"[Examples] Using proposed run '{run_name}' "
          f"(books={rvq_books}, embed={rvq_embed}) for waveform + mel plots.")

    # Build backbones and proposed model
    A_ENC, A_QUANT, T_ENC, T_DEC, C = build_backbones_for_eval()
    net = ProposedEval(A_ENC, A_QUANT, T_ENC, T_DEC,
                       c_lat=C, rvq_books=rvq_books, rvq_embed=rvq_embed).to(DEVICE)
    sd = torch.load(best_ckpt, map_location="cpu")
    net.load_state_dict(sd["model"], strict=False)
    net.eval()

    num_examples = min(num_examples, len(pairs))

    for idx in range(num_examples):
        a_path, t_path, stem = pairs[idx]

        # raw audio & tactile
        a_raw, sr_a = load_wav_raw(a_path)
        t_raw, sr_t = load_wav_raw(t_path)
        a_raw = a_raw.to(DEVICE)
        t_raw = t_raw.to(DEVICE)

        # per-file scale based on tactile
        scale = max(float(t_raw.abs().max().cpu()), 1e-8)

        # to 24k, scaled
        a_24 = resample_to(a_raw / scale, sr_a, EVAL_SR).to(DEVICE)
        t_24 = resample_to(t_raw / scale, sr_t, EVAL_SR).to(DEVICE)
        a_24b = a_24.unsqueeze(0)
        t_24b = t_24.unsqueeze(0)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            y_scaled_24b = net.forward_eval(a_24b, t_24b, books_use=rvq_books)

        # restore amplitude and go back to original tactile SR
        y_24b = y_scaled_24b * scale
        y_24  = y_24b.squeeze(0)
        est_1T = resample_to(y_24, EVAL_SR, sr_t).to(DEVICE)

        # align and crop
        ref_c, est_c = crop_match(t_raw, est_1T)
        ref_a, est_a, _ = align_by_xcorr(ref_c, est_c, MAX_ALIGN_SHIFT)
        ref_a, est_a = crop_match(ref_a, est_a)

        # ---------- WAVEFORM PLOT (per example) ----------
        ref_np = ref_a.squeeze(0).cpu().numpy()
        est_np = est_a.squeeze(0).cpu().numpy()
        T = ref_np.shape[-1]
        t_axis = np.arange(T) / float(sr_t)

        fig_w, ax_w = plt.subplots(figsize=(8, 3))
        ax_w.plot(t_axis, ref_np, label="Original", linewidth=1.0)
        ax_w.plot(t_axis, est_np, label="Reconstructed", linewidth=1.0, alpha=0.8)
        ax_w.set_ylabel("Amplitude")
        ax_w.set_xlabel("Time (s)")
        ax_w.set_title(f" {stem}")
        ax_w.grid(True, alpha=0.3)
        ax_w.legend(loc="upper right", fontsize=8)

        fig_w.tight_layout()
        out_wave = os.path.join(out_dir, f"proposed_example_{idx+1:02d}.png")
        fig_w.savefig(out_wave, dpi=180)
        plt.close(fig_w)
        print(f"Saved waveform example {idx+1} → {out_wave}")

        # ---------- LOG-MEL SPECTROGRAM PLOT (per example) ----------
        # Resample aligned signals to 24 kHz for mel computation
        ref_24 = resample_to(ref_a, sr_t, EVAL_SR).to(DEVICE)
        est_24 = resample_to(est_a, sr_t, EVAL_SR).to(DEVICE)

        # Use same mel front-end as ST-SIM; returns [B, n_mels, T_frames]
        M_ref = _mel_mag(ref_24.unsqueeze(0))  # [1, n_mels, T]
        M_est = _mel_mag(est_24.unsqueeze(0))

        ref_mel = M_ref.squeeze(0).cpu().numpy()
        est_mel = M_est.squeeze(0).cpu().numpy()

        # Log scale (dB)
        ref_log = 20.0 * np.log10(ref_mel + 1e-8)
        est_log = 20.0 * np.log10(est_mel + 1e-8)

        n_frames = ref_log.shape[-1]
        hop = 128  # must match _mel_mag
        t_axis_mel = np.arange(n_frames) * hop / float(EVAL_SR)

        fig_s, axes_s = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

        im0 = axes_s[0].imshow(
            ref_log,
            aspect="auto",
            origin="lower",
            extent=[t_axis_mel[0], t_axis_mel[-1], 0, ref_log.shape[0]],
        )
        axes_s[0].set_title("Original")
        axes_s[0].set_ylabel("Mel bin")
        axes_s[0].set_xlabel("Time (s)")

        im1 = axes_s[1].imshow(
            est_log,
            aspect="auto",
            origin="lower",
            extent=[t_axis_mel[0], t_axis_mel[-1], 0, est_log.shape[0]],
        )
        axes_s[1].set_title("Reconstructed")
        axes_s[1].set_xlabel("Time (s)")

        fig_s.suptitle(f"Log-mel Spectrogram – {stem}", y=0.98)
        fig_s.tight_layout(rect=[0.0, 0.0, 0.93, 0.92])

        # Add a separate axes for the colorbar on the far right
        cax = fig_s.add_axes([0.94, 0.15, 0.015, 0.65])
        cbar = fig_s.colorbar(im1, cax=cax)
        cbar.set_label("Log-mel (dB)")

        out_mel = os.path.join(out_dir, f"proposed_example_{idx+1:02d}_mel.png")
        fig_s.savefig(out_mel, dpi=180)
        plt.close(fig_s)
        print(f"Saved mel-spectrogram example {idx+1} → {out_mel}")

# ===================== MAIN =====================

def main():
    # pair up audio & tactile files
    pairs = list_pairs(AUDIO_DIR, TACT_DIR)
    if not pairs:
        raise RuntimeError("No audio/tactile pairs found.")

    # global peak over ALL tactile files
    tact_paths = [t for _, t, _ in pairs]
    peak_global = compute_global_peak(tact_paths)
    print(f"[Global] MAX amplitude over all tactile files = {peak_global:.6f}")

    # DAC 24k
    dac24 = eval_dac24(pairs, DAC_NQ_LIST, peak_global)

    # Proposed sweep
    proposed_rows = eval_proposed_runs(pairs, SWEEP_ROOT, peak_global)

    # VC-PWQ
    vc_rows = eval_vc_pwq(VC_DEC_DIR, VC_COMP_DIR, TACT_DIR, VC_CONFIGS, peak_global)

    # ---- plot 10 example waveforms for best proposed run (highest ST-SIM) ----
    if proposed_rows:
        best_idx = int(np.argmax([r["stsim_mean"] for r in proposed_rows]))
        best_run = proposed_rows[best_idx]
        plot_proposed_examples(best_run, pairs, num_examples=10, out_dir=OUT_DIR)

    # Save combined JSON
    combined = {
        "dac_24khz": dac24,
        "proposed_runs": proposed_rows,
        "vc_pwq_runs": vc_rows,
        "config": {
            "eval_sr": EVAL_SR,
            "orig_tact_sr_baseline": ORIG_3K,
            "pcm_kbps_tact_orig": PCM_KBPS_TACT_ORIG,
            "dac_nq_list": DAC_NQ_LIST,
            "max_align_shift_samples": MAX_ALIGN_SHIFT,
            "sweep_root": SWEEP_ROOT,
            "vc_dec_dir": VC_DEC_DIR,
            "vc_comp_dir": VC_COMP_DIR,
            "peak_global_raw": peak_global,
            "vc_buffer_delay_ms": VC_BUFFER_MS,
        }
    }
    out_json = os.path.join(OUT_DIR, "eval_all_vs_dac24_vcpwq_rawPSNR_latency.json")
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved JSON → {out_json}")

    # ---------- LATENCY SUMMARY (printed) ----------
    print("\n==== Latency summary (ms) ====")

    # DAC
    nqs = sorted(dac24.keys(), key=lambda q: dac24[q]["kbps"])
    print("\nDAC 24kHz:")
    for q in nqs:
        d = dac24[q]
        print(f"  n_q={q}: enc={d['encoding_delay_ms']:.2f} ms, "
              f"dec={d['decoding_delay_ms']:.2f} ms, "
              f"buffer={d['buffer_delay_ms']:.2f} ms, "
              f"kbps={d['kbps']:.2f}")

    # Proposed
    print("\nProposed AR+RVQ runs:")
    for r in sorted(proposed_rows, key=lambda x: x["kbps"]):
        print(f"  {r['run']}: K={r['embed']}, B={r['books']}, "
              f"enc={r['encoding_delay_ms']:.2f} ms, "
              f"dec={r['decoding_delay_ms']:.2f} ms, "
              f"buffer={r['buffer_delay_ms']:.2f} ms, "
              f"kbps={r['kbps']:.2f}")

    # VC-PWQ
    print("\nVC-PWQ (latency partly from paper, enc/dec not reported):")
    for r in sorted(vc_rows, key=lambda x: x["kbps"] if not math.isnan(x["kbps"]) else 1e9):
        enc_ms = r["encoding_delay_ms"]
        dec_ms = r["decoding_delay_ms"]
        print(f"  {r['label']}: b={r['b']}, "
              f"enc={enc_ms}, dec={dec_ms}, "
              f"buffer={r['buffer_delay_ms']:.2f} ms, "
              f"kbps={r['kbps']:.2f}")

    # ---------- Prepare arrays for plots ----------
    # DAC
    br_d = np.array([dac24[q]["kbps"] for q in nqs], dtype=float)
    ss_d = np.array([dac24[q]["stsim_mean"] for q in nqs], dtype=float)
    sp_d = np.array([dac24[q]["psnr_mean"] for q in nqs], dtype=float)
    cr_d = np.array([dac24[q]["compression_ratio"] for q in nqs], dtype=float)
    cs_d = np.array([dac24[q]["stsim_ci95"] for q in nqs], dtype=float)
    cp_d = np.array([dac24[q]["psnr_ci95"] for q in nqs], dtype=float)

    # Proposed
    def _group_by_embed_local(rows):
        groups = {}
        for r in rows:
            k = int(r["embed"])
            groups.setdefault(k, []).append(r)
        for k in groups:
            groups[k] = sorted(groups[k], key=lambda x: x["kbps"])
        return groups

    groups = _group_by_embed_local(proposed_rows)
    markers = {128: "o", 256: "^", 512: "D"}

    # VC-PWQ
    br_v = np.array([r["kbps"] for r in vc_rows], dtype=float)
    ss_v = np.array([r["stsim_mean"] for r in vc_rows], dtype=float)
    sp_v = np.array([r["psnr_mean"] for r in vc_rows], dtype=float)
    cr_v = np.array([r["compression_ratio"] for r in vc_rows], dtype=float)
    cs_v = np.array([r["stsim_ci95"] for r in vc_rows], dtype=float)
    cp_v = np.array([r["psnr_ci95"] for r in vc_rows], dtype=float)

    # Sort indices for connecting lines
    order_br_v = np.argsort(br_v)
    order_cr_v = np.argsort(cr_v)
    order_cr_d = np.argsort(cr_d)

    all_br = list(br_d) + [r["kbps"] for r in proposed_rows] + list(br_v)
    xmax = max(1.2 * max(all_br), 15.0)

    # ----- fixed colors: 3x Proposed + DAC + VC (5 distinct lines) -----
    proposed_colors = {
        128: "C0",
        256: "C1",
        512: "C2",
    }
    color_dac = "C3"
    color_vc  = "C4"

    # ---------- PLOT 1: ST-SIM vs Bitrate ----------
    plt.figure(figsize=(7.2, 5.4))

    for embed, rows in sorted(groups.items()):
        x = [r["kbps"] for r in rows]
        y = [r["stsim_mean"] for r in rows]
        ci = [r["stsim_ci95"] for r in rows]
        col = proposed_colors.get(embed, "0.5")
        _errfill(x, y, ci, label=f"Proposed (K={embed})",
                 marker=markers.get(embed, "o"), color=col)

    plt.errorbar(br_d, ss_d, yerr=cs_d,
                 fmt="s-", ms=6, lw=2.0,
                 label="DAC 24k", color=color_dac)

    br_v_s = br_v[order_br_v]
    ss_v_s = ss_v[order_br_v]
    cs_v_s = cs_v[order_br_v]
    plt.errorbar(br_v_s, ss_v_s, yerr=cs_v_s,
                 fmt="v-", ms=6, lw=2.0,
                 label="VC-PWQ", color=color_vc)

    plt.ylim(Y_STSIM_MIN, Y_STSIM_MAX)
    plt.xlim(0, xmax)
    plt.xlabel("Bitrate (kbps)")
    plt.ylabel("ST-SIM (↑)")
    plt.grid(True, alpha=0.3)
    plt.title("ST-SIM vs Bitrate — DAC vs Proposed vs VC-PWQ")
    plt.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    plt.tight_layout()
    f1 = os.path.join(OUT_DIR, "stsim_vs_bitrate_all_rawPSNR.png")
    plt.savefig(f1, dpi=180); plt.close()
    print(f"Saved: {f1}")

    # ---------- PLOT 2: PSNR vs Bitrate ----------
    plt.figure(figsize=(7.2, 5.4))

    for embed, rows in sorted(groups.items()):
        x = [r["kbps"] for r in rows]
        y = [r["psnr_mean"] for r in rows]
        ci = [r["psnr_ci95"] for r in rows]
        col = proposed_colors.get(embed, "0.5")
        _errfill(x, y, ci, label=f"Proposed (K={embed})",
                 marker=markers.get(embed, "o"), color=col)

    plt.errorbar(br_d, sp_d, yerr=cp_d,
                 fmt="s-", ms=6, lw=2.0,
                 label="DAC 24k", color=color_dac)

    sp_v_s = sp_v[order_br_v]
    cp_v_s = cp_v[order_br_v]
    plt.errorbar(br_v_s, sp_v_s, yerr=cp_v_s,
                 fmt="v-", ms=6, lw=2.0,
                 label="VC-PWQ", color=color_vc)

    plt.xlim(0, xmax)
    plt.xlabel("Bitrate (kbps)")
    plt.ylabel("PSNR (global peak, raw) [dB, ↑]")
    plt.grid(True, alpha=0.3)
    plt.title("PSNR vs Bitrate — DAC vs Proposed vs VC-PWQ")
    plt.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    plt.tight_layout()
    f2 = os.path.join(OUT_DIR, "psnr_vs_bitrate_all_rawPSNR.png")
    plt.savefig(f2, dpi=180); plt.close()
    print(f"Saved: {f2}")

    # ---------- PLOT 3: ST-SIM vs Compression Ratio ----------
    plt.figure(figsize=(7.2, 5.4))

    for embed, rows in sorted(groups.items()):
        x = [r["compression_ratio"] for r in rows]
        y = [r["stsim_mean"] for r in rows]
        ci = [r["stsim_ci95"] for r in rows]
        col = proposed_colors.get(embed, "0.5")
        _errfill(x, y, ci, label=f"Proposed (K={embed})",
                 marker=markers.get(embed, "o"), color=col)

    cr_d_s = cr_d[order_cr_d]
    ss_d_s = ss_d[order_cr_d]
    cs_d_s = cs_d[order_cr_d]
    plt.errorbar(cr_d_s, ss_d_s, yerr=cs_d_s,
                 fmt="s-", ms=6, lw=2.0,
                 label="DAC 24k", color=color_dac)

    cr_v_s = cr_v[order_cr_v]
    ss_v_s_cr = ss_v[order_cr_v]
    cs_v_s_cr = cs_v[order_cr_v]
    plt.errorbar(cr_v_s, ss_v_s_cr, yerr=cs_v_s_cr,
                 fmt="v-", ms=6, lw=2.0,
                 label="VC-PWQ", color=color_vc)

    plt.xlabel("Compression Ratio")
    plt.ylabel("ST-SIM (↑)")
    plt.grid(True, alpha=0.3)
    plt.title("ST-SIM vs Compression Ratio — DAC vs Proposed vs VC-PWQ")
    plt.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    plt.tight_layout()
    f3 = os.path.join(OUT_DIR, "stsim_vs_cr_all_rawPSNR.png")
    plt.savefig(f3, dpi=180); plt.close()
    print(f"Saved: {f3}")

    # ---------- PLOT 4: PSNR vs Compression Ratio ----------
    plt.figure(figsize=(7.2, 5.4))

    for embed, rows in sorted(groups.items()):
        x = [r["compression_ratio"] for r in rows]
        y = [r["psnr_mean"] for r in rows]
        ci = [r["psnr_ci95"] for r in rows]
        col = proposed_colors.get(embed, "0.5")
        _errfill(x, y, ci, label=f"Proposed (K={embed})",
                 marker=markers.get(embed, "o"), color=col)

    sp_d_s = sp_d[order_cr_d]
    cp_d_s = cp_d[order_cr_d]
    plt.errorbar(cr_d_s, sp_d_s, yerr=cp_d_s,
                 fmt="s-", ms=6, lw=2.0,
                 label="DAC 24k", color=color_dac)

    sp_v_s_cr = sp_v[order_cr_v]
    cp_v_s_cr = cp_v[order_cr_v]
    plt.errorbar(cr_v_s, sp_v_s_cr, yerr=cp_v_s_cr,
                 fmt="v-", ms=6, lw=2.0,
                 label="VC-PWQ", color=color_vc)

    plt.xlabel("Compression Ratio")
    plt.ylabel("PSNR (global peak, raw) [dB, ↑]")
    plt.grid(True, alpha=0.3)
    plt.title("PSNR vs Compression Ratio — DAC vs Proposed vs VC-PWQ")
    plt.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    plt.tight_layout()
    f4 = os.path.join(OUT_DIR, "psnr_vs_cr_all_rawPSNR.png")
    plt.savefig(f4, dpi=180); plt.close()
    print(f"Saved: {f4}")

    # ---------- PLOT 5: Compression Ratio vs Bitrate ----------
    plt.figure(figsize=(7.2, 5.4))

    for embed, rows in sorted(groups.items()):
        x = np.array([r["kbps"] for r in rows], dtype=float)
        y = np.array([r["compression_ratio"] for r in rows], dtype=float)
        col = proposed_colors.get(embed, "0.5")
        plt.plot(x, y, markers.get(embed, "o") + "-",
                 lw=2.0, ms=6, label=f"Proposed (K={embed})", color=col)

    plt.plot(br_d, cr_d, "s-", lw=2.0, ms=6,
             label="DAC 24k", color=color_dac)

    cr_v_s_b = cr_v[order_br_v]
    plt.plot(br_v_s, cr_v_s_b, "v-", lw=2.0, ms=6,
             label="VC-PWQ", color=color_vc)

    plt.xlabel("Bitrate (kbps, →)")
    plt.ylabel("Compression Ratio (↑)")
    plt.xlim(0, xmax)
    plt.grid(True, alpha=0.3)
    plt.title("Compression Ratio vs Bitrate — DAC vs Proposed vs VC-PWQ")
    plt.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    plt.tight_layout()
    f5 = os.path.join(OUT_DIR, "cr_vs_bitrate_all_rawPSNR.png")
    plt.savefig(f5, dpi=180); plt.close()
    print(f"Saved: {f5}")

if __name__ == "__main__":
    main()
