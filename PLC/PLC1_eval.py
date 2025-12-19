#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for Audio→Tactile PLC model (AllPredPLC).

- PSNR computed with GLOBAL PEAK (over all tactile WAVs) and per-file scale,
  following the DAC baseline pattern:
    * use original tactile waveform to compute a per-file scale
    * feed scaled-to-unit signal to the model
    * denormalize the reconstruction with the same scale
    * compute PSNR on denormalized, aligned waveforms with global peak.

- ST-SIM computed at 24 kHz using Mel spectrograms (_mel_mag style).
- Metrics per file:
    * Global: PSNR, ST-SIM
    * Masked: PSNR, SNR, MAE, ST-SIM
    * Unmasked: PSNR, SNR, MAE, ST-SIM
- Only the BEST signals are plotted:
    * Top-K by global PSNR
    * Top-K by global ST-SIM
    * Union of both sets
- For each selected file:
    * Waveform plot (aligned original vs reconstructed) with masked tokens shaded.
    * Log-Mel spectrogram plot (aligned original vs reconstructed) with masked
      regions indicated by thin red stripes at the bottom.

Requirements:
    pip install soundfile matplotlib torch torchaudio scikit-image
"""

import os, glob, math, json, random, csv, warnings
from pathlib import Path
from typing import List, Tuple, Dict

warnings.filterwarnings("once", category=UserWarning)

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("[WARN] scikit-image not found, ST-SIM will fall back to a simple similarity metric.")

# ====================== PATHS / CONFIG ======================
AUDIO_DIR  = r"/home/student/studentdata/WAV_Files_raw"
TACT_DIR   = r"/home/student/studentdata/Vibrotactile_Files_Raw"

OUT_ROOT   = r"/home/student/studentdata/A2T_PLC_AR"
RUN_DIR    = os.path.join(OUT_ROOT, "plc_run")
CKPT_PATH  = os.path.join(RUN_DIR, "best.pth")

EVAL_PLOT_DIR   = os.path.join(RUN_DIR, "eval_plots_best")
EVAL_CSV_PATH   = os.path.join(RUN_DIR, "eval_metrics.csv")
os.makedirs(EVAL_PLOT_DIR, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR  = 24000       # native DAC rate (and PLC model rate)
EVAL_SR    = 24000       # Mel/ST-SIM evaluation SR
BEST_K     = 10          # Top-K by PSNR and by ST-SIM to plot
SEED       = 7
MAX_EVAL_FILES = None    # e.g. 100 to evaluate subset; None = all

# Packet loss settings (should match training)
PACKET_TOK       = 2
PACKET_LOSS_PROB = 0.5

# For time alignment
MAX_ALIGN_SHIFT  = 400   # samples at 24 kHz (~16.7 ms)

# Mel/ST-SIM config
MEL_NFFT = 512
MEL_HOP  = 128
MEL_MELS = 64

EPS = 1e-12

# ====================== UTILS ======================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sanitize_wave(x: torch.Tensor, clamp=True):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.9999, neginf=-0.9999)
    return x.clamp(-1.0, 1.0) if clamp else x

def finite_or_zero(x: torch.Tensor):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def list_wavs(dirpath) -> Dict[str, str]:
    return {Path(p).stem: p for p in glob.glob(os.path.join(dirpath, "*.wav"))}

def load_wav_sf(path):
    """Load WAV -> torch tensor [C,T], sr (no normalization/clipping)."""
    data, sr = sf.read(path, always_2d=True)
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    return torch.from_numpy(data).t().contiguous(), int(sr)  # [C,T], sr

def resample_to(wav, sr_in, sr_out):
    if sr_in == sr_out:
        return wav
    return torchaudio.transforms.Resample(sr_in, sr_out).to(wav.device)(
        wav.to(torch.float32)
    )

def pair_stems(sdir, tdir) -> List[Tuple[str, str, str]]:
    A, T = list_wavs(sdir), list_wavs(tdir)
    stems = sorted(set(A) & set(T))
    return [(A[s], T[s], s) for s in stems]

# ---------- Alignment, cropping, PSNR (global peak) ----------
def crop_match(ref_1T: torch.Tensor, est_1T: torch.Tensor):
    """
    Crop both to the same length (min).
    Inputs: [1,T1], [1,T2]
    Returns: [1,T'], [1,T']
    """
    T = min(ref_1T.size(-1), est_1T.size(-1))
    return ref_1T[..., :T], est_1T[..., :T]

def align_by_xcorr(ref_1T: torch.Tensor, est_1T: torch.Tensor, max_shift: int = MAX_ALIGN_SHIFT):
    """
    Align est to ref by maximizing cross-correlation over integer shifts
    at TARGET_SR (24 kHz).
    Inputs: ref_1T, est_1T: [1,T]
    Returns: ref_aligned[1,T'], est_aligned[1,T'], best_shift
    """
    r = ref_1T.squeeze(0).to(torch.float32)  # [T]
    e = est_1T.squeeze(0).to(torch.float32)  # [T]
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

def psnr_global_peak_db(ref: torch.Tensor, est: torch.Tensor, peak: float, eps: float = EPS):
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

def compute_global_peak(wav_paths: List[str]) -> float:
    """
    Max abs amplitude over ALL tactile wav files (RAW, no clamp).
    """
    max_val = 0.0
    for p in wav_paths:
        w, _ = load_wav_sf(p)  # [C,T], raw
        m = float(w.abs().max().cpu())
        if m > max_val:
            max_val = m
    return max_val if max_val > 0.0 else 1.0

# ---- masked / unmasked waveform metrics ----
def mae_subset(ref_vec: torch.Tensor, est_vec: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return float("nan")
    diff = (ref_vec - est_vec)[mask]
    return float(diff.abs().mean().cpu())

def snr_subset_db(ref_vec: torch.Tensor, est_vec: torch.Tensor, mask: torch.Tensor, eps: float = EPS) -> float:
    if mask.sum().item() == 0:
        return float("nan")
    r = ref_vec[mask].to(torch.float32)
    e = est_vec[mask].to(torch.float32)
    num = torch.mean(r ** 2)
    den = torch.mean((r - e) ** 2) + eps
    return float(10.0 * torch.log10(num / den).cpu())

def psnr_subset_db(ref_vec: torch.Tensor, est_vec: torch.Tensor, mask: torch.Tensor,
                   peak: float, eps: float = EPS) -> float:
    if mask.sum().item() == 0:
        return float("nan")
    r = ref_vec[mask].to(torch.float32)
    e = est_vec[mask].to(torch.float32)
    mse = torch.mean((r - e) ** 2) + eps
    peak = max(float(peak), eps)
    return float(10.0 * torch.log10((peak * peak) / mse).cpu())

# ===================== Mel front-end / ST-SIM (24 kHz) =====================
_MEL_CACHE = {}

def _mel_mag(x_1T: torch.Tensor, sr: int = EVAL_SR,
             n_fft: int = MEL_NFFT, hop: int = MEL_HOP, n_mels: int = MEL_MELS):
    """
    x_1T: [B,1,T] or [B,T]
    Returns: Mel magnitude [B, n_mels, T_frames] normalized to [0,1].
    """
    if x_1T.dim() == 3:
        x = x_1T[:, 0, :]
    else:
        x = x_1T
    with torch.cuda.amp.autocast(enabled=False):
        x = x.to(torch.float32)
        window = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        mag = spec.abs().clamp_min_(1e-8)

        key = (x.device.type, sr, n_fft, hop, n_mels)
        mel = _MEL_CACHE.get(key)
        if mel is None:
            mel = torchaudio.transforms.MelScale(
                n_mels=n_mels,
                sample_rate=sr,
                n_stft=n_fft // 2 + 1,
                f_min=0.0,
                f_max=sr * 0.5,
                norm=None,
                mel_scale="htk",
            ).to(x.device)
            _MEL_CACHE[key] = mel

        M = mel(mag)
        M = M / M.amax(dim=(1, 2), keepdim=True).clamp_min_(1e-8)
    return M

def compute_stsim_mel_with_mask(ref_1T: torch.Tensor,
                                est_1T: torch.Tensor,
                                latent_mask: torch.Tensor,
                                sr: int = EVAL_SR):
    """
    ST-SIM on Mel magnitude (here at 24 kHz), split into:
      - whole segment
      - frames mapped to masked tokens
      - frames mapped to unmasked tokens

    ref_1T, est_1T: [1,T] at TARGET_SR
    latent_mask: [T_lat] bool
    """
    ref_b = ref_1T.unsqueeze(0)  # [1,1,T]
    est_b = est_1T.unsqueeze(0)

    with torch.no_grad():
        Mx = _mel_mag(ref_b, sr=sr)  # [1, n_mels, T_f]
        My = _mel_mag(est_b, sr=sr)

    X = Mx[0].cpu().numpy()  # [n_mels, T_f]
    Y = My[0].cpu().numpy()

    def _stsim_core(A, B):
        if SKIMAGE_AVAILABLE:
            try:
                return float(ssim(A, B, data_range=1.0))
            except Exception as e:
                print(f"[WARN] ssim() failed, falling back. Error: {e}")
        diff = np.linalg.norm(A - B)
        denom = np.linalg.norm(A) + np.linalg.norm(B) + 1e-12
        return float(max(0.0, 1.0 - diff / denom))

    # Global ST-SIM
    stsim_global = _stsim_core(X, Y)

    n_mels, n_frames = X.shape
    T_wave = ref_1T.size(-1)
    T_lat = latent_mask.numel()
    if T_lat == 0 or T_wave == 0 or n_frames == 0:
        return stsim_global, float("nan"), float("nan")

    samples_per_token = float(T_wave) / float(T_lat)
    frame_centers = np.arange(n_frames) * MEL_HOP
    token_idx = np.floor(frame_centers / samples_per_token).astype(np.int64)
    token_idx = np.clip(token_idx, 0, T_lat - 1)
    mask_np = latent_mask.cpu().numpy().astype(bool)
    frame_mask = mask_np[token_idx]  # [T_f] bool

    def _subset(mask_vec: np.ndarray):
        idx = np.where(mask_vec)[0]
        if idx.size == 0:
            return float("nan")
        A = X[:, idx]
        B = Y[:, idx]
        return _stsim_core(A, B)

    stsim_masked   = _subset(frame_mask)
    stsim_unmasked = _subset(~frame_mask)

    return stsim_global, stsim_masked, stsim_unmasked

# ================== MODEL PARTS (MATCH TRAINING) ==================
import dac  # after torch imports


class PosEnc1D(nn.Module):
    def __init__(self, c, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, c)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, c, 2) * (-math.log(10000.0) / c))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

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
    """
    Multi-head cross-attention + FFN:
      - Query: tactile latent (possibly masked)
      - Key/Value: audio latent
    """

    def __init__(self, c, heads=8, mlp_mul=2, dropout=0.1):
        super().__init__()
        self.pos = PosEnc1D(c)
        self.h = heads
        self.dh = c // heads
        assert c % heads == 0

        self.ln_q = nn.LayerNorm(c)
        self.ln_kv = nn.LayerNorm(c)

        self.q_proj = nn.Linear(c, c, bias=False)
        self.k_proj = nn.Linear(c, c, bias=False)
        self.v_proj = nn.Linear(c, c, bias=False)
        self.out = nn.Linear(c, c, bias=False)
        self.drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(c),
            nn.Linear(c, mlp_mul * c),
            nn.GELU(),
            nn.Linear(mlp_mul * c, c),
        )

    def _split(self, x):
        B, T, C = x.shape
        return x.view(B, T, self.h, self.dh).permute(0, 2, 1, 3)

    def _merge(self, x):
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)

    def forward(self, zt_prev, za):
        # zt_prev, za: [B,C,T]
        q = self.pos(zt_prev).permute(0, 2, 1)  # [B,T,C]
        kv = self.pos(za).permute(0, 2, 1)     # [B,T,C]
        q = self.ln_q(q)
        kv = self.ln_kv(kv)

        Q = self._split(self.q_proj(q))
        K = self._split(self.k_proj(kv))
        V = self._split(self.v_proj(kv))

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dh)
        ctx = attn.softmax(dim=-1) @ V
        y = self.out(self.drop(self._merge(ctx)))
        y = self.ffn(y + q) + (y + q)
        return y.permute(0, 2, 1)  # [B,C,T]


def make_token_loss_mask(batch_size: int, T_lat: int, packet_tok: int, p_loss: float, device):
    """
    Simulate packet loss on latent tokens.

    - Split token sequence into packets of length `packet_tok`.
    - Each packet is dropped with probability `p_loss`.
    - Returns mask [B, T_lat] (True where token is lost/masked).
    """
    if packet_tok <= 0 or T_lat <= 0:
        return torch.zeros(batch_size, T_lat, dtype=torch.bool, device=device)

    num_packets = max(1, T_lat // packet_tok)
    probs = torch.rand(batch_size, num_packets, device=device)
    lost = probs < p_loss  # [B, P]

    mask = lost.unsqueeze(-1).expand(batch_size, num_packets, packet_tok).reshape(batch_size, -1)
    if mask.size(1) > T_lat:
        mask = mask[:, :T_lat]
    elif mask.size(1) < T_lat:
        pad = torch.zeros(batch_size, T_lat - mask.size(1), dtype=torch.bool, device=device)
        mask = torch.cat([mask, pad], dim=1)
    return mask  # [B,T_lat]


class AllPredPLC(nn.Module):
    """
    Audio→Tactile packet-loss concealment model:

      - Frozen DAC encoders/decoders.
      - Randomly mask (drop) tactile latent tokens.
      - CrossPredictor uses audio latents + masked tactile latents to
        predict replacements for missing tokens.
    """

    def __init__(self, A_ENC, A_QUANT, T_ENC, T_DEC, c_lat):
        super().__init__()
        self.A_ENC = A_ENC
        self.A_QUANT = A_QUANT
        self.T_ENC = T_ENC
        self.T_DEC = T_DEC

        # Freeze backbone
        for m in [self.A_ENC, self.A_QUANT, self.T_ENC, self.T_DEC]:
            for p in m.parameters():
                p.requires_grad_(False)

        self.predict = CrossPredictor(c=c_lat, heads=8, mlp_mul=2, dropout=0.1)
        self.tokennorm = TokenNorm(c_lat)

    def forward_step(self, a_1T, tc_1T):
        """
        a_1T, tc_1T: [B,1,T_wav] (both at TARGET_SR, tactile already scaled-to-unit)

        Returns dict with:
          y_hat: reconstructed tactile [B,1,T] (same normalized scale as input tactile)
          tgt:   input tactile [B,1,T]
          latent_mask: [B,1,T_lat] bool (True where token is masked)
        """
        B, _, Tw = tc_1T.shape

        # Audio backbone
        za = self.A_ENC(a_1T)        # [B,C,T_lat]
        qa, *_ = self.A_QUANT(za)    # [B,C,T_lat]

        # Tactile backbone
        zt_full = self.T_ENC(tc_1T)  # [B,C,T_lat]
        B, C, T_lat = zt_full.shape

        # Packet-loss mask on latent tokens
        mask_tokens = make_token_loss_mask(
            batch_size=B,
            T_lat=T_lat,
            packet_tok=PACKET_TOK,
            p_loss=PACKET_LOSS_PROB,
            device=zt_full.device,
        )  # [B,T_lat]
        mask_tokens_b1T = mask_tokens.unsqueeze(1)  # [B,1,T_lat]

        # Zero-out masked tactile tokens
        zt_in = zt_full * (~mask_tokens_b1T)

        # Cross-predict all tokens
        z_pred = self.predict(zt_in, qa)  # [B,C,T_lat]

        # Fill in missing tokens
        z_filled = torch.where(mask_tokens_b1T, z_pred, zt_in)  # [B,C,T_lat]

        # Decode tactile waveform
        y_hat = self.T_DEC(z_filled)
        T = min(y_hat.shape[-1], tc_1T.shape[-1], Tw)
        y_hat = y_hat[..., :T]
        tgt = tc_1T[..., :T]

        return {
            "y_hat": finite_or_zero(y_hat),
            "tgt": finite_or_zero(tgt),
            "latent_mask": mask_tokens_b1T,
        }


def build_backbones():
    dac_audio = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    dac_tact = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()

    A_ENC, A_QUANT = dac_audio.encoder, dac_audio.quantizer
    T_ENC, T_DEC = dac_tact.encoder, dac_tact.decoder

    dummy = torch.randn(1, 1, TARGET_SR, device=DEVICE)
    with torch.no_grad():
        za = A_ENC(dummy)
        C = za.size(1)
        toks = za.size(-1)

    print(f"[Latents] C={C}, tokens/sec≈{toks}")
    return A_ENC, A_QUANT, T_ENC, T_DEC, C

# ================== EVALUATION ==================
def eval_model():
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CKPT_PATH}")

    set_seed(SEED)

    print("[Eval] Building backbones and model...")
    A_ENC, A_QUANT, T_ENC, T_DEC, C = build_backbones()
    net = AllPredPLC(A_ENC, A_QUANT, T_ENC, T_DEC, c_lat=C).to(DEVICE)
    net.eval()

    print(f"[Eval] Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt["model"], strict=True)

    # Pair files
    items = pair_stems(AUDIO_DIR, TACT_DIR)
    if MAX_EVAL_FILES is not None:
        items = items[:MAX_EVAL_FILES]

    if not items:
        print("[Eval] No paired files found.")
        return

    # Compute global peak over ALL tactile WAVs (RAW)
    tact_paths = [tp for (_, tp, _) in items]
    print("[Eval] Computing global peak over tactile files...")
    peak_global = compute_global_peak(tact_paths)
    print(f"[Eval] Global peak amplitude: {peak_global:.6f}")

    print(f"[Eval] Evaluating on {len(items)} file pairs...")

    rows = []

    all_psnr_global = []
    all_stsim_global = []

    masked_psnr_list = []
    unmasked_psnr_list = []
    masked_snr_list = []
    unmasked_snr_list = []
    masked_mae_list = []
    unmasked_mae_list = []
    masked_stsim_list = []
    unmasked_stsim_list = []

    BASE_SEED = SEED * 1000

    # -------- Pass 1: metrics only (no plots) --------
    for idx, (ap, tp, stem) in enumerate(items, start=1):
        print(f"\n[Pass1 {idx}/{len(items)}] {stem}")

        # Fix RNG so that mask pattern for this file is reproducible
        torch.manual_seed(BASE_SEED + idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(BASE_SEED + idx)

        # Load audio + tactile (RAW)
        aw_raw, asr = load_wav_sf(ap)      # [C,T_a]
        tw_raw, tsr = load_wav_sf(tp)      # [C,T_t]

        # Per-file tactile scale (RAW)
        scale = max(float(tw_raw.abs().max().cpu()), 1e-8)

        # Resample to TARGET_SR, scale tactile to unit range for the model
        aw_24 = resample_to(aw_raw, asr, TARGET_SR)[:1, :]                # [1,T]
        tw_24_norm = resample_to(tw_raw / scale, tsr, TARGET_SR)[:1, :]   # [1,T]

        # Ensure same length for conditioning
        L = min(aw_24.shape[-1], tw_24_norm.shape[-1])
        aw_24 = aw_24[..., :L]
        tw_24_norm = tw_24_norm[..., :L]

        a_1T = sanitize_wave(aw_24).unsqueeze(0).to(DEVICE)       # [1,1,T]
        t_1T = sanitize_wave(tw_24_norm).unsqueeze(0).to(DEVICE)  # [1,1,T] normalized

        with torch.no_grad():
            out = net.forward_step(a_1T, t_1T)

        # PLC output in normalized domain at 24k
        y_hat_norm = out["y_hat"].detach().cpu()[0, 0, :]  # [T_out_norm]
        latent_mask = out["latent_mask"].detach().cpu()[0, 0, :].bool()  # [T_lat]

        # Reference tactile in original amplitude at 24k
        ref_24 = resample_to(tw_raw, tsr, TARGET_SR)[0].cpu()  # [T_ref24]
        # Denormalize prediction back to original amplitude
        est_24 = y_hat_norm * scale                           # [T_est24]

        # ---- Global metrics on denormalized, aligned 24k waveforms ----
        ref_1T = ref_24.unsqueeze(0)   # [1,T]
        est_1T = est_24.unsqueeze(0)   # [1,T]

        ref_c, est_c = crop_match(ref_1T, est_1T)
        ref_a, est_a, best_shift = align_by_xcorr(ref_c, est_c, MAX_ALIGN_SHIFT)
        ref_a, est_a = crop_match(ref_a, est_a)

        # Global PSNR with GLOBAL PEAK (like your DAC script)
        psnr_global = psnr_global_peak_db(ref_a, est_a, peak_global)

        # ST-SIM (Mel) at 24k from the same aligned signals
        stsim_global, stsim_masked, stsim_unmasked = compute_stsim_mel_with_mask(
            ref_a, est_a, latent_mask, sr=EVAL_SR
        )

        all_psnr_global.append(psnr_global)
        all_stsim_global.append(stsim_global)

        # ---- Masked vs unmasked waveform metrics (same aligned segment) ----
        ref_vec = ref_a.reshape(-1)
        est_vec = est_a.reshape(-1)
        T_wave = ref_vec.numel()
        T_lat = latent_mask.numel()
        if T_lat == 0 or T_wave == 0:
            mae_masked = mae_unmasked = float("nan")
            snr_masked = snr_unmasked = float("nan")
            psnr_masked = psnr_unmasked = float("nan")
        else:
            samples_per_token = float(T_wave) / float(T_lat)
            idx_samples = torch.arange(T_wave, dtype=torch.float32)
            token_idx = torch.floor(idx_samples / samples_per_token).long()
            token_idx = torch.clamp(token_idx, 0, T_lat - 1)
            sample_mask = latent_mask[token_idx]  # bool [T_wave]

            mae_masked   = mae_subset(ref_vec, est_vec, sample_mask)
            mae_unmasked = mae_subset(ref_vec, est_vec, ~sample_mask)
            snr_masked   = snr_subset_db(ref_vec, est_vec, sample_mask)
            snr_unmasked = snr_subset_db(ref_vec, est_vec, ~sample_mask)
            psnr_masked  = psnr_subset_db(ref_vec, est_vec, sample_mask,  peak_global)
            psnr_unmasked= psnr_subset_db(ref_vec, est_vec, ~sample_mask, peak_global)

        masked_psnr_list.append(psnr_masked)
        unmasked_psnr_list.append(psnr_unmasked)
        masked_snr_list.append(snr_masked)
        unmasked_snr_list.append(snr_unmasked)
        masked_mae_list.append(mae_masked)
        unmasked_mae_list.append(mae_unmasked)
        masked_stsim_list.append(stsim_masked)
        unmasked_stsim_list.append(stsim_unmasked)

        print(f"  Global PSNR (peak): {psnr_global:7.3f} dB")
        print(f"  Global ST-SIM     : {stsim_global:7.4f}")
        print(f"  Masked   PSNR     : {psnr_masked:7.3f} dB | SNR: {snr_masked:7.3f} dB | MAE: {mae_masked:8.5f} | ST-SIM: {stsim_masked:7.4f}")
        print(f"  Unmasked PSNR     : {psnr_unmasked:7.3f} dB | SNR: {snr_unmasked:7.3f} dB | MAE: {mae_unmasked:8.5f} | ST-SIM: {stsim_unmasked:7.4f}")

        rows.append(
            {
                "stem": stem,
                "len_samples": int(ref_a.numel()),
                "psnr_global_db": psnr_global,
                "stsim_global": stsim_global,
                "psnr_masked_db": psnr_masked,
                "psnr_unmasked_db": psnr_unmasked,
                "snr_masked_db": snr_masked,
                "snr_unmasked_db": snr_unmasked,
                "mae_masked": mae_masked,
                "mae_unmasked": mae_unmasked,
                "stsim_masked": stsim_masked,
                "stsim_unmasked": stsim_unmasked,
            }
        )

    # -------- Global stats + CSV --------
    if rows:
        mean_psnr_global = float(np.mean(all_psnr_global))
        mean_stsim_global = float(np.mean(all_stsim_global))

        mean_psnr_masked   = float(np.nanmean(masked_psnr_list))
        mean_psnr_unmasked = float(np.nanmean(unmasked_psnr_list))
        mean_snr_masked    = float(np.nanmean(masked_snr_list))
        mean_snr_unmasked  = float(np.nanmean(unmasked_snr_list))
        mean_mae_masked    = float(np.nanmean(masked_mae_list))
        mean_mae_unmasked  = float(np.nanmean(unmasked_mae_list))
        mean_stsim_masked  = float(np.nanmean(masked_stsim_list))
        mean_stsim_unmasked= float(np.nanmean(unmasked_stsim_list))

        print("\n===== Global evaluation summary =====")
        print(f"Global PSNR         : {mean_psnr_global:7.3f} dB")
        print(f"Global ST-SIM       : {mean_stsim_global:7.4f}")
        print(f"Masked   PSNR mean  : {mean_psnr_masked:7.3f} dB")
        print(f"Unmasked PSNR mean  : {mean_psnr_unmasked:7.3f} dB")
        print(f"Masked   SNR  mean  : {mean_snr_masked:7.3f} dB")
        print(f"Unmasked SNR  mean  : {mean_snr_unmasked:7.3f} dB")
        print(f"Masked   MAE  mean  : {mean_mae_masked:9.6f}")
        print(f"Unmasked MAE  mean  : {mean_mae_unmasked:9.6f}")
        print(f"Masked   ST-SIM mean: {mean_stsim_masked:7.4f}")
        print(f"Unmasked ST-SIM mean: {mean_stsim_unmasked:7.4f}")

        # Write CSV
        with open(EVAL_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "stem", "len_samples",
                    "psnr_global_db", "stsim_global",
                    "psnr_masked_db", "psnr_unmasked_db",
                    "snr_masked_db", "snr_unmasked_db",
                    "mae_masked", "mae_unmasked",
                    "stsim_masked", "stsim_unmasked",
                ],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        summary_json = {
            "mean_psnr_global_db": mean_psnr_global,
            "mean_stsim_global": mean_stsim_global,
            "mean_psnr_masked_db": mean_psnr_masked,
            "mean_psnr_unmasked_db": mean_psnr_unmasked,
            "mean_snr_masked_db": mean_snr_masked,
            "mean_snr_unmasked_db": mean_snr_unmasked,
            "mean_mae_masked": mean_mae_masked,
            "mean_mae_unmasked": mean_mae_unmasked,
            "mean_stsim_masked": mean_stsim_masked,
            "mean_stsim_unmasked": mean_stsim_unmasked,
            "num_files": len(rows),
            "peak_global": peak_global,
        }
        with open(os.path.join(RUN_DIR, "eval_summary.json"), "w") as f:
            json.dump(summary_json, f, indent=2)

        print(f"\nPer-file metrics → {EVAL_CSV_PATH}")
        print(f"Global summary   → {os.path.join(RUN_DIR, 'eval_summary.json')}")
    else:
        print("[Eval] No files evaluated (rows empty).")
        return

    # -------- Select best files by global PSNR & ST-SIM --------
    indices = list(range(len(rows)))

    idx_psnr_sorted = sorted(indices, key=lambda i: rows[i]["psnr_global_db"], reverse=True)
    idx_stsim_sorted = sorted(indices, key=lambda i: rows[i]["stsim_global"],   reverse=True)

    top_psnr_idx = idx_psnr_sorted[:BEST_K]
    top_stsim_idx = idx_stsim_sorted[:BEST_K]

    best_indices = sorted(set(top_psnr_idx) | set(top_stsim_idx))
    best_stems = {rows[i]["stem"] for i in best_indices}

    print("\n===== Best files (for plotting) =====")
    print(f"Top-K by PSNR   (K={BEST_K}): {[rows[i]['stem'] for i in top_psnr_idx]}")
    print(f"Top-K by ST-SIM (K={BEST_K}): {[rows[i]['stem'] for i in top_stsim_idx]}")
    print(f"Union (plotted)            : {sorted(best_stems)}")

    # -------- Pass 2: re-run only best files to generate plots (with mask) --------
    print("\n[Eval] Generating plots for best files only...")

    for idx, (ap, tp, stem) in enumerate(items, start=1):
        if stem not in best_stems:
            continue

        print(f"[Plot] {stem}")

        # Same RNG seeding so mask is the same as in pass 1
        torch.manual_seed(BASE_SEED + idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(BASE_SEED + idx)

        # Load audio + tactile RAW again
        aw_raw, asr = load_wav_sf(ap)
        tw_raw, tsr = load_wav_sf(tp)

        scale = max(float(tw_raw.abs().max().cpu()), 1e-8)

        aw_24 = resample_to(aw_raw, asr, TARGET_SR)[:1, :]
        tw_24_norm = resample_to(tw_raw / scale, tsr, TARGET_SR)[:1, :]

        L = min(aw_24.shape[-1], tw_24_norm.shape[-1])
        aw_24 = aw_24[..., :L]
        tw_24_norm = tw_24_norm[..., :L]

        a_1T = sanitize_wave(aw_24).unsqueeze(0).to(DEVICE)
        t_1T = sanitize_wave(tw_24_norm).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = net.forward_step(a_1T, t_1T)

        # latent mask (for marking masked tokens)
        latent_mask = out["latent_mask"].cpu().squeeze(0)  # [1,T_lat] -> [T_lat]
        latent_mask = latent_mask.squeeze(0).numpy().astype(bool)
        T_lat = latent_mask.shape[0]

        # Denormalized prediction and reference at 24k for plotting
        y_hat_norm = out["y_hat"].cpu().squeeze().numpy()  # [T]
        est_24 = y_hat_norm * scale                         # [T]
        ref_24 = resample_to(tw_raw, tsr, TARGET_SR)[0].cpu().numpy()

        ref_1T = torch.from_numpy(ref_24).view(1, -1)
        est_1T = torch.from_numpy(est_24).view(1, -1)

        ref_c, est_c = crop_match(ref_1T, est_1T)
        ref_a, est_a, _ = align_by_xcorr(ref_c, est_c, MAX_ALIGN_SHIFT)
        ref_a, est_a = crop_match(ref_a, est_a)

        tgt_al   = ref_a.squeeze(0).numpy()
        y_hat_al = est_a.squeeze(0).numpy()
        T_wave   = len(tgt_al)

        t_axis = np.arange(T_wave) / TARGET_SR
        samples_per_token = float(T_wave) / max(1, T_lat)

        # ---------- Waveform plot ----------
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t_axis, tgt_al, label="Original tactile", linewidth=1.1)
        ax.plot(t_axis, y_hat_al, label="Reconstructed tactile", linewidth=0.9, alpha=0.85)

        for j in range(T_lat):
            if latent_mask[j]:
                s_idx = int(j * samples_per_token)
                e_idx = int((j + 1) * samples_per_token) if j < T_lat - 1 else T_wave
                s_time = s_idx / TARGET_SR
                e_time = e_idx / TARGET_SR
                ax.axvspan(s_time, e_time, color="red", alpha=0.12)

        ax.set_title(f"{stem} — tactile PLC (red = masked latent tokens)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        plot_path = os.path.join(EVAL_PLOT_DIR, f"{stem}_tactile_plc_best.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Waveform plot saved to: {plot_path}")

        # ---------- Log-Mel spectrogram plot (aligned, at 24k) ----------
        ref_a_1T = ref_a.clone()
        est_a_1T = est_a.clone()

        with torch.no_grad():
            M_ref = _mel_mag(ref_a_1T, sr=EVAL_SR)  # [1, n_mels, T_frames]
            M_est = _mel_mag(est_a_1T, sr=EVAL_SR)

        ref_mel = M_ref.squeeze(0).cpu().numpy()
        est_mel = M_est.squeeze(0).cpu().numpy()

        ref_log = 20.0 * np.log10(ref_mel + 1e-8)
        est_log = 20.0 * np.log10(est_mel + 1e-8)

        n_mels, n_frames = ref_log.shape
        t_axis_mel = np.arange(n_frames) * MEL_HOP / float(EVAL_SR)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        im0 = axes[0].imshow(
            ref_log,
            aspect="auto",
            origin="lower",
            extent=[t_axis_mel[0], t_axis_mel[-1], 0, n_mels],
        )
        axes[0].set_ylabel("Mel bins")
        axes[0].set_title("Original tactile (Mel log-spectrogram)")
        fig.colorbar(im0, ax=axes[0], format="%.2f")

        im1 = axes[1].imshow(
            est_log,
            aspect="auto",
            origin="lower",
            extent=[t_axis_mel[0], t_axis_mel[-1], 0, n_mels],
        )
        axes[1].set_ylabel("Mel bins")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_title("Reconstructed tactile (Mel log-spectrogram)")
        fig.colorbar(im1, ax=axes[1], format="%.2f")

        # Mark masked regions as thin stripes at the bottom (no occlusion)
        for ax_s in axes:
            for j in range(T_lat):
                if latent_mask[j]:
                    s_idx = int(j * samples_per_token)
                    e_idx = int((j + 1) * samples_per_token) if j < T_lat - 1 else T_wave
                    s_time = s_idx / TARGET_SR
                    e_time = e_idx / TARGET_SR
                    ax_s.axvspan(
                        s_time,
                        e_time,
                        ymin=0.0,
                        ymax=0.05,  # bottom 5% of vertical axis
                        color="red",
                        alpha=0.6,
                    )

        # Small legend handle for masked packets
        axes[0].plot([], [], color="red", linewidth=4, alpha=0.6, label="Masked packet")
        axes[0].legend(loc="upper right", fontsize=8)

        mel_plot_path = os.path.join(EVAL_PLOT_DIR, f"{stem}_tactile_plc_mel_best.png")
        plt.tight_layout()
        plt.savefig(mel_plot_path, dpi=150)
        plt.close(fig)
        print(f"  Mel-spectrogram plot saved to: {mel_plot_path}")

    print("\n✅ Evaluation finished (metrics + best plots).")


# ================== MAIN ==================
if __name__ == "__main__":
    eval_model()
