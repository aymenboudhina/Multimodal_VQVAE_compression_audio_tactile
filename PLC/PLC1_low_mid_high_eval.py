#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for Audio→Tactile PLC model (AllPredPLC) with 3 loss categories.

Categories follow the PLC-style burst ranges (approx):
  - low    : max burst between 20–120 ms
  - medium : max burst between 120–320 ms
  - high   : max burst between 320–1000 ms

For each category, we:
  - Simulate burst losses in the tactile latent sequence.
  - Run the trained AllPredPLC model once per file.
  - Compute GLOBAL metrics on denormalized, aligned tactile:
      * PSNR (with GLOBAL peak over all tactile WAVs)
      * ST-SIM (Mel spectrogram similarity at 24 kHz)
      * MAE  (mean absolute error)
  - Aggregate mean PSNR, ST-SIM, MAE over all files for that category.

Outputs:
  RUN_DIR/
    eval_cat_metrics_low.csv
    eval_cat_metrics_medium.csv
    eval_cat_metrics_high.csv
    eval_cat_summary.json
"""

import os, glob, math, json, random, warnings
from pathlib import Path
from typing import List, Tuple, Dict

warnings.filterwarnings("once", category=UserWarning)

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt  # not heavily used but kept if you want plots later

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

import dac  # pip install descript-audio-codec

# ====================== PATHS / CONFIG ======================
AUDIO_DIR  = r"/home/student/studentdata/WAV_Files_raw"
TACT_DIR   = r"/home/student/studentdata/Vibrotactile_Files_Raw"

OUT_ROOT   = r"/home/student/studentdata/A2T_PLC_AR"
RUN_DIR    = os.path.join(OUT_ROOT, "plc_run")
CKPT_PATH  = os.path.join(RUN_DIR, "best.pth")

os.makedirs(RUN_DIR, exist_ok=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR  = 24000     # native DAC rate and PLC model rate
EVAL_SR    = 24000     # Mel/ST-SIM evaluation SR (same here)
SEED       = 7
MAX_EVAL_FILES = None  # e.g., 100 for subset; None = all

# For time alignment
MAX_ALIGN_SHIFT  = 400   # samples at 24 kHz (~16.7 ms)

# Mel/ST-SIM config
MEL_NFFT = 512
MEL_HOP  = 128
MEL_MELS = 64

EPS = 1e-12

# ================== PACKET LOSS CATEGORIES (as in training) ==================
# Probabilities used at training time (for sampling categories) are *not* needed
# here, but we keep them for reference.
CAT_PROBS = {
    "low":   0.52,
    "medium":0.32,
    "high":  0.16,
}

# Burst length ranges in milliseconds per category
CAT_BURST_MS = {
    "low":   (20.0, 120.0),    # ~ up to 120 ms
    "medium":(120.0, 320.0),   # 120–320 ms
    "high":  (320.0, 1000.0),  # 320–1000 ms
}

# Number of bursts per 1 s segment (min,max) for each category
CAT_N_BURSTS = {
    "low":   (1, 2),
    "medium":(1, 3),
    "high":  (1, 4),
}

# ====================== UTILS ======================
def set_global_seed(seed: int):
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

def resample_to(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
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

def mae_global(ref_1T: torch.Tensor, est_1T: torch.Tensor) -> float:
    r = ref_1T.reshape(-1).to(torch.float32)
    e = est_1T.reshape(-1).to(torch.float32)
    return float((r - e).abs().mean().cpu())

# ===================== Mel front-end / ST-SIM =====================
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
    with torch.no_grad():
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

def compute_stsim_mel_global(ref_1T: torch.Tensor, est_1T: torch.Tensor, sr: int = EVAL_SR) -> float:
    """
    Global ST-SIM on Mel magnitude at 24 kHz.
    ref_1T, est_1T: [1,T]
    """
    ref_b = ref_1T.unsqueeze(0)  # [1,1,T] effectively
    est_b = est_1T.unsqueeze(0)
    with torch.no_grad():
        Mx = _mel_mag(ref_b, sr=sr)  # [1, n_mels, T_f]
        My = _mel_mag(est_b, sr=sr)

    X = Mx[0].cpu().numpy()
    Y = My[0].cpu().numpy()

    if SKIMAGE_AVAILABLE:
        try:
            return float(ssim(X, Y, data_range=1.0))
        except Exception as e:
            print(f"[WARN] ssim() failed, falling back. Error: {e}")

    # Fallback similarity measure if scikit-image is not available or fails
    diff = np.linalg.norm(X - Y)
    denom = np.linalg.norm(X) + np.linalg.norm(Y) + 1e-12
    return float(max(0.0, 1.0 - diff / denom))

# ================== MODEL PARTS (MATCH TRAINING, BUT CATEGORY-CONTROLLED) ==================
class PosEnc1D(nn.Module):
    def __init__(self, c, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, c)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, c, 2) * (-math.log(10000.0) / c))
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
    """
    Multi-head cross-attention + FFN:
      - Query: tactile latent (possibly masked)
      - Key/Value: audio latent
    """

    def __init__(self, c, heads=8, mlp_mul=2, dropout=0.1):
        super().__init__()
        self.pos = PosEnc1D(c)
        self.h   = heads
        self.dh  = c // heads
        assert c % heads == 0

        self.ln_q  = nn.LayerNorm(c)
        self.ln_kv = nn.LayerNorm(c)

        self.q_proj = nn.Linear(c, c, bias=False)
        self.k_proj = nn.Linear(c, c, bias=False)
        self.v_proj = nn.Linear(c, c, bias=False)
        self.out    = nn.Linear(c, c, bias=False)
        self.drop   = nn.Dropout(dropout)

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
        # zt_prev, za: [B, C, T]
        q  = self.pos(zt_prev).permute(0, 2, 1)  # [B,T,C]
        kv = self.pos(za).permute(0, 2, 1)       # [B,T,C]
        q  = self.ln_q(q)
        kv = self.ln_kv(kv)

        Q = self._split(self.q_proj(q))
        K = self._split(self.k_proj(kv))
        V = self._split(self.v_proj(kv))

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dh)
        ctx  = attn.softmax(dim=-1) @ V
        y    = self.out(self.drop(self._merge(ctx)))
        y    = self.ffn(y + q) + (y + q)
        return y.permute(0, 2, 1)  # [B,C,T]

# ---------- Category-based latent burst loss ----------
def make_category_token_loss_mask_for_category(
    category: str,
    batch_size: int,
    T_lat: int,
    tokens_per_sec: float,
    device,
) -> torch.Tensor:
    """
    Simulate burst packet loss on latent tokens for a FIXED category:
      - category ∈ {"low","medium","high"}
      - Use CAT_BURST_MS[category] and CAT_N_BURSTS[category] to draw
        burst lengths and number of bursts.

    Returns:
      mask: [B, T_lat] boolean (True where token is lost).
    """
    if T_lat <= 0:
        return torch.zeros(batch_size, 0, dtype=torch.bool, device=device)

    if category not in CAT_BURST_MS:
        raise ValueError(f"Unknown category: {category}")

    min_ms, max_ms = CAT_BURST_MS[category]
    nb_min, nb_max = CAT_N_BURSTS[category]

    mask = torch.zeros(batch_size, T_lat, dtype=torch.bool, device=device)

    for b in range(batch_size):
        # Convert ms → tokens; ensure at least 1 token and not more than T_lat
        min_tok = max(1, int(round(min_ms * tokens_per_sec / 1000.0)))
        max_tok = max(min_tok, int(round(max_ms * tokens_per_sec / 1000.0)))
        max_tok = min(max_tok, T_lat)

        n_bursts = random.randint(nb_min, nb_max)

        for _ in range(n_bursts):
            L_b = random.randint(min_tok, max_tok)
            if L_b >= T_lat:
                mask[b, :] = True
                break
            start_max = max(0, T_lat - L_b)
            s = random.randint(0, start_max)
            mask[b, s:s+L_b] = True

    return mask

class AllPredPLC(nn.Module):
    """
    Audio→Tactile PLC model:
      - Frozen DAC encoders/decoders.
      - CrossPredictor uses audio latents + masked tactile latents to
        predict replacements for missing tokens.
      - In this eval version, we can FIX the loss category per call.
    """

    def __init__(self, A_ENC, A_QUANT, T_ENC, T_DEC, c_lat):
        super().__init__()
        self.A_ENC  = A_ENC
        self.A_QUANT = A_QUANT
        self.T_ENC  = T_ENC
        self.T_DEC  = T_DEC

        for m in [self.A_ENC, self.A_QUANT, self.T_ENC, self.T_DEC]:
            for p in m.parameters():
                p.requires_grad_(False)

        self.predict   = CrossPredictor(c=c_lat, heads=8, mlp_mul=2, dropout=0.1)
        self.tokennorm = TokenNorm(c_lat)

    def forward_step(self, a_1T: torch.Tensor, tc_1T: torch.Tensor, category: str):
        """
        a_1T, tc_1T: [B,1,T_wav]
        category: "low", "medium", "high"
        Returns:
          dict with:
            y_hat: [B,1,T_out]
            tgt:   [B,1,T_out]
            latent_mask: [B,1,T_lat] bool
        """
        B, _, Tw = tc_1T.shape

        # Audio backbone
        za = self.A_ENC(a_1T)        # [B,C,T_lat_a]
        qa, *_ = self.A_QUANT(za)    # [B,C,T_lat_a]

        # Tactile backbone
        zt_full = self.T_ENC(tc_1T)  # [B,C,T_lat_t]
        B, C, T_lat = zt_full.shape

        # tokens_per_sec ≈ T_lat for 1s segments
        tokens_per_sec = float(T_lat)  # SEG_SEC ≈ 1.0

        # Category-based burst mask
        mask_tokens = make_category_token_loss_mask_for_category(
            category=category,
            batch_size=B,
            T_lat=T_lat,
            tokens_per_sec=tokens_per_sec,
            device=zt_full.device,
        )  # [B,T_lat]
        mask_tokens_b1T = mask_tokens.unsqueeze(1)  # [B,1,T_lat]

        # Zero-out masked tactile tokens (receiver sees zeros there)
        zt_in = zt_full * (~mask_tokens_b1T)

        # Align audio latents length if necessary
        if qa.shape[-1] != T_lat:
            qa_res = F.interpolate(qa, size=T_lat, mode="linear", align_corners=False)
        else:
            qa_res = qa

        # Cross-predict all tokens
        z_pred = self.predict(zt_in, qa_res)  # [B,C,T_lat]

        # Fill missing tokens with predictions
        z_filled = torch.where(mask_tokens_b1T, z_pred, zt_in)

        # Decode tactile waveform
        y_hat = self.T_DEC(z_filled)  # [B,1,T_out]
        T = min(y_hat.shape[-1], tc_1T.shape[-1], Tw)
        y_hat = y_hat[..., :T]
        tgt   = tc_1T[..., :T]

        return {
            "y_hat":       finite_or_zero(y_hat),
            "tgt":         finite_or_zero(tgt),
            "latent_mask": mask_tokens_b1T,
        }

def build_backbones():
    dac_audio = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    dac_tact  = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()

    A_ENC, A_QUANT = dac_audio.encoder, dac_audio.quantizer
    T_ENC, T_DEC   = dac_tact.encoder,  dac_tact.decoder

    dummy = torch.randn(1,1,TARGET_SR, device=DEVICE)
    with torch.no_grad():
        za = A_ENC(dummy)
        C  = za.size(1)
        toks = za.size(-1)

    print(f"[Latents] C={C}, tokens/sec≈{toks}")
    return A_ENC, A_QUANT, T_ENC, T_DEC, C

# ================== MAIN EVAL LOOP ==================
def eval_model_categories():
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CKPT_PATH}")

    set_global_seed(SEED)

    print("[Eval] Building backbones and model...")
    A_ENC, A_QUANT, T_ENC, T_DEC, C = build_backbones()
    net = AllPredPLC(A_ENC, A_QUANT, T_ENC, T_DEC, c_lat=C).to(DEVICE)
    net.eval()

    print(f"[Eval] Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt["model"], strict=True)
    print("[Eval] Checkpoint loaded.")

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

    categories = ["low", "medium", "high"]
    summary = {"peak_global": peak_global, "num_files": len(items), "categories": {}}
    BASE_SEED = SEED * 1000

    for cat_idx, cat in enumerate(categories):
        print(f"\n========== Evaluating category: {cat.upper()} ==========")

        cat_rows = []
        psnr_list = []
        stsim_list = []
        mae_list = []

        for f_idx, (ap, tp, stem) in enumerate(items, start=1):
            print(f"[{cat} {f_idx}/{len(items)}] {stem}")

            # Category + file-dependent seed for reproducible masks
            file_seed = BASE_SEED + cat_idx * 100000 + f_idx
            set_global_seed(file_seed)

            # Load RAW audio + tactile
            aw_raw, asr = load_wav_sf(ap)  # [C,T_a]
            tw_raw, tsr = load_wav_sf(tp)  # [C,T_t]

            # Per-file tactile scale
            scale = max(float(tw_raw.abs().max().cpu()), 1e-8)

            # Resample to TARGET_SR and normalize tactile to unit for the model
            aw_24 = resample_to(aw_raw, asr, TARGET_SR)[:1, :]          # [1,T]
            tw_24_norm = resample_to(tw_raw / scale, tsr, TARGET_SR)[:1, :]  # [1,T]

            L = min(aw_24.shape[-1], tw_24_norm.shape[-1])
            aw_24 = aw_24[..., :L]
            tw_24_norm = tw_24_norm[..., :L]

            a_1T = sanitize_wave(aw_24).unsqueeze(0).to(DEVICE)       # [1,1,T]
            t_1T = sanitize_wave(tw_24_norm).unsqueeze(0).to(DEVICE)  # [1,1,T]

            with torch.no_grad():
                out = net.forward_step(a_1T, t_1T, category=cat)

            # Prediction in normalized domain
            y_hat_norm = out["y_hat"].detach().cpu()[0, 0, :]  # [T_pred]

            # Reference tactile at original amplitude at 24k
            ref_24 = resample_to(tw_raw, tsr, TARGET_SR)[0].cpu()  # [T_ref]
            # Denormalize prediction to original amplitude
            est_24 = y_hat_norm * scale                           # [T_est]

            ref_1T = ref_24.unsqueeze(0)  # [1,T]
            est_1T = est_24.unsqueeze(0)  # [1,T]

            # Crop & align
            ref_c, est_c = crop_match(ref_1T, est_1T)
            ref_a, est_a, best_shift = align_by_xcorr(ref_c, est_c, MAX_ALIGN_SHIFT)
            ref_a, est_a = crop_match(ref_a, est_a)

            # Global metrics
            psnr = psnr_global_peak_db(ref_a, est_a, peak_global)
            stsim = compute_stsim_mel_global(ref_a, est_a, sr=EVAL_SR)
            mae = mae_global(ref_a, est_a)

            psnr_list.append(psnr)
            stsim_list.append(stsim)
            mae_list.append(mae)

            print(f"   PSNR  : {psnr:7.3f} dB")
            print(f"   ST-SIM: {stsim:7.4f}")
            print(f"   MAE   : {mae:9.6f}")

            cat_rows.append(
                {
                    "stem": stem,
                    "len_samples": int(ref_a.numel()),
                    "psnr_global_db": psnr,
                    "stsim_global": stsim,
                    "mae_global": mae,
                    "best_shift_samples": int(best_shift),
                }
            )

        # ---- Category summary ----
        if cat_rows:
            mean_psnr = float(np.mean(psnr_list))
            mean_stsim = float(np.mean(stsim_list))
            mean_mae = float(np.mean(mae_list))

            print(f"\n===== Category {cat.upper()} summary =====")
            print(f"Mean PSNR  : {mean_psnr:7.3f} dB")
            print(f"Mean ST-SIM: {mean_stsim:7.4f}")
            print(f"Mean MAE   : {mean_mae:9.6f}")

            # Save per-file metrics CSV for this category
            csv_path = os.path.join(RUN_DIR, f"eval_cat_metrics_{cat}.csv")
            import csv
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["stem", "len_samples", "psnr_global_db",
                                "stsim_global", "mae_global", "best_shift_samples"],
                )
                writer.writeheader()
                for r in cat_rows:
                    writer.writerow(r)
            print(f"[{cat}] Per-file metrics → {csv_path}")

            summary["categories"][cat] = {
                "mean_psnr_global_db": mean_psnr,
                "mean_stsim_global": mean_stsim,
                "mean_mae_global": mean_mae,
                "num_files": len(cat_rows),
            }
        else:
            print(f"[{cat}] No rows collected; skipping summary for this category.")

    # Save overall summary JSON
    summary_path = os.path.join(RUN_DIR, "eval_cat_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ All categories evaluated. Summary → {summary_path}")


# ================== ENTRY POINT ==================
if __name__ == "__main__":
    eval_model_categories()
