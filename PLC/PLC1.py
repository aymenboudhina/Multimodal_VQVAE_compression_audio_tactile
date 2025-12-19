#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio → Tactile Packet-Loss Concealment with Cross-Prediction (no RVQ)
-----------------------------------------------------------------------

Goal:
  - Use a frozen DAC encoder/decoder for audio & tactile.
  - Simulate packet loss on the tactile latent sequence (mask/drop some tokens).
  - Use a cross-prediction Transformer (audio + masked tactile latents)
    to reconstruct the missing tactile information (packet-loss concealment).
  - No residual VQ, no bitrate saving – purely reconstruction of masked signals.

Outputs:
  OUT_ROOT/
    plc_run/
      last.pth, best.pth, curves.png, hist.json, meta.json
"""

import os, math, glob, random, json, warnings
from pathlib import Path
from typing import Tuple, List
warnings.filterwarnings("once", category=UserWarning)

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import dac  # pip install descript-audio-codec

# ====================== FIXED DATA/ROOT ======================
AUDIO_DIR  = r"/home/student/studentdata/WAV_Files_raw"
TACT_DIR   = r"/home/student/studentdata/Vibrotactile_Files_Raw"
OUT_ROOT   = r"/home/student/studentdata/A2T_PLC_AR"
os.makedirs(OUT_ROOT, exist_ok=True)

# ====================== TRAINING CONFIG ======================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR    = 24000
SEG_SEC      = 1.0
SEG          = int(SEG_SEC * TARGET_SR)

BATCH        = 6
EPOCHS       = 50
LR           = 2e-4
WD           = 1e-5
GRAD_CLIP    = 3.0
USE_AMP      = True
SEED         = 7

VAL_FRAC     = 0.2
MAX_VAL      = 300
NUM_WORKERS  = min(4, os.cpu_count() or 1)
PIN_MEMORY   = torch.cuda.is_available()

# Autoregressive latent roll (not strictly AR now but kept for possible future use)
AR_CHUNK_TOK = 16

# Packet loss simulation (on latent tokens)
PACKET_TOK          = 2    # number of latent tokens per "packet"
PACKET_LOSS_PROB    = 0.5  # probability that a packet is dropped (masked)

# Loss weights
W_WAV_L1     = 0.55
W_STFT       = 0.25
W_MELCOS     = 0.20

# Mel/STFT config for losses
MEL_NFFT = 512
MEL_HOP  = 128
MEL_MELS = 64
EPS      = 1e-7

# ================== UTILS ==================
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

def list_wavs(dirpath): 
    return {Path(p).stem: p for p in glob.glob(os.path.join(dirpath, "*.wav"))}

def load_wav_sf(path):
    data, sr = sf.read(path, always_2d=True)
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    return torch.from_numpy(data).t().contiguous(), int(sr)  # [C,T], sr

def resample_to(wav, sr_in, sr_out):
    if sr_in == sr_out:
        return wav
    with autocast('cuda', enabled=False):
        return torchaudio.transforms.Resample(sr_in, sr_out).to(wav.device)(wav.to(torch.float32))

def reflect_pad_right_any(x: torch.Tensor, need: int) -> torch.Tensor:
    assert x.dim() == 2
    while need > 0:
        T = x.size(-1)
        if T <= 1:
            x = F.pad(x, (0, need), mode="replicate")
            break
        step = min(need, T - 1)
        x = F.pad(x, (0, step), mode="reflect")
        need -= step
    return x

def pair_stems(sdir, tdir):
    A, T = list_wavs(sdir), list_wavs(tdir)
    stems = sorted(set(A) & set(T))
    return [(A[s], T[s], s) for s in stems]

# ================== DATASET ==================
class SegDataset(Dataset):
    def __init__(self, items, sr=TARGET_SR, seg=SEG, seed=SEED):
        self.items = items
        self.sr    = sr
        self.seg   = seg
        self.rng   = random.Random(seed)
        print(f"[Dataset] files: {len(items)} | seg={seg}")
    def __len__(self):
        return len(self.items)
    def _prep(self, p):
        w, sr = load_wav_sf(p)
        w = resample_to(w, sr, self.sr)[:1, :]
        return sanitize_wave(w)
    def __getitem__(self, i):
        ap, tp, _ = self.items[i]
        a = self._prep(ap)
        t = self._prep(tp)
        L = min(a.size(-1), t.size(-1))
        a, t = a[..., :L], t[..., :L]
        if a.size(-1) < self.seg:
            a = reflect_pad_right_any(a, self.seg - a.size(-1))
        if t.size(-1) < self.seg:
            t = reflect_pad_right_any(t, self.seg - t.size(-1))
        st = self.rng.randint(0, max(0, a.size(-1) - self.seg)) if a.size(-1) > self.seg else 0
        return a[:, st:st+self.seg].squeeze(0), t[:, st:st+self.seg].squeeze(0)

def collate_fn(batch):
    A  = torch.stack([b[0] for b in batch]).unsqueeze(1)  # [B,1,T]
    TC = torch.stack([b[1] for b in batch]).unsqueeze(1)  # [B,1,T]
    return sanitize_wave(A), sanitize_wave(TC)

# ============== LOSSES (no in-place) ==============
class MultiResSTFTLoss(nn.Module):
    def __init__(self, ffts=(256,512,1024), hops=(64,128,256), wins=(256,512,1024), eps=1e-7):
        super().__init__()
        self.ffts = ffts
        self.hops = hops
        self.wins = wins
        self.eps  = eps
    @staticmethod
    def _stft_mag(x, n_fft, hop, win, eps):
        x32 = torch.nan_to_num(x.squeeze(1).to(torch.float32), 0.0, 0.0, 0.0)
        window = torch.hann_window(win, device=x.device, dtype=torch.float32)
        spec = torch.stft(
            x32,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True
        )
        return spec.abs().clamp_min(eps)
    def forward(self, x, y):
        x = finite_or_zero(x)
        y = finite_or_zero(y)
        used = 0
        sc = 0.0
        mag = 0.0
        for n, h, w in zip(self.ffts, self.hops, self.wins):
            if x.shape[-1] < max(8, w // 2):
                continue
            X = self._stft_mag(x, n, h, w, self.eps)
            Y = self._stft_mag(y, n, h, w, self.eps)
            num = (X - Y).pow(2).sum(dim=(1,2)).sqrt()
            den = Y.pow(2).sum(dim=(1,2)).sqrt().clamp_min(self.eps)
            sc  = sc + (num / den).mean()
            mag = mag + F.l1_loss(X, Y)
            used += 1
        if used == 0:
            return 0.1 * F.l1_loss(x, y)
        return 0.5 * sc / used + 0.5 * mag / used

class MelCosineLoss(nn.Module):
    def __init__(self, sr=TARGET_SR, n_fft=MEL_NFFT, hop=MEL_HOP, n_mels=MEL_MELS, eps=1e-7):
        super().__init__()
        self.sr     = sr
        self.n_fft  = n_fft
        self.hop    = hop
        self.n_mels = n_mels
        self.eps    = eps
        self.mel = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sr,
            n_stft=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sr * 0.5,
            norm=None,
            mel_scale="htk"
        )
    def _mel_mag(self, x_1T: torch.Tensor):
        x = x_1T[:, 0, :].to(torch.float32)
        window = torch.hann_window(self.n_fft, device=x.device, dtype=torch.float32)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True
        )
        mag = spec.abs().clamp_min(self.eps)
        M = self.mel.to(x.device)(mag)
        den = M.amax(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        M = (M / den + self.eps).log()
        return M
    def forward(self, x, y):
        X = self._mel_mag(x)
        Y = self._mel_mag(y)
        T = max(X.size(-1), Y.size(-1))
        if X.size(-1) != T:
            X = F.interpolate(X, size=T, mode="linear", align_corners=False)
        if Y.size(-1) != T:
            Y = F.interpolate(Y, size=T, mode="linear", align_corners=False)
        num = (X * Y).sum(dim=1)
        den = (X.norm(dim=1) * Y.norm(dim=1)).clamp_min(self.eps)
        cos = (num / den).clamp(-1, 1)
        return (1.0 - cos.mean())

MRSTFT = MultiResSTFTLoss().to(DEVICE)
MELCOS = MelCosineLoss().to(DEVICE)

def safe_l1(x, y):
    return F.l1_loss(finite_or_zero(x), finite_or_zero(y))

# ================== MODEL PARTS ==================
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
        q  = self.pos(zt_prev).permute(0, 2, 1)   # [B,T,C]
        kv = self.pos(za).permute(0, 2, 1)        # [B,T,C]
        q  = self.ln_q(q)
        kv = self.ln_kv(kv)
        Q = self._split(self.q_proj(q))
        K = self._split(self.k_proj(kv))
        V = self._split(self.v_proj(kv))
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dh)
        ctx  = attn.softmax(dim=-1) @ V
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
    lost  = probs < p_loss  # [B, P]
    # Expand to token level
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

    At inference, you would:
      - Encode tactile stream, zero-out tokens where packets are lost,
      - Run CrossPredictor with audio latents,
      - Replace only the missing tokens, then decode.
    """
    def __init__(self, A_ENC, A_QUANT, T_ENC, T_DEC, c_lat):
        super().__init__()
        self.A_ENC  = A_ENC
        self.A_QUANT = A_QUANT
        self.T_ENC  = T_ENC
        self.T_DEC  = T_DEC

        # Freeze all backbone parameters
        for m in [self.A_ENC, self.A_QUANT, self.T_ENC, self.T_DEC]:
            for p in m.parameters():
                p.requires_grad_(False)

        self.predict   = CrossPredictor(c=c_lat, heads=8, mlp_mul=2, dropout=0.1)
        self.tokennorm = TokenNorm(c_lat)

    def forward_step(self, a_1T, tc_1T):
        """
        a_1T, tc_1T: [B,1,T_wav]
        Returns:
          dict with y_hat, tgt, latent_mask
        """
        B, _, Tw = tc_1T.shape

        # Audio backbone
        za = self.A_ENC(a_1T)        # [B,C,T_lat]
        qa, *_ = self.A_QUANT(za)    # [B,C,T_lat] quantized audio latents

        # Tactile backbone
        zt_full = self.T_ENC(tc_1T)  # [B,C,T_lat]
        B, C, T_lat = zt_full.shape

        # Create random packet-loss mask on latent tokens
        mask_tokens = make_token_loss_mask(
            batch_size=B,
            T_lat=T_lat,
            packet_tok=PACKET_TOK,
            p_loss=PACKET_LOSS_PROB,
            device=zt_full.device
        )  # [B,T_lat] bool
        mask_tokens_b1T = mask_tokens.unsqueeze(1)  # [B,1,T_lat]

        # Zero-out masked tactile tokens (what receiver "sees")
        zt_in = zt_full * (~mask_tokens_b1T)  # [B,C,T_lat]

        # Cross-predict all tokens from audio + (partially missing) tactile
        z_pred = self.predict(zt_in, qa)  # [B,C,T_lat]

        # Fill in missing tokens with predictions, keep received ones unchanged
        z_filled = torch.where(mask_tokens_b1T, z_pred, zt_in)  # [B,C,T_lat]

        # Decode tactile waveform
        y_hat = self.T_DEC(z_filled)  # [B,1,T_out]
        T = min(y_hat.shape[-1], tc_1T.shape[-1], Tw)
        y_hat = y_hat[..., :T]
        tgt   = tc_1T[..., :T]

        return {
            "y_hat": finite_or_zero(y_hat),
            "tgt":   finite_or_zero(tgt),
            "latent_mask": mask_tokens_b1T
        }

# ================== BUILDERS ==================
def build_backbones():
    """
    Load DAC 24 kHz for audio & tactile (separate instances),
    and return their encoder/quantizer/decoder plus latent dims.
    """
    dac_audio = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    dac_tact  = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()

    A_ENC, A_QUANT = dac_audio.encoder, dac_audio.quantizer
    T_ENC, T_DEC   = dac_tact.encoder,  dac_tact.decoder

    dummy = torch.randn(1,1,TARGET_SR, device=DEVICE)
    with autocast('cuda', enabled=False):
        za = A_ENC(dummy)
        C  = za.size(1)
        toks = za.size(-1)

    print(f"[Latents] C={C}, tokens/sec≈{toks}")
    return A_ENC, A_QUANT, T_ENC, T_DEC, C, int(toks)

def split_items():
    items = pair_stems(AUDIO_DIR, TACT_DIR)
    random.shuffle(items)
    n_val = max(1, int(len(items) * VAL_FRAC))
    val_items   = items[:n_val][:MAX_VAL]
    train_items = items[n_val:]
    return train_items, val_items

# ================== TRAIN ==================
def train():
    set_seed(SEED)
    run_dir = os.path.join(OUT_ROOT, "plc_run")
    os.makedirs(run_dir, exist_ok=True)

    # Data
    train_items, val_items = split_items()
    train_dl = DataLoader(
        SegDataset(train_items),
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_dl = DataLoader(
        SegDataset(val_items),
        batch_size=BATCH,
        shuffle=False,
        num_workers=max(0, NUM_WORKERS // 2),
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn
    )

    # Backbones & model
    A_ENC, A_QUANT, T_ENC, T_DEC, C, tokens_per_sec = build_backbones()
    net = AllPredPLC(A_ENC, A_QUANT, T_ENC, T_DEC, c_lat=C).to(DEVICE)

    params = [p for p in net.parameters() if p.requires_grad]
    opt   = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)
    scaler = GradScaler('cuda', enabled=USE_AMP)

    best_val = float("inf")
    best_ep  = -1

    hist = {
        "train": [],
        "val":   [],
        "l1":    [],
        "stft":  [],
        "mel":   [],
    }

    def step(a, tc, train_mode=True):
        a  = a.to(DEVICE)
        tc = tc.to(DEVICE)
        with autocast('cuda', enabled=USE_AMP):
            out = net.forward_step(a, tc)
            y   = out["y_hat"]
            tgt = out["tgt"]

            l1   = safe_l1(y, tgt)
            lstf = MRSTFT(y, tgt)
            lmel = MELCOS(y, tgt)

            total = W_WAV_L1 * l1 + W_STFT * lstf + W_MELCOS * lmel

        if train_mode and torch.isfinite(total):
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

        return (
            float(total.detach().cpu()),
            float(l1.detach().cpu()),
            float(lstf.detach().cpu()),
            float(lmel.detach().cpu()),
        )

    for ep in range(1, EPOCHS + 1):
        # -------- Train --------
        net.train()
        t_sum = l1_sum = st_sum = me_sum = 0.0
        n = 0
        for a, tc in train_dl:
            Ttot, L1, ST, ME = step(a, tc, train_mode=True)
            n += 1
            t_sum  += Ttot
            l1_sum += L1
            st_sum += ST
            me_sum += ME

        t_avg = t_sum / max(1, n)
        hist["train"].append(t_avg)
        hist["l1"].append(l1_sum / max(1, n))
        hist["stft"].append(st_sum / max(1, n))
        hist["mel"].append(me_sum / max(1, n))

        # -------- Val --------
        net.eval()
        vs = 0.0
        vm = 0
        with torch.no_grad():
            for a, tc in val_dl:
                Vtot, _, _, _ = step(a, tc, train_mode=False)
                vs += Vtot
                vm += 1
        v = vs / max(1, vm)
        hist["val"].append(v)
        sched.step()

        print(f"[PLC] Ep {ep:03d} | train {t_avg:.4f} | val {v:.4f} | "
              f"L1 {hist['l1'][-1]:.4f} | STFT {hist['stft'][-1]:.4f} | MEL {hist['mel'][-1]:.4f}")

        # Save last
        torch.save(
            {
                "model": net.state_dict(),
                "epoch": ep,
                "hist": hist,
                "tokens_per_sec": tokens_per_sec,
                "packet_tok": PACKET_TOK,
                "packet_loss_prob": PACKET_LOSS_PROB,
            },
            os.path.join(run_dir, "last.pth"),
        )

        # Save best (after a few warmup epochs)
        if v + 1e-6 < best_val and ep > 6:
            best_val = v
            best_ep  = ep
            torch.save(
                {
                    "model": net.state_dict(),
                    "epoch": ep,
                    "hist": hist,
                    "tokens_per_sec": tokens_per_sec,
                    "packet_tok": PACKET_TOK,
                    "packet_loss_prob": PACKET_LOSS_PROB,
                },
                os.path.join(run_dir, "best.pth"),
            )
            print("✅ saved best")

    # -------- Curves --------
    plt.figure(figsize=(11, 5))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"],   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.title(f"Audio→Tactile PLC (tokens/sec≈{tokens_per_sec})")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "curves.png"))
    plt.close()

    with open(os.path.join(run_dir, "hist.json"), "w") as f:
        json.dump(hist, f, indent=2)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(
            {
                "tokens_per_sec": tokens_per_sec,
                "packet_tok": PACKET_TOK,
                "packet_loss_prob": PACKET_LOSS_PROB,
                "best_val": best_val,
                "best_epoch": best_ep,
            },
            f,
            indent=2,
        )

    print("\n✅ Training complete.")
    print(f"Best val loss: {best_val:.4f} at epoch {best_ep}")

# ================== MAIN ==================
if __name__ == "__main__":
    train()
