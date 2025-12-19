#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate & Compare:
  • Proposed AR+RVQ models from a sweep (auto-detect rvqB{books}_K{embed}/best.pth)
  • DAC 24kHz pretrained at multiple n_quantizers (rate scalable)
  • VC-PWQ (using decoded and compressed files from VC-PWQ codec)

Metrics (for DAC / Proposed / VC-PWQ):
  - ST-SIM (mel-cosine in [0,1]) at 24 kHz
  - PSNR_3k_aligned (dB): align at 24 kHz, then both → 3 kHz, peak=1.0

Bitrate:
  - Proposed: kbps = tokens/sec (24k DAC encoder) * books_used * log2(embed) / 1000
  - DAC 24k:  kbps = tokens/sec * n_q * log2(codebook_size) / 1000
  - VC-PWQ:   kbps = total_compressed_bits / total_signal_duration / 1000

Compression ratio:
  - DAC / Proposed: CR = 48 / kbps  (vs 3 kHz 16-bit PCM, 48 kbps)
  - VC-PWQ:        CR = total_orig_bytes / total_comp_bytes (empirical, same as older VC script)

Outputs:
  - combined JSON summary (DAC, Proposed, VC-PWQ)
  - Plots (all 3 models):
      • STSIM vs Bitrate
      • PSNR vs Bitrate
      • STSIM vs Compression Ratio
      • Compression Ratio vs Bitrate
"""

import os, json, glob, math, random, warnings
from pathlib import Path
warnings.filterwarnings("once", category=UserWarning)

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

import dac  # pip install descript-audio-codec

# ===================== USER CONFIG =====================

# AUDIO_DIR: audio modality (24 kHz or other, resampled to EVAL_SR internally)
# TACT_DIR:  vibrotactile modality (original 3 kHz signals)
AUDIO_DIR   = r"/home/student/studentdata/WAV_Files_raw"
TACT_DIR    = r"/home/student/studentdata/Vibrotactile_Files_Raw"

# Root of your training sweep (as produced by the sweep script)
SWEEP_ROOT  = r"/home/student/studentdata/SWEEP_ALLPRED_AR_RVQ"

OUT_DIR     = os.path.join(SWEEP_ROOT, "eval_vs_dac24_with_vcpwq")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_SR  = 24000
ORIG_3K  = 3000
SEG_SEC  = 1.0
BATCH    = 6
SEED     = 7
NUM_WORK = min(4, os.cpu_count() or 1)
PIN_MEM  = torch.cuda.is_available()
USE_AMP  = True

# DAC 24k settings
DAC_MODEL_TYPE = "24khz"
DAC_NQ_LIST    = [1, 2, 3, 4, 8]

# 3 kHz mono 16-bit PCM baseline
PCM_KBPS_TACT_ORIG = ORIG_3K * 16.0 / 1000.0  # = 48 kbps

# Alignment (± samples @ 24 kHz)
ALIGN_MAX_SHIFT_SAMPLES = 200

# Plot Y-range for ST-SIM
Y_STSIM_MIN, Y_STSIM_MAX = 0.80, 1.00

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ===================== VC-PWQ CONFIG =====================
# These should point to the outputs of your VC-PWQ C++ program
VC_DEC_DIR  = r"/home/student/studentdata/dac_eval_3khz_fixed/VC-PWQ/build/source/testprogram/data_decoded"
VC_COMP_DIR = r"/home/student/studentdata/dac_eval_3khz_fixed/VC-PWQ/build/source/testprogram/data_compressed"

# List of VC-PWQ configs you actually encoded/decoded
# Labels are just for the plot legend; only 'b' is used to find files.
#VC_CONFIGS = [
#    {"label": "VC bl32 b60", "b": 60},
#    {"label": "VC bl32 b46", "b": 46},
#    {"label": "VC bl32 b29", "b": 29},
#    {"label": "VC bl32 b17", "b": 17},
#    {"label": "VC bl32 b10", "b": 10},
#    {"label": "VC bl64 b73", "b": 73},  # if you don't have this, it'll just warn
#    {"label": "VC bl64 b57", "b": 57},
#    {"label": "VC bl64 b41", "b": 41},
#    {"label": "VC bl64 b27", "b": 27},
#    {"label": "VC bl64 b18", "b": 18},
#]

VC_CONFIGS = [
    {"label": "VC bl512 b8", "b": 8},
    {"label": "VC bl512 b16", "b": 16},
    {"label": "VC bl512 b24", "b": 24},
    {"label": "VC bl512 b48", "b": 48},
    {"label": "VC bl512 b64", "b": 64},
    {"label": "VC bl512 b80", "b": 80},  # if you don't have this, it'll just warn
    {"label": "VC bl512 b100", "b": 100},
    {"label": "VC bl512 b120", "b": 120},

]

# ===================== DATA =====================

def list_wavs(dirpath):
    return {Path(p).stem: p for p in glob.glob(os.path.join(dirpath, "*.wav"))}

def load_wav_sf(path):
    data, sr = sf.read(path, always_2d=True)
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    wav = torch.from_numpy(data).t().contiguous()  # [C,T]
    return wav, int(sr)

def sanitize_wave(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.9999, neginf=-0.9999).clamp(-1, 1)

def resample_f32(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    x = x.to(torch.float32)
    with torch.cuda.amp.autocast(enabled=False):
        res = torchaudio.transforms.Resample(orig_freq=sr_in, new_freq=sr_out).to(x.device)
        return res(x.contiguous())

class SegDataset(Dataset):
    def __init__(self, sdir, tdir, out_sr=EVAL_SR, seg_sec=SEG_SEC, seed=SEED):
        self.out_sr = out_sr
        self.seg = int(seg_sec * out_sr)
        self.rng = random.Random(seed)
        A = list_wavs(sdir); T = list_wavs(tdir)
        stems = sorted(set(A) & set(T))
        self.items = [(A[s], T[s], s) for s in stems]
        print(f"[Dataset] pairs={len(self.items)} | seg={self.seg} @ {out_sr} Hz")

    def __len__(self): return len(self.items)

    def _prep(self, path):
        w, sr = load_wav_sf(path)
        w = resample_f32(w, sr, self.out_sr)[:1, :]
        return sanitize_wave(w)

    def __getitem__(self, i):
        ap, tp, name = self.items[i]
        a = self._prep(ap)
        t = self._prep(tp)
        L = min(a.size(-1), t.size(-1))
        a, t = a[..., :L], t[..., :L]
        if L < self.seg:
            need = self.seg - L
            while need > 0:
                T_ = t.size(-1)
                step = min(need, max(1, T_-1))
                t = F.pad(t, (0, step), mode='reflect')
                need -= step
        st = self.rng.randint(0, max(0, t.size(-1) - self.seg)) if t.size(-1) > self.seg else 0
        t = t[:, st:st+self.seg]
        return a.squeeze(0), t.squeeze(0), name

def collate_fn(batch):
    A = torch.stack([b[0] for b in batch]).unsqueeze(1)
    T = torch.stack([b[1] for b in batch]).unsqueeze(1)
    names = [b[2] for b in batch]
    return sanitize_wave(A), sanitize_wave(T), names

# ===================== METRICS =====================
_MEL_CACHE = {}

def _mel_mag(x_1T, sr=EVAL_SR, n_fft=512, hop=128, n_mels=64):
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
        M = M / M.amax(dim=(1,2), keepdim=True).clamp_min_(1e-8)
    return M

@torch.no_grad()
def stsim_batch(ref_1T, est_1T):
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

@torch.no_grad()
def psnr_batch(ref_1T, est_1T, eps=1e-12):
    """PSNR(dB), peak=1.0."""
    ref = ref_1T.to(torch.float32); est = est_1T.to(torch.float32)
    mse = (ref - est).pow(2).mean(dim=(1,2)).clamp_min(eps)  # [B]
    psnr = 10.0 * torch.log10(1.0 / mse)
    return [float(v.item()) for v in psnr]

# ---------- alignment ----------
def align_pair_24k(ref_24, est_24, max_shift=ALIGN_MAX_SHIFT_SAMPLES):
    r = ref_24.squeeze(0).squeeze(0)
    e = est_24.squeeze(0).squeeze(0)
    r_f = r.to(torch.float32); e_f = e.to(torch.float32)
    best_shift = 0; best_corr = -1e18
    for s in range(-max_shift, max_shift + 1):
        if s < 0:
            r_seg = r_f[-s:]; e_seg = e_f[: r_seg.numel()]
        elif s > 0:
            r_seg = r_f[:-s]; e_seg = e_f[s : s + r_seg.numel()]
        else:
            r_seg = r_f;      e_seg = e_f[: r_seg.numel()]
        if r_seg.numel()==0 or e_seg.numel()==0: continue
        c = torch.sum(r_seg * e_seg)
        if c > best_corr: best_corr, best_shift = c, s
    s = best_shift
    if s < 0:
        r_a = r[-s:]; e_a = e[: r_a.numel()]
    elif s > 0:
        r_a = r[:-s]; e_a = e[s : s + r_a.numel()]
    else:
        r_a = r;      e_a = e[: r.numel()]
    return r_a.unsqueeze(0).unsqueeze(0), e_a.unsqueeze(0).unsqueeze(0), best_shift

@torch.no_grad()
def psnr_3k_aligned_batch(ref_24, est_24):
    B = ref_24.size(0)
    vals = []
    for b in range(B):
        r24 = ref_24[b:b+1]
        e24 = est_24[b:b+1]
        r_al, e_al, _ = align_pair_24k(r24, e24)
        r_3k = resample_f32(r_al, EVAL_SR, ORIG_3K)
        e_3k = resample_f32(e_al, EVAL_SR, ORIG_3K)
        vals += psnr_batch(r_3k, e_3k)
    return vals

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
    if (n_books is not None) and (bins is not None): return n_books, bins
    books = 0
    for n, p in quantizer.named_parameters():
        if "codebook" in n.lower() or "embed" in n.lower():
            if p.dim()==2:
                books += 1
                bins = p.size(0) if bins is None else max(bins, p.size(0))
    if n_books is None: n_books = books if books>0 else 8
    if bins    is None: bins    = 1024
    return int(n_books), int(bins)

@torch.no_grad()
def eval_dac24(loader: DataLoader, n_q_list):
    mdl = dac.DAC.load(dac.utils.download(DAC_MODEL_TYPE)).to(DEVICE).eval()
    native_sr = 24000
    tps = probe_tokens_per_sec(mdl, native_sr)
    _, bins = get_n_books_and_bins(mdl.quantizer)
    bits_per_code = math.log2(bins)
    out = {}
    for n_q in n_q_list:
        print(f"[DAC24] n_q={n_q}")
        st_vals, ps_vals = [], []
        for _, t, _ in loader:
            t = t.to(DEVICE)
            t_native = resample_f32(t, EVAL_SR, native_sr)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                z, *_ = mdl.encode(t_native, n_quantizers=int(n_q))
                y_native = mdl.decode(z)
            y_24 = resample_f32(y_native, native_sr, EVAL_SR)
            Tlen = min(t.shape[-1], y_24.shape[-1])
            t_24 = t[..., :Tlen]; y_24 = y_24[..., :Tlen]
            st_vals += stsim_batch(t_24, y_24)
            ps_vals += psnr_3k_aligned_batch(t_24, y_24)
        arr_s = np.array(st_vals, dtype=np.float64)
        arr_p = np.array(ps_vals, dtype=np.float64)
        n = int(arr_s.size)
        st_m = float(arr_s.mean()); st_ci = 1.96*float(arr_s.std(ddof=0))/max(1, math.sqrt(n))
        ps_m = float(arr_p.mean()); ps_ci = 1.96*float(arr_p.std(ddof=0))/max(1, math.sqrt(n))
        kbps = (tps * n_q * bits_per_code) / 1000.0
        cr   = PCM_KBPS_TACT_ORIG / kbps if kbps > 0 else float('inf')
        out[int(n_q)] = {"stsim_mean":st_m,"stsim_ci95":st_ci,
                         "psnr_mean":ps_m,"psnr_ci95":ps_ci,
                         "kbps":float(kbps),"compression_ratio":float(cr),
                         "n":n,"tps":float(tps),"bins":int(bins)}
    return out

# ===================== Proposed Model (Eval wrapper) =====================
CODE_DIM = 96
AR_CHUNK_TOK = 16

class PosEnc1D(nn.Module):
    def __init__(self, c, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, c)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, c, 2) * (-math.log(10000.0)/c))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x): T=x.size(-1); return x + self.pe[:T,:].T.unsqueeze(0).to(x.dtype)

class TokenNorm(nn.Module):
    def __init__(self, c): super().__init__(); self.ln = nn.LayerNorm(c)
    def forward(self, z): zt=z.permute(0,2,1); zt=self.ln(zt); return zt.permute(0,2,1)

class CrossPredictor(nn.Module):
    def __init__(self, c, heads=8, mlp_mul=2, dropout=0.1):
        super().__init__()
        self.pos = PosEnc1D(c); self.h=heads; self.dh=c//heads; assert c%heads==0
        self.ln_q=nn.LayerNorm(c); self.ln_kv=nn.LayerNorm(c)
        self.q_proj=nn.Linear(c,c,False); self.k_proj=nn.Linear(c,c,False); self.v_proj=nn.Linear(c,c,False)
        self.out=nn.Linear(c,c,False); self.drop=nn.Dropout(dropout)
        self.ffn=nn.Sequential(nn.LayerNorm(c), nn.Linear(c,mlp_mul*c), nn.GELU(), nn.Linear(mlp_mul*c,c))
    def _split(self,x): B,T,C=x.shape; return x.view(B,T,self.h,self.dh).permute(0,2,1,3)
    def _merge(self,x): B,H,T,D=x.shape; return x.permute(0,2,1,3).contiguous().view(B,T,H*D)
    def forward(self, zt_prev, za):
        q=self.pos(zt_prev).permute(0,2,1); kv=self.pos(za).permute(0,2,1)
        q=self.ln_q(q); kv=self.ln_kv(kv)
        Q=self._split(self.q_proj(q)); K=self._split(self.k_proj(kv)); V=self._split(self.v_proj(kv))
        attn=(Q @ K.transpose(-2,-1))/math.sqrt(self.dh)
        ctx=(attn.softmax(dim=-1) @ V)
        y=self.out(self.drop(self._merge(ctx))); y=y+q; y=y+self.ffn(y)
        return y.permute(0,2,1)

class ResidualVQEMA(nn.Module):
    def __init__(self, dim: int, n_books: int, n_embed: int):
        super().__init__()
        self.books = nn.ParameterList([nn.Parameter(torch.randn(n_embed, dim)/math.sqrt(dim))
                                       for _ in range(n_books)])
    @staticmethod
    def _nearest_l2(x, emb):
        return (x @ emb.t() - 0.5*(emb*emb).sum(dim=1).unsqueeze(0)).argmax(dim=1)
    def forward(self, z, n_books_use=None):
        if n_books_use is None: n_books_use = len(self.books)
        n_books_use = min(n_books_use, len(self.books))
        B,D,T=z.shape; x=z.permute(0,2,1).reshape(B*T, D)
        residual=x; q_sum=torch.zeros_like(x)
        for cb in self.books[:n_books_use]:
            emb=cb.detach().to(z.dtype).to(z.device)
            idx=self._nearest_l2(residual, emb)
            q=F.embedding(idx, emb)
            q_sum = q_sum + (q - residual).detach() + residual
            residual = residual - q
        return q_sum.view(B,T,D).permute(0,2,1).contiguous()

class ProposedEval(nn.Module):
    def __init__(self, A_ENC, A_QUANT, T_ENC, T_DEC, c_lat, rvq_books, rvq_embed):
        super().__init__()
        self.A_ENC, self.A_QUANT, self.T_ENC, self.T_DEC = A_ENC, A_QUANT, T_ENC, T_DEC
        for m in [self.A_ENC, self.A_QUANT, self.T_ENC, self.T_DEC]:
            for p in m.parameters(): p.requires_grad_(False)
        self.predict   = CrossPredictor(c=c_lat)
        self.tokennorm = TokenNorm(c_lat)
        self.scale     = nn.Parameter(torch.tensor(0.08))
        self.proj_down = nn.Conv1d(c_lat, CODE_DIM, 1)
        self.proj_up   = nn.Conv1d(CODE_DIM, c_lat, 1)
        self.vq        = ResidualVQEMA(dim=CODE_DIM, n_books=rvq_books, n_embed=rvq_embed)

    @torch.no_grad()
    def forward_eval(self, a_1T, t_1T, books_use=None):
        za = self.A_ENC(a_1T); qa, *_ = self.A_QUANT(za)
        zt = self.T_ENC(t_1T); B,C,Tlat = zt.shape
        z_run = torch.zeros_like(zt)
        for s in range(0, Tlat, AR_CHUNK_TOK):
            e = min(Tlat, s+AR_CHUNK_TOK)
            zt_prev = torch.zeros(B,C,e-s, device=zt.device, dtype=zt.dtype)
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
            z_hat    = z_pred + self.proj_up(qD)
            z_run[..., s:e] = z_hat
        y = self.T_DEC(z_run)
        return y

def build_backbones_for_eval():
    da = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    dt = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    A_ENC, A_QUANT = da.encoder, da.quantizer
    T_ENC, T_DEC   = dt.encoder, dt.decoder
    dummy = torch.zeros(1,1,EVAL_SR, device=DEVICE)
    with torch.cuda.amp.autocast(enabled=False):
        C = A_ENC(dummy).size(1)
    return A_ENC, A_QUANT, T_ENC, T_DEC, C

@torch.no_grad()
def eval_proposed_runs(loader: DataLoader, sweep_root: str):
    # discover runs like rvqB{books}_K{embed}
    candidates = sorted([p for p in glob.glob(os.path.join(sweep_root, "rvqB*_K*")) if os.path.isdir(p)])
    if not candidates:
        raise RuntimeError(f"No runs found under {sweep_root} (expected rvqB*_K*/).")
    A_ENC,A_QUANT,T_ENC,T_DEC,C = build_backbones_for_eval()
    # tokens/sec from DAC 24k encoder at EVAL_SR
    da24 = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    tps = probe_tokens_per_sec(da24, EVAL_SR)
    results = []
    for run in candidates:
        meta_path = os.path.join(run, "meta.json")
        best_path = os.path.join(run, "best.pth")
        if not os.path.isfile(best_path):
            print(f"[Skip] no best.pth in {run}")
            continue
        # parse books/embed from path if meta missing
        rvq_books = None; rvq_embed = None
        stem = Path(run).name
        # try meta.json first
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                rvq_books = int(meta.get("rvq_books", 0)) or None
                rvq_embed = int(meta.get("rvq_embed", 0)) or None
            except Exception:
                pass
        if rvq_books is None or rvq_embed is None:
            # fallback parse
            try:
                partB = stem.split("_")[0]; partK = stem.split("_")[1]
                rvq_books = int(partB.replace("rvqB",""))
                rvq_embed = int(partK.replace("K",""))
            except Exception:
                raise RuntimeError(f"Cannot infer (books, embed) for run: {run}")

        print(f"[Proposed] {stem} | books={rvq_books}, embed={rvq_embed}")
        # build eval model with correct RVQ shape
        net = ProposedEval(A_ENC, A_QUANT, T_ENC, T_DEC, c_lat=C,
                           rvq_books=rvq_books, rvq_embed=rvq_embed).to(DEVICE)
        sd = torch.load(best_path, map_location="cpu")
        missing, unexpected = net.load_state_dict(sd["model"], strict=False)
        if missing or unexpected:
            print(f"  (state-dict mismatch tolerated) missing={len(missing)} unexpected={len(unexpected)}")
        net.eval()

        st_vals, ps_vals = [], []
        for a, t, _ in loader:
            a = a.to(DEVICE); t = t.to(DEVICE)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                y = net.forward_eval(a, t, books_use=rvq_books)  # use ALL books trained
            Tlen = min(t.shape[-1], y.shape[-1])
            t_24 = t[..., :Tlen]; y_24 = y[..., :Tlen]
            st_vals += stsim_batch(t_24, y_24)
            ps_vals += psnr_3k_aligned_batch(t_24, y_24)

        arr_s = np.array(st_vals, dtype=np.float64)
        arr_p = np.array(ps_vals, dtype=np.float64)
        n     = int(arr_s.size)
        st_m, st_ci = float(arr_s.mean()), 1.96*float(arr_s.std(ddof=0))/max(1, math.sqrt(n))
        ps_m, ps_ci = float(arr_p.mean()), 1.96*float(arr_p.std(ddof=0))/max(1, math.sqrt(n))
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
            "psnr_mean": ps_m, "psnr_ci95": ps_ci
        })
    return results

# ===================== VC-PWQ EVAL =====================
@torch.no_grad()
def eval_vc_pwq(vc_dec_dir, vc_comp_dir, tact_dir, vc_configs):
    """
    Evaluate VC-PWQ on the tactile files used in the DAC/proposed script.

    For each config (with bit-budget b):
      - match original tactile:  TACT_DIR/stem.wav
      - match VC decoded:        VC_DEC_DIR/*stem*_{b}.wav
      - match VC compressed:     VC_COMP_DIR/*stem*_{b}.binary

    Metrics:
      - STSIM (mel-cosine at 24 kHz)
      - PSNR_3k_aligned (like DAC/proposed)

    Bitrate:
      - kbps = total_compressed_bits / total_duration / 1000

    Compression ratio (to match old VC-only script):
      - CR = total_orig_bytes / total_comp_bytes
        (i.e. uncompressed WAV size vs compressed binary size)
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

            # load and resample to 24 kHz
            t_wav, t_sr = load_wav_sf(t_path)
            y_wav, y_sr = load_wav_sf(dec_path)
            t_wav = t_wav[:1, :]
            y_wav = y_wav[:1, :]

            t_24 = resample_f32(t_wav, t_sr, EVAL_SR)
            y_24 = resample_f32(y_wav, y_sr, EVAL_SR)
            t_24 = sanitize_wave(t_24)
            y_24 = sanitize_wave(y_24)

            Tlen = min(t_24.shape[-1], y_24.shape[-1])
            if Tlen <= 0:
                continue
            t_24c = t_24[..., :Tlen]
            y_24c = y_24[..., :Tlen]

            # metrics
            st_vals += stsim_batch(t_24c, y_24c)
            ps_vals += psnr_3k_aligned_batch(t_24c, y_24c)

            # compressed file size
            comp_pattern = os.path.join(vc_comp_dir, f"*{t_name}*_{b_val}.binary")
            comp_candidates = glob.glob(comp_pattern)
            if comp_candidates:
                comp_path = sorted(comp_candidates)[0]
                comp_bytes = os.path.getsize(comp_path)
                total_comp_bytes += comp_bytes

                # original size (WAV on disk)
                orig_bytes = os.path.getsize(t_path)
                total_orig_bytes += orig_bytes

                # duration from original tactile (3 kHz)
                data_t, sr_t2 = sf.read(t_path)
                if data_t.ndim > 1:
                    n_samples = data_t.shape[0]
                else:
                    n_samples = len(data_t)
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
        st_ci = 1.96*float(arr_s.std(ddof=0))/max(1, math.sqrt(n))
        ps_m = float(arr_p.mean())
        ps_ci = 1.96*float(arr_p.std(ddof=0))/max(1, math.sqrt(n))

        if total_comp_bytes > 0 and total_time_sec > 0:
            total_comp_bits = total_comp_bytes * 8.0
            bitrate_bps = total_comp_bits / total_time_sec
            kbps = bitrate_bps / 1000.0

            # CR from file sizes (matching earlier VC-only script)
            cr = (total_orig_bytes * 1.0) / (total_comp_bytes * 1.0)
        else:
            kbps = float("nan")
            cr   = float("nan")

        print(f"  pairs={used_pairs}, nseg={n}")
        print(f"  STSIM_mean={st_m:.4f}, STSIM_CI={st_ci:.4f}")
        print(f"  PSNR_mean ={ps_m:.2f} dB, PSNR_CI={ps_ci:.2f}")
        print(f"  kbps={kbps:.3f}, CR={cr:.2f}x")

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

def _errfill(x, y, ci, label, marker):
    x = np.asarray(x); y = np.asarray(y); ci = np.asarray(ci)
    plt.plot(x, y, marker + "--", lw=2.0, ms=0, label=label)
    plt.scatter(x, y, s=36, zorder=3)
    plt.fill_between(x, y-ci, y+ci, alpha=0.20)

# ===================== MAIN =====================
def main():
    # Dataset for DAC & Proposed
    ds = SegDataset(AUDIO_DIR, TACT_DIR, out_sr=EVAL_SR, seg_sec=SEG_SEC, seed=SEED)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORK,
                    pin_memory=PIN_MEM, collate_fn=collate_fn)

    # DAC 24k
    dac24 = eval_dac24(dl, DAC_NQ_LIST)

    # Proposed sweep
    proposed_rows = eval_proposed_runs(dl, SWEEP_ROOT)

    # VC-PWQ
    vc_rows = eval_vc_pwq(VC_DEC_DIR, VC_COMP_DIR, TACT_DIR, VC_CONFIGS)

    # Save combined JSON
    combined = {
        "dac_24khz": dac24,
        "proposed_runs": proposed_rows,
        "vc_pwq_runs": vc_rows,
        "config": {
            "eval_sr": EVAL_SR,
            "orig_tact_sr": ORIG_3K,
            "pcm_kbps_tact_orig": PCM_KBPS_TACT_ORIG,
            "dac_nq_list": DAC_NQ_LIST,
            "align_max_shift_samples": ALIGN_MAX_SHIFT_SAMPLES,
            "sweep_root": SWEEP_ROOT,
            "vc_dec_dir": VC_DEC_DIR,
            "vc_comp_dir": VC_COMP_DIR,
        }
    }
    out_json = os.path.join(OUT_DIR, "eval_all_vs_dac24_vcpwq.json")
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved JSON → {out_json}")

    # ---------- Prepare arrays ----------
    # DAC
    nqs = sorted(dac24.keys(), key=lambda q: dac24[q]["kbps"])
    br_d = np.array([dac24[q]["kbps"] for q in nqs], dtype=float)
    ss_d = np.array([dac24[q]["stsim_mean"] for q in nqs], dtype=float)
    sp_d = np.array([dac24[q]["psnr_mean"] for q in nqs], dtype=float)
    cr_d = np.array([dac24[q]["compression_ratio"] for q in nqs], dtype=float)
    cs_d = np.array([dac24[q]["stsim_ci95"] for q in nqs], dtype=float)
    cp_d = np.array([dac24[q]["psnr_ci95"] for q in nqs], dtype=float)

    # Proposed
    groups = _group_by_embed(proposed_rows)
    markers = {128:"o", 256:"^", 512:"D"}

    # VC-PWQ
    br_v = np.array([r["kbps"] for r in vc_rows], dtype=float)
    ss_v = np.array([r["stsim_mean"] for r in vc_rows], dtype=float)
    sp_v = np.array([r["psnr_mean"] for r in vc_rows], dtype=float)
    cr_v = np.array([r["compression_ratio"] for r in vc_rows], dtype=float)
    cs_v = np.array([r["stsim_ci95"] for r in vc_rows], dtype=float)
    cp_v = np.array([r["psnr_ci95"] for r in vc_rows], dtype=float)
    lbl_v = [r["label"] for r in vc_rows]

    all_br = list(br_d) + [r["kbps"] for r in proposed_rows] + list(br_v)
    xmin = 0.8 * min(all_br)
    xmax = max(1.2 * max(all_br), 15.0)

    # ---------- PLOT 1: ST-SIM vs Bitrate ----------
    plt.figure(figsize=(7.2, 5.2))

    for embed, rows in sorted(groups.items()):
        x = [r["kbps"] for r in rows]
        y = [r["stsim_mean"] for r in rows]
        ci = [r["stsim_ci95"] for r in rows]
        _errfill(x, y, ci, label=f"Proposed (K={embed})", marker=markers.get(embed, "o"))

    for i, nq in enumerate(nqs):
        plt.errorbar([br_d[i]], [ss_d[i]], yerr=[cs_d[i]],
                     fmt="s", ms=7, lw=1.6, label=f"DAC 24k (n_q={nq})")

    for i, lab in enumerate(lbl_v):
        plt.errorbar([br_v[i]], [ss_v[i]], yerr=[cs_v[i]],
                     fmt="v", ms=7, lw=1.6, label=f"VC-PWQ ({lab})")

    plt.ylim(Y_STSIM_MIN, Y_STSIM_MAX)
    plt.xlim(0, xmax)
    plt.xlabel("Bitrate (kbps)")
    plt.ylabel("ST-SIM (↑)")
    plt.grid(True, alpha=0.3)
    plt.title("ST-SIM vs Bitrate — DAC vs Proposed vs VC-PWQ")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    f1 = os.path.join(OUT_DIR, "stsim_vs_bitrate_all.png")
    plt.savefig(f1, dpi=180); plt.close()
    print(f"Saved: {f1}")

    # ---------- PLOT 2: PSNR vs Bitrate ----------
    plt.figure(figsize=(7.2, 5.2))

    for embed, rows in sorted(groups.items()):
        x = [r["kbps"] for r in rows]
        y = [r["psnr_mean"] for r in rows]
        ci = [r["psnr_ci95"] for r in rows]
        _errfill(x, y, ci, label=f"Proposed (K={embed})", marker=markers.get(embed, "o"))

    for i, nq in enumerate(nqs):
        plt.errorbar([br_d[i]], [sp_d[i]], yerr=[cp_d[i]],
                     fmt="s", ms=7, lw=1.6, label=f"DAC 24k (n_q={nq})")

    for i, lab in enumerate(lbl_v):
        plt.errorbar([br_v[i]], [sp_v[i]], yerr=[cp_v[i]],
                     fmt="v", ms=7, lw=1.6, label=f"VC-PWQ ({lab})")

    plt.xlim(0, xmax)
    plt.xlabel("Bitrate (kbps)")
    plt.ylabel("PSNR (3 kHz, aligned) [dB, ↑]")
    plt.grid(True, alpha=0.3)
    plt.title("PSNR vs Bitrate — DAC vs Proposed vs VC-PWQ")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    f2 = os.path.join(OUT_DIR, "psnr_vs_bitrate_all.png")
    plt.savefig(f2, dpi=180); plt.close()
    print(f"Saved: {f2}")

    # ---------- PLOT 3: ST-SIM vs Compression Ratio ----------
    plt.figure(figsize=(7.2, 5.2))

    for embed, rows in sorted(groups.items()):
        x = [r["compression_ratio"] for r in rows]
        y = [r["stsim_mean"] for r in rows]
        ci = [r["stsim_ci95"] for r in rows]
        _errfill(x, y, ci, label=f"Proposed (K={embed})", marker=markers.get(embed, "o"))

    for i, nq in enumerate(nqs):
        plt.errorbar([cr_d[i]], [ss_d[i]], yerr=[cs_d[i]],
                     fmt="s", ms=7, lw=1.6, label=f"DAC 24k (n_q={nq})")

    for i, lab in enumerate(lbl_v):
        plt.errorbar([cr_v[i]], [ss_v[i]], yerr=[cs_v[i]],
                     fmt="v", ms=7, lw=1.6, label=f"VC-PWQ ({lab})")

    plt.xlabel("Compression Ratio")
    plt.ylabel("ST-SIM (↑)")
    plt.grid(True, alpha=0.3)
    plt.title("ST-SIM vs Compression Ratio — DAC vs Proposed vs VC-PWQ")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    f3 = os.path.join(OUT_DIR, "stsim_vs_cr_all.png")
    plt.savefig(f3, dpi=180); plt.close()
    print(f"Saved: {f3}")

    # ---------- PLOT 4: CR vs Bitrate ----------
    plt.figure(figsize=(7.2, 5.2))

    for embed, rows in sorted(groups.items()):
        x = [r["kbps"] for r in rows]
        y = [r["compression_ratio"] for r in rows]
        plt.plot(x, y, markers.get(embed, "o")+"--", lw=2.0, ms=0,
                 label=f"Proposed (K={embed})")
        plt.scatter(x, y, s=36, zorder=3)

    for i, nq in enumerate(nqs):
        plt.scatter([br_d[i]], [cr_d[i]], marker="s", s=64,
                    label=f"DAC 24k (n_q={nq})")

    for i, lab in enumerate(lbl_v):
        plt.scatter([br_v[i]], [cr_v[i]], marker="v", s=64,
                    label=f"VC-PWQ ({lab})")

    plt.xlabel("Bitrate (kbps, →)")
    plt.ylabel("Compression Ratio (↑)")
    plt.xlim(0, xmax)
    plt.grid(True, alpha=0.3)
    plt.title("Compression Ratio vs Bitrate — DAC vs Proposed vs VC-PWQ")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    f4 = os.path.join(OUT_DIR, "cr_vs_bitrate_all.png")
    plt.savefig(f4, dpi=180); plt.close()
    print(f"Saved: {f4}")

if __name__ == "__main__":
    main()
