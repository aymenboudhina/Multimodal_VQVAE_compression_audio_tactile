#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate & Compare:
  • Proposed AR+RVQ models from a sweep (auto-detect rvqB{books}_K{embed}/best.pth)
  • DAC 24kHz pretrained at multiple n_quantizers (rate scalable)

Metrics:
  - ST-SIM (mel-cosine in [0,1]) at 24 kHz
  - PSNR_3k_aligned (dB): align at 24 kHz, then both → 3 kHz, peak=1.0

Bitrate:
  - Proposed: kbps = tokens/sec (24k DAC encoder) * books_used * log2(embed) / 1000
  - DAC 24k:  kbps = tokens/sec * n_q * log2(codebook_size) / 1000

Compression ratio vs 3 kHz 16-bit PCM baseline (48 kbps):
  CR = 48 / kbps

Outputs:
  - combined JSON summary
  - 4 plots: STSIM vs Bitrate, PSNR vs Bitrate, STSIM vs CR, CR vs Bitrate
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
AUDIO_DIR   = r"/home/student/studentdata/WAV_Files_raw"
TACT_DIR    = r"/home/student/studentdata/Vibrotactile_Files_Raw"

# Root of your training sweep (as produced by the sweep script)
SWEEP_ROOT  = r"/home/student/studentdata/SWEEP_ALLPRED_AR_RVQ"

OUT_DIR     = os.path.join(SWEEP_ROOT, "eval_vs_dac24")
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
DAC_NQ_LIST    = [1,2,3, 4, 8]

# 3 kHz mono 16-bit PCM baseline
PCM_KBPS_TACT_ORIG = ORIG_3K * 16.0 / 1000.0  # = 48 kbps

# Alignment (± samples @ 24 kHz)
ALIGN_MAX_SHIFT_SAMPLES = 200

# Plot Y-range for ST-SIM
Y_STSIM_MIN, Y_STSIM_MAX = 0.80, 1.00

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

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

# ===================== EVAL LOOPS =====================
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
            # rvqB{B}_K{K}
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

# ===================== PLOTTING =====================
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
    ds = SegDataset(AUDIO_DIR, TACT_DIR, out_sr=EVAL_SR, seg_sec=SEG_SEC, seed=SEED)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORK,
                    pin_memory=PIN_MEM, collate_fn=collate_fn)

    # Evaluate DAC 24k
    dac24 = eval_dac24(dl, DAC_NQ_LIST)

    # Evaluate proposed sweep runs
    proposed_rows = eval_proposed_runs(dl, SWEEP_ROOT)

    # Save combined JSON
    combined = {
        "dac_24khz": dac24,
        "proposed_runs": proposed_rows,
        "config": {
            "eval_sr": EVAL_SR,
            "orig_tact_sr": ORIG_3K,
            "pcm_kbps_tact_orig": PCM_KBPS_TACT_ORIG,
            "dac_nq_list": DAC_NQ_LIST,
            "align_max_shift_samples": ALIGN_MAX_SHIFT_SAMPLES,
            "sweep_root": SWEEP_ROOT
        }
    }
    out_json = os.path.join(OUT_DIR, "eval_proposed_sweep_vs_dac24.json")
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved JSON → {out_json}")

    # ---------- Prepare arrays ----------
    # DAC arrays sorted by bitrate
    nqs = sorted(dac24.keys(), key=lambda q: dac24[q]["kbps"])
    br_d = np.array([dac24[q]["kbps"] for q in nqs], dtype=float)
    ss_d = np.array([dac24[q]["stsim_mean"] for q in nqs], dtype=float)
    sp_d = np.array([dac24[q]["psnr_mean"] for q in nqs], dtype=float)
    cr_d = np.array([dac24[q]["compression_ratio"] for q in nqs], dtype=float)
    cs_d = np.array([dac24[q]["stsim_ci95"] for q in nqs], dtype=float)
    cp_d = np.array([dac24[q]["psnr_ci95"] for q in nqs], dtype=float)

    # Proposed groups per embed size
    groups = _group_by_embed(proposed_rows)
    markers = {128:"o", 256:"^", 512:"D"}

    # ---------- PLOT 1: ST-SIM vs Bitrate ----------
    plt.figure(figsize=(7.2, 5.2))
    for embed, rows in sorted(groups.items()):
        x = [r["kbps"] for r in rows]
        y = [r["stsim_mean"] for r in rows]
        ci = [r["stsim_ci95"] for r in rows]
        _errfill(x, y, ci, label=f"Proposed (K={embed})", marker=markers.get(embed, "o"))

    # DAC points
    for i, nq in enumerate(nqs):
        plt.errorbar([br_d[i]], [ss_d[i]], yerr=[cs_d[i]], fmt="s", ms=7, lw=1.6,
                     label=f"DAC 24k (n_q={nq})")

    plt.ylim(Y_STSIM_MIN, Y_STSIM_MAX)
    all_br = list(br_d) + [r["kbps"] for r in proposed_rows]
    xmin = 0.8 * min(all_br); xmax = 1.2 * max(all_br)
    xmax=15
    plt.xlim(xmin, xmax)
    plt.xlabel("Bitrate (kbps)"); plt.ylabel("ST-SIM (↑)")
    plt.grid(True, alpha=0.3); plt.title("ST-SIM vs Bitrate — Proposed sweep vs DAC 24k")
    plt.legend(loc="lower right"); plt.tight_layout()
    f1 = os.path.join(OUT_DIR, "stsim_vs_bitrate.png"); plt.savefig(f1, dpi=180); plt.close()
    print(f"Saved: {f1}")

    # ---------- PLOT 2: PSNR (3 kHz aligned) vs Bitrate ----------
    plt.figure(figsize=(7.2, 5.2))
    for embed, rows in sorted(groups.items()):
        x = [r["kbps"] for r in rows]
        y = [r["psnr_mean"] for r in rows]
        ci = [r["psnr_ci95"] for r in rows]
        _errfill(x, y, ci, label=f"Proposed (K={embed})", marker=markers.get(embed, "o"))

    for i, nq in enumerate(nqs):
        plt.errorbar([br_d[i]], [sp_d[i]], yerr=[cp_d[i]], fmt="s", ms=7, lw=1.6,
                     label=f"DAC 24k (n_q={nq})")

    xmin = 0.8 * min(all_br); xmax = 1.2 * max(all_br)
    xmax=15
    plt.xlim(xmin, xmax)
    plt.xlabel("Bitrate (kbps)"); plt.ylabel("PSNR (3 kHz, aligned) [dB, ↑]")
    plt.grid(True, alpha=0.3); plt.title("PSNR vs Bitrate — Proposed sweep vs DAC 24k")
    plt.legend(loc="lower right"); plt.tight_layout()
    f2 = os.path.join(OUT_DIR, "psnr_vs_bitrate.png"); plt.savefig(f2, dpi=180); plt.close()
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
    plt.xlabel("Compression Ratio (48 kbps / model kbps, ↑)")
    plt.ylabel("ST-SIM (↑)")
    plt.grid(True, alpha=0.3); plt.title("ST-SIM vs Compression Ratio")
    plt.legend(loc="lower right"); plt.tight_layout()
    f3 = os.path.join(OUT_DIR, "stsim_vs_cr.png"); plt.savefig(f3, dpi=180); plt.close()
    print(f"Saved: {f3}")

    # ---------- PLOT 4: Compression Ratio vs Bitrate ----------
    plt.figure(figsize=(7.2, 5.2))
    for embed, rows in sorted(groups.items()):
        x = [r["kbps"] for r in rows]
        y = [r["compression_ratio"] for r in rows]
        plt.plot(x, y, markers.get(embed, "o")+"--", lw=2.0, ms=0, label=f"Proposed (K={embed})")
        plt.scatter(x, y, s=36, zorder=3)
    for i, nq in enumerate(nqs):
        plt.scatter([br_d[i]], [cr_d[i]], marker="s", s=64, label=f"DAC 24k (n_q={nq})")

    plt.xlabel("Bitrate (kbps, →)"); plt.ylabel("Compression Ratio (48 / kbps, ↑)")
    xmax=15
    plt.xlim(0, xmax)
    plt.grid(True, alpha=0.3); plt.title("Compression Ratio vs Bitrate")
    plt.legend(loc="upper right"); plt.tight_layout()
    f4 = os.path.join(OUT_DIR, "cr_vs_bitrate.png"); plt.savefig(f4, dpi=180); plt.close()
    print(f"Saved: {f4}")

if __name__ == "__main__":
    main()
