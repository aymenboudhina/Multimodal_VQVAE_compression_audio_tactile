#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sweep-train — AR A->T with Residual RVQ (vary n_books and n_embed ∈ {128,256,512})
-----------------------------------------------------------------------------------
For each (RVQ_N_BOOKS, RVQ_EMBED) combo:
  • Build model with that RVQ shape
  • Train/validate, save last/best checkpoints and curves
  • Log tokens/sec, bits/code, est. kbps and best val

Outputs:
  OUT_ROOT/
    rvqB{n_books}_K{n_embed}/
      last.pth, best.pth, curves.png, hist.json, meta.json
  sweep_summary.csv  (one line per combo)

Requires:
  pip install descript-audio-codec torch torchaudio soundfile matplotlib
"""

import os, math, glob, random, json, csv, warnings
from pathlib import Path
from typing import Tuple, List, Dict
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
OUT_ROOT   = r"/home/student/studentdata/SWEEP_ALLPRED_AR_RVQ"
os.makedirs(OUT_ROOT, exist_ok=True)

# ====================== TRAINING CONFIG ======================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR    = 24000
SEG_SEC      = 1.0
SEG          = int(SEG_SEC * TARGET_SR)

BATCH        = 6
EPOCHS       = 100         # tweak as needed
LR           = 2e-4
WD           = 1e-5
GRAD_CLIP    = 3.0
USE_AMP      = True
SEED         = 7

VAL_FRAC     = 0.2
MAX_VAL      = 300
NUM_WORKERS  = min(4, os.cpu_count() or 1)
PIN_MEMORY   = torch.cuda.is_available()

# Autoregressive latent roll
AR_CHUNK_TOK = 16

# Backbone / feature dims
CODE_DIM     = 96          # projection bottle-neck dim before RVQ (unchanged)
EMA_DECAY    = 0.99
EMA_WARM_E   = 5

# Loss weights
W_WAV_L1     = 0.55
W_STFT       = 0.25
W_MELCOS     = 0.20

# Mel/STFT config for losses
MEL_NFFT = 512
MEL_HOP  = 128
MEL_MELS = 64
EPS      = 1e-7

# ====================== SWEEP GRID ======================
# Try these book counts × embed sizes (bits per code = log2(embed)):
BOOKS_LIST  = [1, 2, 3, 4, 6, 8]
EMBED_LIST  = [128, 256, 512]     # 7, 8, 9 bits/code
SWEEP: List[Tuple[int, int]] = [(b, k) for b in BOOKS_LIST for k in EMBED_LIST]

# ================== UTILS ==================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def sanitize_wave(x: torch.Tensor, clamp=True):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.9999, neginf=-0.9999)
    return x.clamp(-1.0, 1.0) if clamp else x

def finite_or_zero(x: torch.Tensor):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def list_wavs(dirpath): return {Path(p).stem: p for p in glob.glob(os.path.join(dirpath, "*.wav"))}

def load_wav_sf(path):
    data, sr = sf.read(path, always_2d=True)
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    return torch.from_numpy(data).t().contiguous(), int(sr)  # [C,T], sr

def resample_to(wav, sr_in, sr_out):
    if sr_in == sr_out: return wav
    with autocast('cuda', enabled=False):
        return torchaudio.transforms.Resample(sr_in, sr_out).to(wav.device)(wav.to(torch.float32))

def reflect_pad_right_any(x: torch.Tensor, need: int) -> torch.Tensor:
    assert x.dim() == 2
    while need > 0:
        T = x.size(-1)
        if T <= 1:
            x = F.pad(x, (0, need), mode="replicate"); break
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
        self.items = items; self.sr = sr; self.seg = seg; self.rng = random.Random(seed)
        print(f"[Dataset] files: {len(items)} | seg={seg}")
    def __len__(self): return len(self.items)
    def _prep(self, p):
        w, sr = load_wav_sf(p); w = resample_to(w, sr, self.sr)[:1, :]
        return sanitize_wave(w)
    def __getitem__(self, i):
        ap, tp, _ = self.items[i]
        a = self._prep(ap); t = self._prep(tp)
        L = min(a.size(-1), t.size(-1)); a, t = a[..., :L], t[..., :L]
        if a.size(-1) < self.seg: a = reflect_pad_right_any(a, self.seg - a.size(-1))
        if t.size(-1) < self.seg: t = reflect_pad_right_any(t, self.seg - t.size(-1))
        st = self.rng.randint(0, max(0, a.size(-1) - self.seg)) if a.size(-1) > self.seg else 0
        return a[:, st:st+self.seg].squeeze(0), t[:, st:st+self.seg].squeeze(0)

def collate_fn(batch):
    A  = torch.stack([b[0] for b in batch]).unsqueeze(1)
    TC = torch.stack([b[1] for b in batch]).unsqueeze(1)
    return sanitize_wave(A), sanitize_wave(TC)

# ============== LOSSES (no in-place) ==============
class MultiResSTFTLoss(nn.Module):
    def __init__(self, ffts=(256,512,1024), hops=(64,128,256), wins=(256,512,1024), eps=1e-7):
        super().__init__()
        self.ffts=ffts; self.hops=hops; self.wins=wins; self.eps=eps
    @staticmethod
    def _stft_mag(x, n_fft, hop, win, eps):
        x32 = torch.nan_to_num(x.squeeze(1).to(torch.float32), 0.0, 0.0, 0.0)
        window = torch.hann_window(win, device=x.device, dtype=torch.float32)
        spec = torch.stft(x32, n_fft=n_fft, hop_length=hop, win_length=win,
                          window=window, center=True, pad_mode="reflect", return_complex=True)
        return spec.abs().clamp_min(eps)
    def forward(self, x, y):
        x = finite_or_zero(x); y = finite_or_zero(y)
        used=0; sc=0.0; mag=0.0
        for n,h,w in zip(self.ffts, self.hops, self.wins):
            if x.shape[-1] < max(8, w//2): continue
            X = self._stft_mag(x,n,h,w,self.eps); Y = self._stft_mag(y,n,h,w,self.eps)
            num = (X - Y).pow(2).sum(dim=(1,2)).sqrt()
            den = Y.pow(2).sum(dim=(1,2)).sqrt().clamp_min(self.eps)
            sc  = sc + (num/den).mean()
            mag = mag + F.l1_loss(X, Y)
            used+=1
        if used==0: return 0.1*F.l1_loss(x,y)
        return 0.5*sc/used + 0.5*mag/used

class MelCosineLoss(nn.Module):
    def __init__(self, sr=TARGET_SR, n_fft=MEL_NFFT, hop=MEL_HOP, n_mels=MEL_MELS, eps=1e-7):
        super().__init__()
        self.sr=sr; self.n_fft=n_fft; self.hop=hop; self.n_mels=n_mels; self.eps=eps
        self.mel = torchaudio.transforms.MelScale(
            n_mels=n_mels, sample_rate=sr, n_stft=n_fft//2 + 1,
            f_min=0.0, f_max=sr*0.5, norm=None, mel_scale="htk"
        )
    def _mel_mag(self, x_1T: torch.Tensor):
        x = x_1T[:,0,:].to(torch.float32)
        window = torch.hann_window(self.n_fft, device=x.device, dtype=torch.float32)
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft,
                          window=window, center=True, return_complex=True)
        mag = spec.abs().clamp_min(self.eps)
        M = self.mel.to(x.device)(mag)
        den = M.amax(dim=(1,2), keepdim=True).clamp_min(self.eps)
        M = (M / den + self.eps).log()
        return M
    def forward(self, x, y):
        X = self._mel_mag(x); Y = self._mel_mag(y)
        T = max(X.size(-1), Y.size(-1))
        if X.size(-1) != T: X = F.interpolate(X, size=T, mode="linear", align_corners=False)
        if Y.size(-1) != T: Y = F.interpolate(Y, size=T, mode="linear", align_corners=False)
        num = (X * Y).sum(dim=1)
        den = (X.norm(dim=1) * Y.norm(dim=1)).clamp_min(self.eps)
        cos = (num / den).clamp(-1, 1)
        return (1.0 - cos.mean())

MRSTFT = MultiResSTFTLoss().to(DEVICE)
MELCOS = MelCosineLoss().to(DEVICE)

def safe_l1(x, y): return F.l1_loss(finite_or_zero(x), finite_or_zero(y))

# ================== MODEL PARTS ==================
class PosEnc1D(nn.Module):
    def __init__(self, c, max_len=8192):
        super().__init__()
        pe=torch.zeros(max_len,c); pos=torch.arange(0,max_len).unsqueeze(1)
        div=torch.exp(torch.arange(0,c,2)*(-math.log(10000.0)/c))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe', pe)
    def forward(self, x): T=x.size(-1); return x + self.pe[:T,:].T.unsqueeze(0).to(x.dtype)

class TokenNorm(nn.Module):
    def __init__(self, c): super().__init__(); self.ln=nn.LayerNorm(c)
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
        y=self.out(self.drop(self._merge(ctx))); y=self.ffn(y+q)+(y+q)
        return y.permute(0,2,1)

class ResidualVQEMA(nn.Module):
    def __init__(self, dim: int, n_books: int, n_embed: int, decay: float):
        super().__init__()
        self.books=nn.ParameterList([nn.Parameter(torch.randn(n_embed, dim)/math.sqrt(dim))
                                     for _ in range(n_books)])
        self.decay=float(decay)
        self.n_books=int(n_books); self.n_embed=int(n_embed)
    @staticmethod
    def _nearest_l2(x, emb):
        return (x @ emb.t() - 0.5*(emb*emb).sum(dim=1).unsqueeze(0)).argmax(dim=1)
    def forward(self, z):
        B,D,T=z.shape; x=z.permute(0,2,1).reshape(B*T, D)
        residual=x; q_sum=torch.zeros_like(x)
        for cb in self.books:
            emb=cb.detach().to(z.dtype).to(z.device)
            idx=self._nearest_l2(residual, emb)
            q=F.embedding(idx, emb)
            q_sum = q_sum + (q - residual).detach() + residual
            residual = residual - q
        return q_sum.view(B,T,D).permute(0,2,1).contiguous()
    @torch.no_grad()
    def ema_step(self, z_tokens):
        B,D,T=z_tokens.shape; X=z_tokens.permute(0,2,1).reshape(B*T, D)
        for cb in self.books:
            emb=cb.data
            idx=(X.to(emb) @ emb.t() - 0.5*(emb*emb).sum(dim=1).unsqueeze(0)).argmax(dim=1)
            K=emb.size(0)
            counts=torch.bincount(idx, minlength=K).float().unsqueeze(1)
            sums=torch.zeros_like(emb); sums.index_add_(0, idx, X.to(emb))
            mask = counts.squeeze(1) > 0
            means=torch.zeros_like(emb); means[mask]=sums[mask]/(counts[mask]+1e-9)
            emb[mask] = self.decay*emb[mask] + (1.0-self.decay)*means[mask]

class AllPredAR(nn.Module):
    def __init__(self, A_ENC, A_QUANT, T_ENC, T_DEC, c_lat, rvq_books, rvq_embed):
        super().__init__()
        self.A_ENC, self.A_QUANT, self.T_ENC, self.T_DEC = A_ENC, A_QUANT, T_ENC, T_DEC
        for m in [self.A_ENC, self.A_QUANT, self.T_ENC, self.T_DEC]:
            for p in m.parameters(): p.requires_grad_(False)
        self.predict = CrossPredictor(c=c_lat, heads=8, mlp_mul=2, dropout=0.1)
        self.tokennorm = TokenNorm(c_lat)
        self.scale = nn.Parameter(torch.tensor(0.08))
        self.proj_down = nn.Conv1d(c_lat, CODE_DIM, 1)
        self.proj_up   = nn.Conv1d(CODE_DIM, c_lat, 1)
        self.vq        = ResidualVQEMA(dim=CODE_DIM, n_books=rvq_books, n_embed=rvq_embed, decay=EMA_DECAY)

    def forward_step(self, a_1T, tc_1T):
        B,_,Tw = tc_1T.shape
        za = self.A_ENC(a_1T)
        qa, *_ = self.A_QUANT(za)
        zt_teacher = self.T_ENC(tc_1T)
        B,C,Tlat = zt_teacher.shape
        z_run = torch.zeros_like(zt_teacher)
        rD_all = []

        for s in range(0, Tlat, AR_CHUNK_TOK):
            e = min(Tlat, s+AR_CHUNK_TOK)
            zt_prev = torch.zeros(B, C, e - s, device=zt_teacher.device, dtype=zt_teacher.dtype)
            if s == 0:
                zt_prev[..., 1:] = z_run[..., s:e-1]
            else:
                zt_prev[...]     = z_run[..., s-1:e-1]

            qa_chunk = qa[..., s:e]
            z_pred_chunk = self.predict(zt_prev, qa_chunk)

            r_chunk  = (zt_teacher[..., s:e] - z_pred_chunk.detach())
            rN_chunk = torch.tanh(self.tokennorm(r_chunk))
            scale = self.scale.clamp(5e-3, 0.5)
            rD_chunk = self.proj_down(scale * rN_chunk)
            qD_chunk = self.vq(rD_chunk)
            z_hat_chunk = z_pred_chunk + self.proj_up(qD_chunk)

            z_run[..., s:e] = z_hat_chunk
            rD_all.append(rD_chunk.detach())

        y_hat = self.T_DEC(z_run)
        T = min(y_hat.shape[-1], tc_1T.shape[-1], Tw)
        return {"y_hat":finite_or_zero(y_hat[...,:T]),
                "tgt":finite_or_zero(tc_1T[...,:T]),
                "r_tokens": torch.cat(rD_all, dim=-1) if rD_all else None}

# ================== BUILDERS ==================
def build_backbones():
    dac_audio = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    dac_tact  = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    A_ENC, A_QUANT = dac_audio.encoder, dac_audio.quantizer
    T_ENC, T_DEC   = dac_tact.encoder,  dac_tact.decoder
    dummy = torch.randn(1,1,TARGET_SR, device=DEVICE)
    with autocast('cuda', enabled=False):
        C = A_ENC(dummy).size(1); toks = A_ENC(dummy).size(-1)
    print(f"[Latents] C={C}, tokens/sec≈{toks}")
    return A_ENC, A_QUANT, T_ENC, T_DEC, C, int(toks)

def split_items():
    items = pair_stems(AUDIO_DIR, TACT_DIR)
    random.shuffle(items)
    n_val = max(1, int(len(items) * VAL_FRAC))
    val_items = items[:n_val][:MAX_VAL]
    train_items = items[n_val:]
    return train_items, val_items

# ================== TRAIN ONE COMBO ==================
def train_one(run_dir: str, rvq_books: int, rvq_embed: int,
              tokens_per_sec: int) -> Dict[str, float]:
    os.makedirs(run_dir, exist_ok=True)

    # Dataloaders
    train_items, val_items = split_items()
    train_dl = DataLoader(SegDataset(train_items), batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          collate_fn=collate_fn, drop_last=True)
    val_dl   = DataLoader(SegDataset(val_items), batch_size=BATCH, shuffle=False,
                          num_workers=max(0,NUM_WORKERS//2), pin_memory=PIN_MEMORY,
                          collate_fn=collate_fn)

    # Backbones & model
    A_ENC, A_QUANT, T_ENC, T_DEC, C, _tps = build_backbones()
    net = AllPredAR(A_ENC, A_QUANT, T_ENC, T_DEC, c_lat=C,
                    rvq_books=rvq_books, rvq_embed=rvq_embed).to(DEVICE)

    params = [p for n,p in net.named_parameters() if p.requires_grad and not n.startswith("vq.books")]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.1)
    scaler = GradScaler('cuda', enabled=USE_AMP)

    bits_per_code = math.log2(rvq_embed)
    est_kbps = (tokens_per_sec * rvq_books * bits_per_code) / 1000.0

    best_val = float("inf")
    best_ep = -1
    hist = {"train":[], "val":[], "l1":[], "stft":[], "mel":[], "ema":[]}

    def step(a, tc, train_mode=True, epoch=1):
        a=a.to(DEVICE); tc=tc.to(DEVICE)
        with autocast('cuda', enabled=USE_AMP):
            out = net.forward_step(a, tc)
            y, tgt = out["y_hat"], out["tgt"]

            l1   = safe_l1(y, tgt)
            lstf = MRSTFT(y, tgt)
            lmel = MELCOS(y, tgt)
            total = W_WAV_L1*l1 + W_STFT*lstf + W_MELCOS*lmel

        emau = 0.0
        if train_mode and torch.isfinite(total):
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            scaler.step(opt); scaler.update()
            if epoch > EMA_WARM_E and out["r_tokens"] is not None:
                net.vq.ema_step(out["r_tokens"]); emau = 1.0

        return float(total.detach().cpu()), float(l1.detach().cpu()), float(lstf.detach().cpu()), float(lmel.detach().cpu()), emau

    for ep in range(1, EPOCHS+1):
        net.train()
        t_sum=l1_sum=st_sum=me_sum=ema_sum=0.0; n=0
        for a,tc in train_dl:
            T,L1,ST,ME,EM = step(a, tc, train_mode=True, epoch=ep); n+=1
            t_sum+=T; l1_sum+=L1; st_sum+=ST; me_sum+=ME; ema_sum+=EM
        t_avg=t_sum/max(1,n); hist["train"].append(t_avg)
        hist["l1"].append(l1_sum/max(1,n)); hist["stft"].append(st_sum/max(1,n))
        hist["mel"].append(me_sum/max(1,n)); hist["ema"].append(ema_sum/max(1,n))

        # val
        net.eval(); vs=0.; vm=0
        with torch.no_grad():
            for a,tc in val_dl:
                V,_,_,_,_ = step(a, tc, train_mode=False, epoch=ep)
                vs += V; vm += 1
        v = vs/max(1,vm); hist["val"].append(v)
        sched.step()

        print(f"[{Path(run_dir).name}] Ep {ep:03d} | train {t_avg:.4f} | val {v:.4f} | "
              f"L1 {hist['l1'][-1]:.4f} | STFT {hist['stft'][-1]:.4f} | MEL {hist['mel'][-1]:.4f} | EMA {hist['ema'][-1]:.1f}")

        torch.save({"model":net.state_dict(),"epoch":ep,"hist":hist,
                    "rvq_books":rvq_books,"rvq_embed":rvq_embed,
                    "tokens_per_sec":tokens_per_sec,"bits_per_code":bits_per_code,
                    "kbps":est_kbps},
                   os.path.join(run_dir, "last.pth"))
        if v + 1e-6 < best_val and ep>6:
            best_val = v; best_ep = ep
            torch.save({"model":net.state_dict(),"epoch":ep,"hist":hist,
                        "rvq_books":rvq_books,"rvq_embed":rvq_embed,
                        "tokens_per_sec":tokens_per_sec,"bits_per_code":bits_per_code,
                        "kbps":est_kbps},
                       os.path.join(run_dir, "best.pth"))
            print("✅ saved best")

    # curves
    plt.figure(figsize=(11,5))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.legend()
    plt.title(f"ALL-PRED AR A->T — RVQ B={rvq_books}, K={rvq_embed} (kbps≈{est_kbps:.2f})")
    plt.tight_layout(); plt.savefig(os.path.join(run_dir, "curves.png")); plt.close()

    with open(os.path.join(run_dir,"hist.json"),"w") as f: json.dump(hist, f, indent=2)
    with open(os.path.join(run_dir,"meta.json"),"w") as f:
        json.dump({
            "rvq_books": rvq_books,
            "rvq_embed": rvq_embed,
            "bits_per_code": bits_per_code,
            "tokens_per_sec": tokens_per_sec,
            "kbps_est": est_kbps,
            "best_val": best_val,
            "best_epoch": best_ep
        }, f, indent=2)

    return {"rvq_books":rvq_books,"rvq_embed":rvq_embed,"bits_per_code":bits_per_code,
            "tokens_per_sec":tokens_per_sec,"kbps":est_kbps,
            "best_val":best_val,"best_epoch":best_ep}

# ================== MAIN (SWEEP) ==================
def main():
    set_seed(SEED)

    # probe tokens/sec once (from backbone encoder at TARGET_SR)
    dac_audio = dac.DAC.load(dac.utils.download("24khz")).to(DEVICE).eval()
    with autocast('cuda', enabled=False):
        tps_probe = dac_audio.encoder(torch.zeros(1,1,TARGET_SR, device=DEVICE)).size(-1)
    tokens_per_sec = int(tps_probe)
    print(f"[Probe] tokens/sec ≈ {tokens_per_sec} at {TARGET_SR} Hz")

    # sweep
    summary_rows = []
    for (books, embed) in SWEEP:
        run_dir = os.path.join(OUT_ROOT, f"rvqB{books}_K{embed}")
        print(f"\n=== Combo: books={books}, embed={embed} (≈{math.log2(embed):.1f} bits/code) ===")
        res = train_one(run_dir, books, embed, tokens_per_sec)
        summary_rows.append(res)

    # write sweep summary CSV (sorted by kbps then best_val)
    summary_rows.sort(key=lambda r: (r["kbps"], r["best_val"]))
    csv_path = os.path.join(OUT_ROOT, "sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rvq_books","rvq_embed","bits_per_code","tokens_per_sec","kbps","best_val","best_epoch"
        ])
        w.writeheader()
        for r in summary_rows: w.writerow(r)
    print(f"\n✅ Sweep complete. Summary → {csv_path}")
    for r in summary_rows:
        print(f"B={r['rvq_books']:>2d} K={r['rvq_embed']:>3d} | "
              f"kbps={r['kbps']:6.2f} | best_val={r['best_val']:.4f} @ ep {r['best_epoch']}")

if __name__ == "__main__":
    main()
