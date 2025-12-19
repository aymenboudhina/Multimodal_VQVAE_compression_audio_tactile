#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract raw accelDFT from .mat files (no normalization), plot, and save as WAV.

- Supports classic .mat (scipy) and v7.3 HDF5 (h5py).
- Keeps original amplitudes (no division by max, no clamp).
- Saves WAV as 32-bit float so values outside [-1, 1] are preserved.
- Writes a small JSON with stats for verification.

pip install numpy scipy h5py soundfile matplotlib
"""

import os, sys, json, glob
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import loadmat

try:
    import h5py
    H5PY_AVAILABLE = True
except Exception:
    H5PY_AVAILABLE = False

# ----------- USER CONFIG -----------
ROOT_PATH   = r"C:\Users\Aymen\Master thesis\CBM_FinalDatabase\C1"  # folder with .mat files
OUTPUT_DIR  = r"C:\Users\Aymen\Master thesis\C1_RAW_WAV"            # where wav/plots/stats go
SAMPLE_RATE = 3000  # Hz (set to the recording rate of accelDFT)
FILE_GLOB   = "**/*.mat"  # recursive
# -----------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------- helpers -----------
def _ensure_1d_mono(x: np.ndarray) -> np.ndarray:
    """Return a contiguous 1-D float32 signal. If 2-D, downmix by mean."""
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim == 2:
        # treat one dim as channels → average across the smaller dimension
        if x.shape[0] == 1 or x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            if x.shape[0] <= 8:
                x = x.mean(axis=0)
            elif x.shape[1] <= 8:
                x = x.mean(axis=1)
            else:
                x = x.mean(axis=-1)
    return np.ascontiguousarray(x.astype(np.float32).ravel())


def load_mat_safely(mat_path: str):
    """Try scipy first; fall back to h5py for v7.3. Returns (obj, 'scipy'|'h5py') or (None,None)."""
    try:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        return mat, 'scipy'
    except NotImplementedError:
        if not H5PY_AVAILABLE:
            print(f"⚠️  {os.path.basename(mat_path)} is v7.3. Install h5py.")
            return None, None
        try:
            f = h5py.File(mat_path, 'r')
            return f, 'h5py'
        except Exception as e:
            print(f"❌  h5py open failed: {e}")
            return None, None
    except Exception as e:
        print(f"❌  loadmat failed: {e}")
        return None, None


def _extract_from_finalMaterialRecording(fmr):
    """Common ways to access finalMaterialRecording.accelDFT (scipy-loaded object)."""
    # attribute
    if hasattr(fmr, "accelDFT"):
        return np.asarray(getattr(fmr, "accelDFT"))
    # dict-like
    try:
        return np.asarray(fmr["accelDFT"])
    except Exception:
        pass
    # unwrap 0-d object arrays
    try:
        inner = fmr[()]
        if hasattr(inner, "accelDFT"):
            return np.asarray(getattr(inner, "accelDFT"))
        try:
            return np.asarray(inner["accelDFT"])
        except Exception:
            pass
    except Exception:
        pass
    return None


def extract_accelDFT(mat_obj, backend: str):
    """Return 1-D float32 accelDFT or None."""
    if backend == 'scipy':
        d = mat_obj
        # preferred
        fmr = d.get('finalMaterialRecording', None)
        if fmr is not None:
            sig = _extract_from_finalMaterialRecording(fmr)
            if sig is not None:
                return _ensure_1d_mono(sig)
        # top-level
        if 'accelDFT' in d:
            return _ensure_1d_mono(d['accelDFT'])
        # last resort: search keys
        for k, v in d.items():
            if k.startswith("__"): continue
            if "accelDFT".lower() in k.lower():
                arr = np.asarray(v)
                if np.issubdtype(arr.dtype, np.number):
                    return _ensure_1d_mono(arr)
        return None

    elif backend == 'h5py':
        f = mat_obj
        # try common paths
        for path in ["finalMaterialRecording/accelDFT", "accelDFT"]:
            if path in f:
                try:
                    return _ensure_1d_mono(np.array(f[path]))
                except Exception:
                    pass
        # broad search
        def walk(group, pref=""):
            for k in group.keys():
                full = f"{pref}{k}"
                item = group[k]
                if isinstance(item, h5py.Dataset) and ("accelDFT".lower() in k.lower()):
                    yield full
                elif isinstance(item, h5py.Group):
                    yield from walk(item, pref=full + "/")
        for p in walk(f, ""):
            try:
                return _ensure_1d_mono(np.array(f[p]))
            except Exception:
                pass
        return None

    return None


def save_wav_float32(path: str, x: np.ndarray, sr: int):
    """Write WAV as 32-bit float (no scaling/clipping)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x.astype(np.float32), sr, subtype="FLOAT")


def save_plot(path_png: str, x: np.ndarray, sr: int, title: str):
    t = np.arange(x.size, dtype=np.float64) / float(sr)
    plt.figure(figsize=(10, 3.4))
    plt.plot(t, x, lw=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (raw)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.savefig(path_png, dpi=160)
    plt.close()
# ----------------------------------


def process_one(mat_path: str) -> bool:
    obj, backend = load_mat_safely(mat_path)
    if backend is None:
        return False
    try:
        sig = extract_accelDFT(obj, backend)
    finally:
        if backend == 'h5py':
            try: obj.close()
            except Exception: pass

    if sig is None or sig.size == 0:
        print(f"⏩  No accelDFT in {mat_path}")
        return False

    # keep original values — DO NOT normalize/scale/clamp
    stem = Path(mat_path).stem
    out_base = os.path.join(OUTPUT_DIR, stem)

    # WAV
    wav_path = out_base + ".wav"
    save_wav_float32(wav_path, sig, SAMPLE_RATE)

    # Plot
    png_path = out_base + ".png"
    title = f"{stem} — accelDFT (raw), min={sig.min():.3g}, max={sig.max():.3g}"
    save_plot(png_path, sig, SAMPLE_RATE, title)

    # Stats JSON
    stats = {
        "file": mat_path,
        "wav": wav_path,
        "sr": SAMPLE_RATE,
        "num_samples": int(sig.size),
        "min": float(sig.min()),
        "max": float(sig.max()),
        "mean": float(sig.mean()),
        "std": float(sig.std(ddof=0)),
    }
    with open(out_base + ".json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ {stem}: saved WAV ({wav_path}), plot, and stats.")
    return True


def main():
    mats = sorted(glob.glob(os.path.join(ROOT_PATH, FILE_GLOB), recursive=True))
    if not mats:
        print("No .mat files found.")
        return
    print(f"Found {len(mats)} .mat files.")
    ok = 0
    for p in mats:
        try:
            ok += 1 if process_one(p) else 0
        except Exception as e:
            print(f"❌ Error processing {p}: {e}")
    print(f"Done. Converted {ok}/{len(mats)} files to WAV with raw amplitudes.")

if __name__ == "__main__":
    # Optional CLI overrides:
    #   python extract_acceldft_raw.py "C:\...\C6" "C:\...\C6_RAW_WAV" 3000
    if len(sys.argv) >= 2: ROOT_PATH = sys.argv[1]
    if len(sys.argv) >= 3: OUTPUT_DIR = sys.argv[2]
    if len(sys.argv) >= 4:
        try: SAMPLE_RATE = int(sys.argv[3])
        except Exception: pass
    main()
