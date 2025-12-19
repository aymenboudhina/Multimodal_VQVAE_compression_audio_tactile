#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract raw 'sound' from .mat files across C1..C8 (no normalization),
and save as 32-bit float WAVs into one output folder.

- Recursively searches BASE_DIR/C1..C8 for *.mat
- Handles classic .mat (scipy) and v7.3 HDF5 (h5py) files
- Keeps original amplitudes (no scaling/clamping)
- Writes:
    * WAV (FLOAT, 32-bit)  -> WAV_Files_raw/<stem>.wav
    * PNG quick-look plot  -> WAV_Files_raw/plots/<stem>.png
    * JSON stats           -> WAV_Files_raw/stats/<stem>.json

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
# Parent folder that contains C1, C2, ..., C8
BASE_DIR    = r"C:\Users\Aymen\Master thesis\CBM_FinalDatabase"
OUTPUT_DIR  = r"C:\Users\Aymen\Master thesis\WAV_Files_raw"
SAMPLE_RATE = 44100   # Hz (use the correct recording SR for 'sound')
# -----------------------------------

PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)


# ----------- helpers -----------
def _ensure_1d_mono(x: np.ndarray) -> np.ndarray:
    """Return a contiguous 1-D float32 signal. If 2-D, downmix by mean."""
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim == 2:
        if x.shape[0] == 1 or x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            # average across the likely channel axis
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
            print(f"⚠️  {os.path.basename(mat_path)} is v7.3. Install h5py to read it.")
            return None, None
        try:
            f = h5py.File(mat_path, 'r')
            return f, 'h5py'
        except Exception as e:
            print(f"❌  h5py open failed for {mat_path}: {e}")
            return None, None
    except Exception as e:
        print(f"❌  loadmat failed for {mat_path}: {e}")
        return None, None


def _extract_from_finalMaterialRecording(fmr):
    """Common ways to access finalMaterialRecording.sound (scipy-loaded)."""
    # attribute-style
    if hasattr(fmr, "sound"):
        return np.asarray(getattr(fmr, "sound"))
    # dict-like
    try:
        return np.asarray(fmr["sound"])
    except Exception:
        pass
    # unwrap zero-d object arrays
    try:
        inner = fmr[()]
        if hasattr(inner, "sound"):
            return np.asarray(getattr(inner, "sound"))
        try:
            return np.asarray(inner["sound"])
        except Exception:
            pass
    except Exception:
        pass
    return None


def extract_sound(mat_obj, backend: str):
    """Return 1-D float32 'sound' or None."""
    if backend == 'scipy':
        d = mat_obj
        fmr = d.get('finalMaterialRecording', None)
        if fmr is not None:
            sig = _extract_from_finalMaterialRecording(fmr)
            if sig is not None:
                return _ensure_1d_mono(sig)
        if 'sound' in d:
            return _ensure_1d_mono(d['sound'])
        # last-resort search
        for k, v in d.items():
            if k.startswith("__"):
                continue
            if "sound" in k.lower():
                arr = np.asarray(v)
                if np.issubdtype(arr.dtype, np.number):
                    return _ensure_1d_mono(arr)
        return None

    elif backend == 'h5py':
        f = mat_obj
        # common paths
        for path in ["finalMaterialRecording/sound", "sound"]:
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
                if isinstance(item, h5py.Dataset) and ("sound" in k.lower()):
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
    plt.savefig(path_png, dpi=160)
    plt.close()
# ----------------------------------


def process_one(mat_path: str) -> bool:
    obj, backend = load_mat_safely(mat_path)
    if backend is None:
        return False
    try:
        sig = extract_sound(obj, backend)
    finally:
        if backend == 'h5py':
            try: obj.close()
            except Exception: pass

    if sig is None or sig.size == 0:
        print(f"⏩  No 'sound' in {mat_path}")
        return False

    # keep original values — DO NOT normalize/scale/clamp
    stem = Path(mat_path).stem

    wav_path  = os.path.join(OUTPUT_DIR, f"{stem}.wav")
    png_path  = os.path.join(PLOTS_DIR,  f"{stem}.png")
    stats_path= os.path.join(STATS_DIR,  f"{stem}.json")

    save_wav_float32(wav_path, sig, SAMPLE_RATE)
    save_plot(png_path, sig, SAMPLE_RATE,
              f"{stem} — sound (raw), min={sig.min():.3g}, max={sig.max():.3g}")

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
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ {stem}: saved WAV, plot, and stats.")
    return True


def main():
    # Collect *.mat from all C1..C8 under BASE_DIR
    class_glob = [os.path.join(BASE_DIR, f"C{i}", "**", "*.mat") for i in range(1, 9)]
    mats = []
    for g in class_glob:
        mats.extend(glob.glob(g, recursive=True))

    mats = sorted(set(mats))
    if not mats:
        print("No .mat files found under C1..C8.")
        return

    print(f"Found {len(mats)} .mat files across C1..C8.")
    ok = 0
    for p in mats:
        try:
            ok += 1 if process_one(p) else 0
        except Exception as e:
            print(f"❌ Error processing {p}: {e}")
    print(f"\nDone. Converted {ok}/{len(mats)} files to WAV with raw amplitudes.")
    print(f"Output WAVs:  {OUTPUT_DIR}")
    print(f"Quick-look plots: {PLOTS_DIR}")
    print(f"Stats JSON:  {STATS_DIR}")


if __name__ == "__main__":
    # Optional CLI overrides:
    #   python extract_sound_raw_allclasses.py "C:\...\CBM_FinalDatabase" "C:\...\WAV_Files_raw" 44100
    if len(sys.argv) >= 2: BASE_DIR = sys.argv[1]
    if len(sys.argv) >= 3: OUTPUT_DIR = sys.argv[2]
    if len(sys.argv) >= 4:
        try: SAMPLE_RATE = int(sys.argv[3])
        except Exception: pass
    main()
