#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import soundfile as sf

# You can install scipy and (optionally) h5py via:
#   pip install scipy accelDFTfile h5py
from scipy.io import loadmat

try:
    import h5py  # optional, only used for v7.3 .mat fallback
    H5PY_AVAILABLE = True
except Exception:
    H5PY_AVAILABLE = False


# =========================
# Config
# =========================
ROOT_PATH   = r"C:\Users\Aymen\Master thesis\CBM_FinalDatabase\C5"   # <-- change if needed
OUTPUT_DIR  = r"C:\Users\Aymen\Master thesis\Vibrotactile_Files"               # <-- change if needed
SAMPLE_RATE = 3000  # Hz


# =========================
# Helpers
# =========================
def normalize_audio(x: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] as float32. Safe for zeros."""
    x = np.asarray(x, dtype=np.float32)
    max_val = np.max(np.abs(x)) if x.size else 0.0
    if max_val > 0:
        x = x / max_val
    return x


def _extract_from_finalMaterialRecording(fmr):
    """
    Try common ways to access 'accelDFT' field inside finalMaterialRecording.
    Works for both MATLAB structs loaded via scipy (numpy.void) and objects.
    Returns np.ndarray or None.
    """
    # Attribute-style (object) access
    if hasattr(fmr, "accelDFT"):
        return np.asarray(getattr(fmr, "accelDFT"))
    # Dict/field-style access
    try:
        return np.asarray(fmr["accelDFT"])
    except Exception:
        pass
    # Some MATLAB structs come as 0-d object arrays or nested arrays
    # Try indexing [()] to unwrap if available
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


def _first_1d_numeric_named(mat_dict, name_hint="accelDFT"):
    """
    Fallback: search the loaded dict for the first 1D numeric array whose key
    contains the given name hint (case-insensitive).
    """
    hint = name_hint.lower()
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if hint in k.lower():
            arr = np.asarray(v)
            # Flatten multi-d to 1D if plausible
            if np.issubdtype(arr.dtype, np.number):
                return arr
    return None


def _ensure_1d_mono(x: np.ndarray) -> np.ndarray:
    """
    Make sure the signal is 1D. If 2D, downmix to mono by mean across the last dim.
    Then flatten to 1D.
    """
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim == 2:
        # Downmix channels (rows or cols) to mono
        if x.shape[0] == 1 or x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            # average across the smaller dimension interpreted as channels
            # Heuristic: if one dim <= 8 treat that as channels
            if x.shape[0] <= 8:
                x = x.mean(axis=0)
            elif x.shape[1] <= 8:
                x = x.mean(axis=1)
            else:
                # If uncertain, just average across last axis
                x = x.mean(axis=-1)
    # Any remaining dims: flatten
    return x.astype(np.float32).ravel()


def load_mat_safely(mat_path: str):
    """
    Load a .mat file. Try scipy first; if it fails due to v7.3 HDF5
    and h5py is available, use h5py.
    Returns either (mat_dict, 'scipy') or (h5file, 'h5py').
    """
    try:
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        return mat, 'scipy'
    except NotImplementedError as e:
        # Likely a v7.3 MAT-file (HDF5-backed)
        if H5PY_AVAILABLE:
            try:
                f = h5py.File(mat_path, 'r')
                return f, 'h5py'
            except Exception as e2:
                print(f"❌  ERROR opening v7.3 MAT {mat_path}: {e2}")
                return None, None
        else:
            print(f"⚠️  {os.path.basename(mat_path)} is MAT v7.3 (HDF5). Install h5py to read. Skipping.")
            return None, None
    except Exception as e:
        print(f"❌  ERROR loading {mat_path}: {e}")
        return None, None


def extract_accelDFT_signal(mat_obj, backend: str):
    """
    Try to extract the 'accelDFT' 1D signal from the loaded MAT content.
    Returns np.ndarray (float32) or None.
    """
    if backend == 'scipy':
        mat = mat_obj  # dict-like
        # 1) Preferred: finalMaterialRecording.accelDFT
        fmr = mat.get('finalMaterialRecording', None)
        if fmr is not None:
            snd = _extract_from_finalMaterialRecording(fmr)
            if snd is not None:
                return _ensure_1d_mono(snd)

        # 2) Fallback: top-level 'accelDFT' key
        if 'accelDFT' in mat:
            return _ensure_1d_mono(mat['accelDFT'])

        # 3) Last resort: scan for anything named like 'accelDFT'
        snd = _first_1d_numeric_named(mat, name_hint="accelDFT")
        if snd is not None:
            return _ensure_1d_mono(snd)

        return None

    elif backend == 'h5py':
        f = mat_obj  # h5py.File
        # Try common paths
        candidate_paths = [
            "finalMaterialRecording/accelDFT",
            "accelDFT",
        ]
        for p in candidate_paths:
            if p in f:
                try:
                    data = np.array(f[p])
                    return _ensure_1d_mono(data)
                except Exception:
                    pass

        # Broad search for datasets containing "accelDFT"
        def walk_keys(group, prefix=""):
            for k in group.keys():
                full = f"{prefix}{k}"
                item = group[k]
                if isinstance(item, h5py.Dataset) and ("accelDFT" in k.lower()):
                    yield full
                elif isinstance(item, h5py.Group):
                    yield from walk_keys(item, prefix=full + "/")

        for path in walk_keys(f, ""):
            try:
                data = np.array(f[path])
                return _ensure_1d_mono(data)
            except Exception:
                pass

        return None

    else:
        return None


def save_wav(signal: np.ndarray, wav_path: str, sr: int):
    """Write float32 WAV."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    sf.write(wav_path, signal.astype(np.float32), sr)


def process_mat_file(mat_path: str, wav_output_dir: str, sample_rate: int = SAMPLE_RATE) -> bool:
    """
    Try to load and extract 'accelDFT' from .mat.
    Returns True if saved, False if skipped.
    """
    mat_obj, backend = load_mat_safely(mat_path)
    if backend is None:
        return False

    try:
        accelDFT = extract_accelDFT_signal(mat_obj, backend)
    finally:
        # Close h5py file if used
        if backend == 'h5py':
            try:
                mat_obj.close()
            except Exception:
                pass

    if accelDFT is None or accelDFT.size == 0:
        print(f"⏩  Skipping (no 'accelDFT' found): {mat_path}")
        return False

    # Normalize and save
    accelDFT = normalize_audio(accelDFT)
    base_name = os.path.splitext(os.path.basename(mat_path))[0]
    wav_path = os.path.join(wav_output_dir, f"{base_name}.wav")
    save_wav(accelDFT, wav_path, sample_rate)
    print(f"✅ Saved: {wav_path}")
    return True


# =========================
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_mats = 0
    saved = 0
    skipped = 0
    errors = 0

    for dirpath, _, filenames in os.walk(ROOT_PATH):
        for filename in filenames:
            if not filename.lower().endswith(".mat"):
                continue
            total_mats += 1
            mat_file_path = os.path.join(dirpath, filename)
            try:
                ok = process_mat_file(mat_file_path, OUTPUT_DIR, SAMPLE_RATE)
                if ok:
                    saved += 1
                else:
                    skipped += 1
            except Exception as e:
                errors += 1
                print(f"❌  ERROR processing {mat_file_path}: {e}")

    print("\n========== SUMMARY ==========")
    print(f"Total .mat files:   {total_mats}")
    print(f"WAV saved:          {saved}")
    print(f"Skipped (no accelDFT): {skipped}")
    print(f"Errors:             {errors}")
    print("=============================\n")

if __name__ == "__main__":
    # Allow overriding paths from command line if desired:
    #   python script.py "C:\path\to\CBM_FinalDatabase\C5" "C:\path\to\WAV_Files" 3000
    if len(sys.argv) >= 2:
        ROOT_PATH = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_DIR = sys.argv[2]
    if len(sys.argv) >= 4:
        try:
            SAMPLE_RATE = int(sys.argv[3])
        except Exception:
            pass
    main()
