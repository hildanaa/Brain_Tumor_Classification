# src/models/infer.py  (DEBUG mode)
"""
Debug wrapper for inference.
Will print which candidate import succeeded and any exception when calling it.
Exposes: predict(pil_image) -> (probs, pred_label, pred_id)
"""

from typing import Tuple, Any
import numpy as np
import traceback

_real_predict = None
_real_source = None

def _try_import(module_path: str, attr: str):
    global _real_predict, _real_source
    try:
        mod = __import__(module_path, fromlist=[attr])
        fn = getattr(mod, attr)
        if callable(fn):
            _real_predict = fn
            _real_source = f"{module_path}.{attr}"
            print(f"[infer wrapper] imported {_real_source}")
            return True
    except Exception as e:
        print(f"[infer wrapper] import {module_path}.{attr} failed: {e}")
    return False

# Try likely candidates (based on repo files you shared)
_try_import("src.training.infer_05", "predict")
_try_import("src.training.infer_05", "predict_pil")
_try_import("src.training.infer_05", "infer")
_try_import("src.models.model_factory_03", "predict")
_try_import("src.models.model_factory_03", "predict_pil")
_try_import("src.models.infer", "predict")  # fallback if someone already added it

def _mock_predict(pil_image) -> Tuple[np.ndarray, str, int]:
    probs = np.array([0.10, 0.08, 0.05, 0.77], dtype=float)
    pred_id = int(np.argmax(probs))
    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
    pred_label = class_names[pred_id]
    # special marker to indicate mock
    return probs, pred_label + "_MOCK", pred_id

def predict(pil_image) -> Tuple[Any, str, int]:
    """
    Try to run the real predict if available; otherwise fallback to demo.
    Logs exceptions to stdout (visible in Streamlit logs).
    """
    if _real_predict is None:
        print("[infer wrapper] No real predict function found. Using MOCK.")
        return _mock_predict(pil_image)

    print(f"[infer wrapper] Calling real predict from {_real_source} ...")
    try:
        out = _real_predict(pil_image)
        print("[infer wrapper] raw output from real predict:", type(out))
        # Normalize outputs into (probs, label, id)
        if isinstance(out, tuple):
            if len(out) == 2:
                probs, label = out
                if hasattr(probs, "argmax"):
                    pred_id = int(np.argmax(probs))
                elif isinstance(probs, dict):
                    pred_id = 0
                else:
                    pred_id = 0
                print("[infer wrapper] returning (probs,label,id) from 2-tuple")
                return probs, label, pred_id
            elif len(out) >= 3:
                print("[infer wrapper] returning (probs,label,id) from >=3 tuple")
                return out[0], out[1], int(out[2])
        if hasattr(out, "argmax") or isinstance(out, (list, np.ndarray, dict)):
            probs = out
            if isinstance(probs, dict):
                label = max(probs, key=probs.get)
                pid = 0
            else:
                pid = int(np.argmax(probs))
                class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
                label = class_names[pid] if pid < len(class_names) else str(pid)
            print("[infer wrapper] returning normalized single-return")
            return probs, label, pid

        print("[infer wrapper] Unexpected return type from real predict. Falling back to MOCK.")
    except Exception as e:
        print("[infer wrapper] Exception when calling real predict:")
        traceback.print_exc()

    return _mock_predict(pil_image)
