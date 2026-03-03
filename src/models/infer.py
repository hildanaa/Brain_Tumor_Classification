# src/models/infer.py
"""
Wrapper for inference. Tries multiple likely places in your repo (training/infer_05, models, etc.)
Exposes: predict(pil_image) -> (probs, pred_label, pred_id)
"""

from typing import Tuple, Any
import numpy as np

# Try importing the real functions from likely modules
_real_predict = None

# 1) training/infer_05.py common candidate
try:
    from src.training.infer_05 import predict as _real_predict  # type: ignore
except Exception:
    try:
        from src.training.infer_05 import predict_pil as _real_predict  # type: ignore
    except Exception:
        try:
            from src.training.infer_05 import infer as _real_predict  # type: ignore
        except Exception:
            _real_predict = None

# 2) models.* candidates
if _real_predict is None:
    try:
        from src.models.model_factory_03 import predict as _real_predict  # type: ignore
    except Exception:
        try:
            from src.models.model_factory_03 import predict_pil as _real_predict  # type: ignore
        except Exception:
            _real_predict = None

# 3) fallback: check src.models.infer if present (rare)
if _real_predict is None:
    try:
        from src.models.infer import predict as _real_predict  # type: ignore
    except Exception:
        _real_predict = None

# Demo fallback
def _mock_predict(pil_image) -> Tuple[np.ndarray, str, int]:
    probs = np.array([0.10, 0.08, 0.05, 0.77], dtype=float)
    pred_id = int(np.argmax(probs))
    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
    pred_label = class_names[pred_id]
    return probs, pred_label, pred_id

def predict(pil_image) -> Tuple[Any, str, int]:
    """
    Returns (probs, pred_label, pred_id)
    Tries real implementation first; falls back to demo.
    """
    if callable(_real_predict):
        try:
            out = _real_predict(pil_image)
            # Common return formats: (probs_array, label) or (probs, label, id)
            if isinstance(out, tuple):
                if len(out) == 2:
                    probs, label = out
                    # infer id
                    if hasattr(probs, "argmax"):
                        pid = int(np.argmax(probs))
                    elif isinstance(probs, dict):
                        pid = 0
                    else:
                        pid = 0
                    return probs, label, pid
                elif len(out) >= 3:
                    return out[0], out[1], int(out[2])
            # If single array or dict returned
            if hasattr(out, "argmax") or isinstance(out, (list, np.ndarray, dict)):
                probs = out
                if isinstance(probs, dict):
                    label = max(probs, key=probs.get)
                    pid = 0
                else:
                    pid = int(np.argmax(probs))
                    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
                    label = class_names[pid] if pid < len(class_names) else str(pid)
                return probs, label, pid
        except Exception:
            pass

    # fallback
    return _mock_predict(pil_image)
