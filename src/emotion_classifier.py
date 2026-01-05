from pathlib import Path
from typing import Callable, Optional
import torch
import numpy as np
import torch.nn.functional as F

from src.emotion_infer import load_labels, preprocess_face_roi
from src.model.emotion_predict import EmotionPredict


class EmotionClassifier:
    """
    Load a PyTorch model from .pth and run inference.
    Supports:
    - TorchScript (recommended for code-free deployment)
    - state_dict (requires a build_model callback)
    """

    def __init__(self, model_path: str, labels_path: Optional[str] = None, 
                 device: str = "cpu", 
                 build_model: Optional[Callable[[], torch.nn.Module]] = None) -> None:
        self.device = torch.device(device)
        self.labels = load_labels(label_path=labels_path)

        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Emotion model not found: {model_path}")

        try:
            self.model = torch.jit.load(str(p), map_location=self.device)
        except:
            obj = torch.load(str(p), map_location=self.device)

            if isinstance(obj, torch.nn.Module):
                self.model = obj
            else:
                state_dict = None

                if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
                    state_dict = obj.get("state_dict", None) or obj
                if state_dict is None:
                    raise RuntimeError(
                        "Unsupported model file. Expected TorchScript, nn.Module, or state_dict."
                    )
                if build_model is None:
                    raise RuntimeError(
                        "Detected state_dict. Provide build_model=lambda: YourModel(...)."
                    )
                
                # strip 'module.' if needed
                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

                self.model = build_model().to(device=self.device)
                missing, unexpected = self.model.load_state_dict(
                    state_dict, strict=False)
                if missing or unexpected:
                    print(
                        f"Warning: missing keys: {missing}, unexpected keys: {unexpected}")

        self.model.eval()
        self.model.to(self.device)

    def predict_from_roi(self, roi_bgr: np.ndarray) -> EmotionPredict:
        x = preprocess_face_roi(roi_bgr).to(self.device)  # (1, C, H, W)

        print(x.shape, x.dtype, x.min().item(), x.max().item(), x.mean().item())

        with torch.inference_mode():
            logits = self.model(x)
            probs_t = torch.softmax(logits, dim=1)[0]   # (7,)
            probs = probs_t.cpu().numpy()

            idx = int(probs.argmax())
            conf = float(probs[idx])

            top3 = probs.argsort()[-3:][::-1]
            gap = float(probs[top3[0]] - probs[top3[1]])
            print("top3=", [(self.labels[i], float(probs[i])) for i in top3], "gap=", gap)
            print("PREDICT:", self.labels[idx], conf)
        
        idx = np.argmax(probs)
        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        conf = float(probs[idx])
        return EmotionPredict(label=label, conf=conf, probs=probs.tolist())
