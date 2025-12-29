from pathlib import Path
from typing import Optional, List, Tuple, Dict

import cv2
from insightface.app import FaceAnalysis
import numpy as np

from src.model.recognition_result import RecognitionResult
from src.utils.util import l2norm, cosine_sim, sim_to_percent


class FaceEngine:
    """
    Face recognition engine using InsightFace FaceAnalysis.
    - Enroll: save embeddings per user
    - Recognize: detect faces -> compute embeddings -> match with cosine similarity
    """

    def __init__(self,
                 model_name: str = "buffalo_s",
                 providers: Optional[List[str]] = None,
                 det_size: Tuple[int, int] = (640, 640),
                 db_dir: str = "db/embeddings",
                 threshold: float = 0.40,
                 unknown_percent_cutoff: float = 30  # requirement: <30% => Unknown
                 ):
        self.model_name = model_name
        self.providers = providers or ["CPUExecutionProvider"]
        self.det_size = det_size
        self.db_dir = Path(db_dir)

        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.threshold = float(threshold)
        self.unknown_percent_cutoff = float(unknown_percent_cutoff)

        self.app = FaceAnalysis(name=self.model_name, providers=self.providers)
        # ctx_id: 0 is fine even for CPU; InsightFace uses it as internal param
        self.app.prepare(ctx_id=0, det_size=self.det_size)

        self.alpha = 0.3
        self.ema_sim = None
        self.threshold_high = 0.48
        self.threshold_low = 0.43
        self.last_name_stable = "Unknown"

        # Load known embeddings at init
        self.known: Dict[str, np.ndarray] = {}
        self.reload_db()

    def reload_db(self) -> None:
        self.known.clear()

        for p in sorted(self.db_dir.glob("*.npy")):
            name = p.stem
            embedding = np.load(p)
            if embedding.ndim != 1:
                continue
            self.known[name] = embedding

    def extract_embedding_from_image(self, bgr_image: np.ndarray, max_faces=1) -> List[np.ndarray]:
        faces = self._extract_faces(bgr_image)

        # sort by bbox area descending, take top max_faces
        def area(_f):
            x_1, y_1, x_2, y_2 = _f.bbox
            return (x_2 - x_1) * (y_2 - y_1)

        faces = sorted(faces, key=area, reverse=True)[:max_faces]

        embeddings = []
        for f in faces:
            embedding = f.embedding
            if embedding is None:
                continue
            embeddings.append(l2norm(np.asarray(embedding, dtype=np.float32)))
        return embeddings

    def enroll(self, name: str, bgr_images: List[np.ndarray], require_one_face: bool = True) -> np.ndarray:
        """
        Enroll a user from multiple images:
        - detect face in each image
        - take the biggest face
        - average embeddings
        - normalize & save to db
        """
        name = name.strip()
        if not name:
            raise ValueError("Name is empty")

        embeddings: List[np.ndarray] = []

        for bgr_image in bgr_images:
            embedding = self.extract_embedding_from_image(bgr_image, max_faces=1)
            if len(embedding) == 0:
                continue

            embeddings.append(embedding[0])

        if len(embeddings) == 0 and require_one_face:
            raise ValueError("No face found in provided images")

        mean_embedding = l2norm(np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32))
        output_path = self.db_dir / f"{name}.npy"
        np.save(output_path, mean_embedding)

        # update in-memory
        self.known[name] = mean_embedding
        return mean_embedding

    def match(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Return (best_name, best_similarity). If no known users, returns ("Unknown", -1).
        """
        if not self.known:
            return "Unknown", -1.0

        best_name = "Unknown"
        best_similarity = -1.0

        for name, ref_embedding in self.known.items():
            similarity = cosine_sim(embedding, ref_embedding)
            if similarity > best_similarity:
                best_name = name
                best_similarity = similarity

        return best_name, best_similarity

    def recognize(self, bgr_frame: np.ndarray, max_faces: int = 5) -> List[RecognitionResult]:
        """
        Detect faces and recognize each face.
        """
        faces = self._extract_faces(bgr_frame)

        # take up to max_faces (largest first)
        def area(_f):
            x_1, y_1, x_2, y_2 = _f.bbox
            return (x_2 - x_1) * (y_2 - y_1)

        faces = sorted(faces, key=area, reverse=True)[:max_faces]

        results: List[RecognitionResult] = []

        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            det_score = float(getattr(f, "det_score", 0.0))
            emb = f.embedding
            if emb is None:
                continue

            emb = l2norm(np.asarray(emb, dtype=np.float32))
            best_name, best_sim = self.match(emb)

            is_known = best_sim >= self.threshold

            shown_name = best_name if is_known else "Unknown"
            percent = sim_to_percent(best_sim, center=0.40, steep=20) if best_sim >= 0 else 0.0

            results.append(
                RecognitionResult(
                    bbox=(x1, y1, x2, y2),
                    det_score=det_score,
                    name=shown_name,
                    similarity=float(best_sim),
                    percent=float(percent),
                )
            )
        return results

    def _extract_faces(self, bgr: np.ndarray):
        # InsightFace expects BGR (OpenCV) image
        return self.app.get(bgr)


def draw_overlays(bgr_frame: np.ndarray, recognizes: List[RecognitionResult]) -> np.ndarray:
    """
    Draw bbox + name + percent on frame (OpenCV).
    """
    out = bgr_frame.copy()
    h, w = out.shape[:2]

    for recognize in recognizes:
        x1, y1, x2, y2 = recognize.bbox

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{recognize.name} {recognize.percent:.1f}%"

        # background box for readability
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(out, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return out
