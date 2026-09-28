"""EdgeFace Base embeddings and similarity for L2-normalized vectors."""
import cv2
import numpy as np


def embed(aligned_img: np.ndarray, session) -> np.ndarray:
    """Run EdgeFace on a 112x112 BGR face and return its normalized 512D vector."""
    rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: blob})[0]
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Dot product of L2-normalized embeddings, clamped to [0, 1]."""
    return float(max(0.0, min(1.0, np.dot(emb1.flatten(), emb2.flatten()))))
