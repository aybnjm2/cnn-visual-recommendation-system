import numpy as np
import pandas as pd
from typing import Optional

class ProductRecommender:
    def __init__(self, feature_db: dict, styles_df: pd.DataFrame, alpha: float = 0.7):
        self.styles_df = styles_df.set_index("id")
        self.alpha = alpha
        ids, vecs = zip(*feature_db.items())
        self.ids = np.array(ids, dtype=np.int32)
        self.matrix = np.array(vecs, dtype=np.float32)

    def recommend(self, query_vector: np.ndarray, top_k: int = 10, query_metadata: Optional[dict] = None) -> pd.DataFrame:
        visual_scores = (self.matrix @ query_vector).astype(float)

        if query_metadata and self.alpha < 1.0:
            meta_scores = self._metadata_score(query_metadata)
            scores = self.alpha * visual_scores + (1 - self.alpha) * meta_scores
        else:
            scores = visual_scores

        top_idx = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_idx:
            pid = int(self.ids[idx])
            row = {"id": pid, "visual_score": visual_scores[idx], "final_score": scores[idx]}
            if pid in self.styles_df.index:
                row.update(self.styles_df.loc[pid].to_dict())
            results.append(row)

        return pd.DataFrame(results)

    def _metadata_score(self, query_meta: dict) -> np.ndarray:
        weights = {"articleType": 0.4, "masterCategory": 0.3, "baseColour": 0.2, "gender": 0.1}
        total_meta_score = np.zeros(len(self.ids))
        
        # Use vectorized pandas comparisons instead of a loop
        subset_df = self.styles_df.reindex(self.ids)
        for col, w in weights.items():
            if col in query_meta:
                val = str(query_meta[col]).lower()
                total_meta_score += (subset_df[col].astype(str).str.lower() == val).values * w
        return total_meta_score

def load_styles(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    return df.dropna(subset=["id"]).astype({"id": int})