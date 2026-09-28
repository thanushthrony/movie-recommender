"""Export the trained model to demo/model-data.js for the in-browser demo.

GitHub Pages can't run Python, so the demo precomputes what doesn't depend on
the visitor (all 847 movie embeddings from the trained item network) and ships
the small user network's weights plus the scaler parameters. The browser then
runs the user network and 847 dot products itself.

    python export_model_for_web.py
"""
import json
from pathlib import Path

import numpy as np

from data_utils import GENRES, USER_FEATURES_START
from recommender import Recommender

OUT = Path(__file__).resolve().parent / "demo" / "model-data.js"


def main():
    rec = Recommender()
    (W1, b1), (W2, b2), (W3, b3) = rec.user_tower
    r4 = lambda a: np.round(a, 4).tolist()
    r5 = lambda a: np.round(a, 5).tolist()

    payload = {
        "genres": GENRES,
        "movies": [
            {"id": int(v[0]), "t": rec.movies[int(v[0])]["title"],
             "g": rec.movies[int(v[0])]["genres"], "y": int(v[1]),
             "r": round(float(v[2]), 2), "gv": [int(x) for x in v[3:]]}
            for v in rec.data.item_vecs
        ],
        "vm": r4(rec.item_embeddings),
        "userTower": {"W1": r5(W1), "b1": r5(b1), "W2": r5(W2), "b2": r5(b2),
                      "W3": r5(W3), "b3": r5(b3)},
        "userScaler": {"mean": r5(rec.user_scaler.mean_[USER_FEATURES_START:]),
                       "scale": r5(rec.user_scaler.scale_[USER_FEATURES_START:])},
        "target": {"min": float(rec.target_scaler.data_min_[0]),
                   "max": float(rec.target_scaler.data_max_[0])},
    }
    OUT.write_text("const MODEL_DATA = " + json.dumps(payload, separators=(",", ":")) + ";")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(payload['movies'])} movies)")


if __name__ == "__main__":
    main()
