"""Inference for the dual-network (two-tower) content-based recommender.

The trained Keras model (model/new_model.h5) has two towers:
    user network: 14 genre preferences        -> 256 -> 128 -> 32, L2-normalised
    item network: 16 movie content features   -> 256 -> 128 -> 32, L2-normalised
and predicts a rating as the dot product of the two embeddings.

The weights are read straight from the .h5 file with h5py and the forward pass
is done in NumPy, so serving needs no TensorFlow. tests/test_recommender.py
checks that the NumPy output matches Keras' model.predict().
"""
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_utils import (DATA_DIR, GENRES, ITEM_FEATURES_START, USER_FEATURES_START,
                        USER_ID, load_data)

MODEL_PATH = Path(__file__).resolve().parent / "model" / "new_model.h5"

LIKED = 3.0         # a genre rated at least this counts as "liked" for retrieval
NUM_CANDIDATES = 80  # size of the retrieval shortlist that gets ranked


def load_towers(model_path=MODEL_PATH):
    """Return (user_tower, item_tower), each a list of (kernel, bias) per dense layer."""
    towers = []
    with h5py.File(model_path, "r") as f:
        config = json.loads(f.attrs["model_config"])
        weights = f["model_weights"]
        # Each tower is a Sequential sub-model; its config lists the Dense layers in order.
        for layer in config["config"]["layers"]:
            if layer["class_name"] != "Sequential":
                continue
            group = weights[layer["config"]["name"]]
            dense = [sub["config"]["name"] for sub in layer["config"]["layers"]
                     if sub["class_name"] == "Dense"]
            towers.append([(np.array(group[d]["kernel:0"]), np.array(group[d]["bias:0"]))
                           for d in dense])
    # The user tower takes 14 inputs (genres), the item tower 16.
    towers.sort(key=lambda layers: layers[0][0].shape[0])
    user_tower, item_tower = towers
    assert user_tower[0][0].shape[0] == len(GENRES)
    assert item_tower[0][0].shape[0] == len(GENRES) + 2
    return user_tower, item_tower


def forward(tower, x):
    """Dense(ReLU) -> Dense(ReLU) -> Dense, then L2-normalise, as in training."""
    for kernel, bias in tower[:-1]:
        x = np.maximum(x @ kernel + bias, 0.0)
    kernel, bias = tower[-1]
    v = x @ kernel + bias
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class Recommender:
    def __init__(self, model_path=MODEL_PATH, data_dir=DATA_DIR):
        self.data = data = load_data(data_dir)
        self.movies = data.movies

        # Refit the scalers exactly as they were fitted before training.
        self.user_scaler = StandardScaler().fit(data.user_train)
        self.item_scaler = StandardScaler().fit(data.item_train)
        self.target_scaler = MinMaxScaler((-1, 1)).fit(data.y_train.reshape(-1, 1))

        self.user_tower, self.item_tower = load_towers(model_path)

        # Movie embeddings don't depend on the user: compute them once.
        scaled_items = self.item_scaler.transform(data.item_vecs)
        self.item_embeddings = forward(self.item_tower, scaled_items[:, ITEM_FEATURES_START:])
        self.movie_ids = data.item_vecs[:, 0].astype(int)
        self.row_of = {movie_id: row for row, movie_id in enumerate(self.movie_ids)}
        self.genre_flags = data.item_vecs[:, -len(GENRES):] > 0

    # ---- embeddings and scores -------------------------------------------

    def user_embedding(self, genre_ratings):
        """Embed a user from their average rating per genre (0 = not rated)."""
        prefs = np.array([float(genre_ratings.get(g, 0.0)) for g in GENRES])
        user_vec = np.zeros((1, self.data.user_train.shape[1]))
        user_vec[0, USER_FEATURES_START:] = prefs
        scaled = self.user_scaler.transform(user_vec)
        return forward(self.user_tower, scaled[:, USER_FEATURES_START:])

    def _to_rating(self, dots):
        return self.target_scaler.inverse_transform(dots.reshape(-1, 1)).ravel()

    def predict(self, genre_ratings):
        """Predicted rating (0.5 - 5.0) for every movie in the catalogue."""
        v_u = self.user_embedding(genre_ratings)
        return self._to_rating(self.item_embeddings @ v_u.ravel())

    # ---- retrieval and ranking -------------------------------------------

    def retrieve(self, genre_ratings, n=NUM_CANDIDATES):
        """Cheap shortlist: movies sharing the most genres the user likes."""
        liked = np.array([genre_ratings.get(g, 0.0) >= LIKED for g in GENRES])
        if not liked.any():
            return np.arange(len(self.movie_ids))
        overlap = (self.genre_flags & liked).sum(axis=1)
        order = np.argsort(-overlap, kind="stable")
        return order[overlap[order] > 0][:n]

    def recommend(self, genre_ratings, n=10):
        candidates = self.retrieve(genre_ratings)
        predicted = self.predict(genre_ratings)
        ranked = candidates[np.argsort(-predicted[candidates], kind="stable")][:n]
        return [self._movie(row, predicted=float(predicted[row]),
                            similar=self.similar(self.movie_ids[row], n=1)[0])
                for row in ranked]

    def similar(self, movie_id, n=5):
        """Nearest movies in the trained item-embedding space."""
        row = self.row_of[int(movie_id)]
        dist = ((self.item_embeddings - self.item_embeddings[row]) ** 2).sum(axis=1)
        dist[row] = np.inf
        return [self._movie(r, distance=float(dist[r])) for r in np.argsort(dist)[:n]]

    # ---- evaluating on a MovieLens user ------------------------------------

    def user_ids(self):
        return np.unique(self.data.user_train[:, USER_ID]).astype(int)

    def training_user(self, user_id):
        """A MovieLens user's real ratings next to what the model predicts."""
        rows = np.flatnonzero(self.data.user_train[:, USER_ID] == user_id)
        if rows.size == 0:
            raise KeyError(user_id)
        user_vec = self.data.user_train[rows[0]]
        genre_ratings = dict(zip(GENRES, user_vec[USER_FEATURES_START:]))
        predicted = self.predict(genre_ratings)
        rated = {}  # the training matrices repeat some (user, movie) rows
        for i in rows:
            row = self.row_of[int(self.data.item_train[i, 0])]
            rated[row] = self._movie(row, actual=float(self.data.y_train[i]),
                                     predicted=float(predicted[row]))
        return genre_ratings, sorted(rated.values(), key=lambda m: -m["actual"])

    # ---- helpers ----------------------------------------------------------

    def _movie(self, row, **extra):
        movie_id = int(self.movie_ids[row])
        return {"id": movie_id, "title": self.movies[movie_id]["title"],
                "genres": self.movies[movie_id]["genres"].replace("|", " · "),
                "year": int(self.data.item_vecs[row, 1]), **extra}

    def movie(self, movie_id):
        return self._movie(self.row_of[int(movie_id)])

    def search(self, query, limit=20):
        query = query.lower().strip()
        hits = [row for row, movie_id in enumerate(self.movie_ids)
                if query in self.movies[movie_id]["title"].lower()]
        return [self._movie(row) for row in hits[:limit]]
