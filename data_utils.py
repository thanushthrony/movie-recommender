"""Loading the preprocessed MovieLens training data.

Every row i of the training matrices is one observed rating:
    user_train[i] = [user id, rating count, rating average, 14 per-genre averages]
    item_train[i] = [movie id, year, average rating, 14 one-hot genre flags]
    y_train[i]    = the rating that user gave that movie (0.5 - 5.0)
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Horror", "Mystery", "Romance",
    "Sci-Fi", "Thriller",
]

# Column layout of the training matrices
USER_ID, USER_COUNT, USER_AVE = 0, 1, 2
USER_FEATURES_START = 3  # user network sees the 14 genre columns only
ITEM_FEATURES_START = 1  # item network sees year, average rating and genres


@dataclass
class Dataset:
    item_train: np.ndarray
    user_train: np.ndarray
    y_train: np.ndarray
    item_vecs: np.ndarray  # one row per movie in the catalogue (847)
    movies: dict           # movie id -> {"title": str, "genres": str}
    item_features: list
    user_features: list


def _read_header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f))


def load_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    item_train = np.genfromtxt(data_dir / "content_item_train.csv", delimiter=",")
    user_train = np.genfromtxt(data_dir / "content_user_train.csv", delimiter=",")
    y_train = np.genfromtxt(data_dir / "content_y_train.csv", delimiter=",")
    item_vecs = np.genfromtxt(data_dir / "content_item_vecs.csv", delimiter=",")

    movies = {}
    with open(data_dir / "content_movie_list.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for movie_id, title, genres in reader:
            movies[int(movie_id)] = {"title": title, "genres": genres}

    return Dataset(
        item_train=item_train,
        user_train=user_train,
        y_train=y_train,
        item_vecs=item_vecs,
        movies=movies,
        item_features=_read_header(data_dir / "content_item_train_header.txt"),
        user_features=_read_header(data_dir / "content_user_train_header.txt"),
    )
