import numpy as np
import pytest

from data_utils import GENRES, ITEM_FEATURES_START, USER_FEATURES_START
from recommender import MODEL_PATH

SCI_FI_FAN = {"Action": 5.0, "Sci-Fi": 5.0}
FAMILY = {"Animation": 5.0, "Children": 5.0, "Comedy": 4.0}


def test_predictions_are_valid_ratings(rec):
    for prefs in ({}, SCI_FI_FAN, FAMILY):
        predicted = rec.predict(prefs)
        assert predicted.shape == (len(rec.movie_ids),)
        assert predicted.min() >= 0.5 and predicted.max() <= 5.0


def test_item_embeddings_are_unit_vectors(rec):
    assert rec.item_embeddings.shape == (len(rec.movie_ids), 32)
    np.testing.assert_allclose(np.linalg.norm(rec.item_embeddings, axis=1), 1.0, rtol=1e-6)


def test_numpy_forward_pass_matches_keras(rec):
    tf = pytest.importorskip("tensorflow")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    rng = np.random.default_rng(0)
    users = rec.user_scaler.transform(rec.data.user_train[rng.choice(len(rec.data.user_train), 64)])
    items = rec.item_scaler.transform(rec.data.item_vecs[rng.choice(len(rec.data.item_vecs), 64)])
    user_x, item_x = users[:, USER_FEATURES_START:], items[:, ITEM_FEATURES_START:]

    keras_out = model.predict([user_x, item_x], verbose=0).ravel()
    from recommender import forward
    numpy_out = (forward(rec.user_tower, user_x) * forward(rec.item_tower, item_x)).sum(axis=1)
    np.testing.assert_allclose(numpy_out, keras_out, atol=1e-5)


def test_user_preferences_drive_recommendations(rec):
    sci_fi = rec.recommend(SCI_FI_FAN, n=10)
    family = rec.recommend(FAMILY, n=10)
    assert len(sci_fi) == len(family) == 10
    # Retrieval only shortlists movies sharing a liked genre.
    assert all("Action" in m["genres"] or "Sci-Fi" in m["genres"] for m in sci_fi)
    assert {m["id"] for m in sci_fi}.isdisjoint({m["id"] for m in family})
    scores = [m["predicted"] for m in sci_fi]
    assert scores == sorted(scores, reverse=True)


def test_similar_movies_use_trained_embeddings(rec):
    # With the untrained item network in the original app.py this failed.
    spider_man = rec.search("Spider-Man (2002)")[0]
    assert rec.similar(spider_man["id"], n=1)[0]["title"] == "Spider-Man 2 (2004)"


def test_training_user_predictions_are_close(rec):
    genre_ratings, rated = rec.training_user(2)
    assert set(genre_ratings) == set(GENRES)
    assert len({m["id"] for m in rated}) == len(rated)
    mae = np.mean([abs(m["predicted"] - m["actual"]) for m in rated])
    assert mae < 1.0


def test_unknown_training_user(rec):
    with pytest.raises(KeyError):
        rec.training_user(-1)
