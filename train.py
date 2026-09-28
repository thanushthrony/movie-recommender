"""Train the two-tower recommender and save it as a Keras .h5 file.

    pip install -r requirements-train.txt
    python train.py                       # thesis settings -> model/retrained.h5
    python train.py --l2 1e-4 --out model/l2_model.h5

The shipped model/new_model.h5 was trained with these settings and no L2 penalty.
"""
import argparse

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_utils import ITEM_FEATURES_START, USER_FEATURES_START, load_data


def tower(num_outputs, l2):
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(num_outputs, kernel_regularizer=reg),
    ])


def build_model(num_user_features, num_item_features, num_outputs=32, l2=0.0):
    user_input = tf.keras.layers.Input(shape=(num_user_features,))
    v_u = tf.linalg.l2_normalize(tower(num_outputs, l2)(user_input), axis=1)

    item_input = tf.keras.layers.Input(shape=(num_item_features,))
    v_m = tf.linalg.l2_normalize(tower(num_outputs, l2)(item_input), axis=1)

    # predicted (scaled) rating = v_u . v_m
    output = tf.keras.layers.Dot(axes=1)([v_u, v_m])
    return tf.keras.Model([user_input, item_input], output)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.0, help="L2 weight penalty (λ)")
    parser.add_argument("--out", default="model/retrained.h5")
    args = parser.parse_args()

    data = load_data()

    # Standardise features and scale ratings to [-1, 1] to match the
    # range of a dot product of two unit vectors.
    item_train = StandardScaler().fit_transform(data.item_train)
    user_train = StandardScaler().fit_transform(data.user_train)
    y_train = MinMaxScaler((-1, 1)).fit_transform(data.y_train.reshape(-1, 1))

    # Same seed for all three splits keeps the rows aligned.
    item_train, item_test = train_test_split(item_train, train_size=0.8, shuffle=True, random_state=1)
    user_train, user_test = train_test_split(user_train, train_size=0.8, shuffle=True, random_state=1)
    y_train, y_test = train_test_split(y_train, train_size=0.8, shuffle=True, random_state=1)
    print(f"train: {len(y_train)} ratings, test: {len(y_test)} ratings")

    tf.random.set_seed(1)
    model = build_model(user_train.shape[1] - USER_FEATURES_START,
                        item_train.shape[1] - ITEM_FEATURES_START, l2=args.l2)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(name="mse"),
                           tf.keras.metrics.MeanAbsoluteError(name="mae")])

    model.fit([user_train[:, USER_FEATURES_START:], item_train[:, ITEM_FEATURES_START:]],
              y_train, epochs=args.epochs)

    _, mse, mae = model.evaluate([user_test[:, USER_FEATURES_START:], item_test[:, ITEM_FEATURES_START:]],
                               y_test, verbose=0)
    # Ratings span 0.5 - 5.0, i.e. 4.5 stars over the scaled range of 2.
    print(f"test MSE (scaled): {mse:.4f}   test MAE: {mae * 4.5 / 2:.3f} stars")

    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
