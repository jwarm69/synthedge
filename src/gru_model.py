"""
GRU (Gated Recurrent Unit) model for cryptocurrency classification.

Based on research showing GRU outperforms LSTM for crypto forecasting:
- MAPE: 0.09% vs 0.11% for LSTM
- Faster training with simpler architecture
- Better at capturing temporal dependencies

Reference: Rodrigues & Machado (2025) - High-Frequency Cryptocurrency Price
Forecasting Using Machine Learning Models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import joblib
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        GRU, Dense, Dropout, BatchNormalization,
        Bidirectional, Input
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
    print("TensorFlow not installed. GRU model unavailable.")


MODELS_DIR = Path(__file__).parent.parent / "models"


def prepare_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for GRU input.

    Args:
        X: Feature matrix
        y: Target variable
        lookback: Number of past timesteps to include

    Returns:
        X_seq: (samples, lookback, features)
        y_seq: (samples,)
    """
    X_values = X.values
    y_values = y.values

    X_seq, y_seq = [], []

    for i in range(lookback, len(X_values)):
        X_seq.append(X_values[i-lookback:i])
        y_seq.append(y_values[i])

    return np.array(X_seq), np.array(y_seq)


def build_gru_classifier(
    input_shape: Tuple[int, int],
    units: int = 64,
    dropout: float = 0.3,
    l2_reg: float = 0.001,
    bidirectional: bool = False
) -> 'Sequential':
    """
    Build GRU model for binary classification (up/down).

    Architecture based on paper findings:
    - 2 GRU layers with BatchNorm and Dropout
    - Dense layers for classification
    - Adam optimizer with early stopping

    Args:
        input_shape: (lookback, n_features)
        units: Number of GRU units per layer
        dropout: Dropout rate
        l2_reg: L2 regularization strength
        bidirectional: Use bidirectional GRU (slower but may be better)
    """
    if not HAS_KERAS:
        raise ImportError("TensorFlow required for GRU model")

    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # First GRU layer
    gru_layer_1 = GRU(
        units,
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    )
    if bidirectional:
        model.add(Bidirectional(gru_layer_1))
    else:
        model.add(gru_layer_1)

    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    # Second GRU layer
    gru_layer_2 = GRU(
        units // 2,
        return_sequences=False,
        kernel_regularizer=l2(l2_reg)
    )
    if bidirectional:
        model.add(Bidirectional(gru_layer_2))
    else:
        model.add(gru_layer_2)

    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    # Dense layers
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout / 2))

    # Output layer (binary classification)
    model.add(Dense(1, activation='sigmoid'))

    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_gru_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    lookback: int = 12,
    epochs: int = 100,
    batch_size: int = 64,
    units: int = 64,
    dropout: float = 0.3,
    patience: int = 15
) -> Tuple['Sequential', dict]:
    """
    Train GRU classifier with early stopping.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        lookback: Sequence length
        epochs: Max training epochs
        batch_size: Batch size
        units: GRU units
        dropout: Dropout rate
        patience: Early stopping patience

    Returns:
        Trained model and training history
    """
    # Prepare sequences
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, lookback)
    X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, lookback)

    print(f"Training sequences: {X_train_seq.shape}")
    print(f"Validation sequences: {X_val_seq.shape}")

    # Build model
    input_shape = (lookback, X_train.shape[1])
    model = build_gru_classifier(input_shape, units=units, dropout=dropout)

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history.history


class GRUClassifierWrapper:
    """
    Wrapper to make GRU model compatible with sklearn-style interface.
    Used for ensemble integration.
    """

    def __init__(
        self,
        lookback: int = 12,
        units: int = 64,
        dropout: float = 0.3,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15
    ):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        """Fit the GRU model."""
        self.feature_names = list(X.columns)

        # Prepare sequences
        X_seq, y_seq = prepare_sequences(X, y, self.lookback)

        # Split for validation if not provided
        if X_val is None:
            split = int(len(X_seq) * 0.8)
            X_train_seq = X_seq[:split]
            y_train_seq = y_seq[:split]
            X_val_seq = X_seq[split:]
            y_val_seq = y_seq[split:]
        else:
            X_train_seq = X_seq
            y_train_seq = y_seq
            X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, self.lookback)

        # Build model
        input_shape = (self.lookback, len(self.feature_names))
        self.model = build_gru_classifier(
            input_shape,
            units=self.units,
            dropout=self.dropout
        )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0
            )
        ]

        # Train
        self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        # Need to pad with lookback history
        if len(X) < self.lookback:
            # Not enough data - return 0.5 probability
            return np.array([[0.5, 0.5]] * len(X))

        X_seq = []
        for i in range(self.lookback, len(X) + 1):
            X_seq.append(X.values[i-self.lookback:i])

        X_seq = np.array(X_seq)
        prob_up = self.model.predict(X_seq, verbose=0).flatten()

        # Pad beginning with 0.5
        full_prob_up = np.concatenate([
            np.full(self.lookback, 0.5),
            prob_up
        ])[:len(X)]

        prob_down = 1 - full_prob_up
        return np.column_stack([prob_down, full_prob_up])

    def save(self, path: Path):
        """Save model."""
        path = Path(path)
        self.model.save(path.with_suffix('.keras'))
        # Save metadata
        meta = {
            'lookback': self.lookback,
            'units': self.units,
            'dropout': self.dropout,
            'feature_names': self.feature_names
        }
        joblib.dump(meta, path.with_suffix('.meta'))

    @classmethod
    def load(cls, path: Path) -> 'GRUClassifierWrapper':
        """Load model."""
        path = Path(path)
        model = load_model(path.with_suffix('.keras'))
        meta = joblib.load(path.with_suffix('.meta'))

        wrapper = cls(
            lookback=meta['lookback'],
            units=meta['units'],
            dropout=meta['dropout']
        )
        wrapper.model = model
        wrapper.feature_names = meta['feature_names']
        return wrapper


if __name__ == "__main__":
    from data_fetch import load_data
    from features import prepare_features, get_feature_columns
    from sklearn.metrics import accuracy_score, roc_auc_score

    print("Testing GRU classifier on BTC data...")

    # Load and prepare data
    df = load_data("btc")
    df = prepare_features(df, include_fgi=True, include_kalman=True)

    feature_cols = get_feature_columns(include_fgi=True, include_kalman=True)
    available_features = [f for f in feature_cols if f in df.columns]

    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[available_features]
    y_train = train_df["target"]
    X_test = test_df[available_features]
    y_test = test_df["target"]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(available_features)}")

    # Train GRU
    gru = GRUClassifierWrapper(lookback=12, epochs=50, patience=10)
    gru.fit(X_train, y_train)

    # Evaluate
    y_pred = gru.predict(X_test)
    y_prob = gru.predict_proba(X_test)[:, 1]

    # Adjust for lookback offset
    y_test_adj = y_test.values[12:]
    y_pred_adj = y_pred[12:]
    y_prob_adj = y_prob[12:]

    acc = accuracy_score(y_test_adj, y_pred_adj)
    auc = roc_auc_score(y_test_adj, y_prob_adj)

    print(f"\nGRU Results:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  AUC: {auc:.4f}")
