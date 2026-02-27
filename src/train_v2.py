"""
Enhanced training script v2 with:
- Purged time-series cross-validation
- Feature selection
- Extended hyperparameter tuning
- LSTM ensemble option
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import warnings
from typing import Optional
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

# Try CatBoost
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Try TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False
    print("TensorFlow not installed. LSTM will be skipped.")

from data_fetch import load_data, SYMBOLS
from features import prepare_features, get_feature_columns
from ensemble import EnsembleClassifier
from experiment import ExperimentTracker, load_config

MODELS_DIR = Path(__file__).parent.parent / "models"
optuna.logging.set_verbosity(optuna.logging.WARNING)


class PurgedTimeSeriesCV:
    """
    Purged time-series cross-validation.

    Key differences from regular CV:
    1. Always train on past, validate on future
    2. Purge gap between train and validation to prevent leakage
    3. Embargo period after validation to ensure no overlap
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 24, embargo: int = 12):
        """
        Args:
            n_splits: Number of CV splits
            purge_gap: Hours to purge between train and validation
            embargo: Hours to embargo after validation
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        """Generate train/validation indices with purging."""
        n_samples = len(X)

        # Calculate fold size (validation set size)
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Training set: from start to current fold
            train_end = fold_size * (i + 1)
            train_indices = np.arange(0, train_end - self.purge_gap)

            # Validation set: after purge gap
            val_start = train_end
            val_end = min(val_start + fold_size, n_samples - self.embargo)
            val_indices = np.arange(val_start, val_end)

            if len(train_indices) > 0 and len(val_indices) > 0:
                yield train_indices, val_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def select_features(X: pd.DataFrame, y: pd.Series, top_k: int = 30) -> list:
    """
    Select top features using mutual information.

    Args:
        X: Feature matrix
        y: Target variable
        top_k: Number of features to select

    Returns:
        List of selected feature names
    """
    # Calculate mutual information
    mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)

    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    # Select top features
    selected = feature_importance.head(top_k)['feature'].tolist()

    print(f"\nTop {top_k} features by mutual information:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['mi_score']:.4f}")

    return selected


def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 100) -> dict:
    """Extended Optuna optimization for XGBoost."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest XGBoost AUC: {study.best_value:.4f}")
    return study.best_params


def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials: int = 100) -> dict:
    """Extended Optuna optimization for LightGBM."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        y_pred = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred)

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest LightGBM AUC: {study.best_value:.4f}")
    return study.best_params


def build_lstm_model(input_shape, units=64, dropout=0.3):
    """Build LSTM model for sequence prediction."""
    model = Sequential([
        LSTM(units, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(units // 2),
        BatchNormalization(),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def prepare_lstm_sequences(X: pd.DataFrame, y: pd.Series, lookback: int = 12):
    """
    Prepare sequences for LSTM.

    Args:
        X: Feature matrix
        y: Target variable
        lookback: Number of past hours to include

    Returns:
        X_seq, y_seq as numpy arrays
    """
    X_values = X.values
    y_values = y.values

    X_seq, y_seq = [], []

    for i in range(lookback, len(X_values)):
        X_seq.append(X_values[i-lookback:i])
        y_seq.append(y_values[i])

    return np.array(X_seq), np.array(y_seq)


def train_with_purged_cv(
    coin: str,
    n_trials: int = 200,
    n_cv_splits: int = 5,
    feature_selection: bool = True,
    top_k_features: int = 35,
    use_lstm: bool = True,
    tracker: Optional[ExperimentTracker] = None
) -> dict:
    """
    Train with all improvements.

    Args:
        coin: Coin to train
        n_trials: Optuna trials per model
        n_cv_splits: Number of CV splits
        feature_selection: Whether to use feature selection
        top_k_features: Number of features to select
        use_lstm: Whether to include LSTM in ensemble

    Returns:
        Dict with results
    """
    print(f"\n{'='*60}")
    print(f"TRAINING V2 FOR {coin.upper()}")
    print(f"{'='*60}")

    if tracker:
        tracker.start(
            run_name=f"{coin}_train_v2",
            config={
                "coin": coin,
                "n_trials": n_trials,
                "n_cv_splits": n_cv_splits,
                "feature_selection": feature_selection,
                "top_k_features": top_k_features,
                "use_lstm": use_lstm
            }
        )

    # Load data
    df = load_data(coin)

    # For altcoins, load BTC for cross-asset features
    btc_df = None
    include_cross_asset = coin.lower() != "btc"
    if include_cross_asset:
        btc_df = load_data("btc")

    # Prepare features with all improvements
    df = prepare_features(
        df,
        lookahead=1,
        btc_df=btc_df,
        include_fgi=True,
        include_onchain=coin.lower() == "btc",
        include_lags=True,
        include_multitimeframe=True,
        include_macro=False,
        include_flows=False,
        include_orderbook=False
    )

    print(f"Total samples: {len(df)}")

    # Get all feature columns
    feature_cols = get_feature_columns(
        include_fgi=True,
        include_cross_asset=include_cross_asset,
        include_onchain=coin.lower() == "btc",
        include_lags=True,
        include_multitimeframe=True,
        include_macro=False,
        include_flows=False,
        include_orderbook=False
    )

    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Available features: {len(available_features)}")

    # Split data (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_all = train_df[available_features]
    y_all = train_df["target"]
    X_test = test_df[available_features]
    y_test = test_df["target"]

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Feature selection
    if feature_selection:
        print(f"\nSelecting top {top_k_features} features...")
        selected_features = select_features(X_all, y_all, top_k=top_k_features)
        X_all = X_all[selected_features]
        X_test = X_test[selected_features]
    else:
        selected_features = available_features

    # Use purged CV for optimization
    cv = PurgedTimeSeriesCV(n_splits=n_cv_splits, purge_gap=24, embargo=12)

    # Get one train/val split for hyperparameter tuning
    splits = list(cv.split(X_all))
    train_idx, val_idx = splits[-1]  # Use last split for tuning

    X_train = X_all.iloc[train_idx]
    y_train = y_all.iloc[train_idx]
    X_val = X_all.iloc[val_idx]
    y_val = y_all.iloc[val_idx]

    print(f"\nOptimization split - Train: {len(X_train)}, Val: {len(X_val)}")

    # Optimize XGBoost
    print("\n--- Optimizing XGBoost ---")
    xgb_params = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=n_trials)

    # Optimize LightGBM
    print("\n--- Optimizing LightGBM ---")
    lgb_params = optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=n_trials)

    # Cross-validated training
    print("\n--- Cross-Validated Training ---")

    cv_scores = {'xgb': [], 'lgb': [], 'cat': [], 'lstm': []}
    oof_targets = []
    oof_probs = {'xgb': [], 'lgb': [], 'cat': []}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_all)):
        print(f"\nFold {fold + 1}/{n_cv_splits}")

        X_tr = X_all.iloc[train_idx]
        y_tr = y_all.iloc[train_idx]
        X_va = X_all.iloc[val_idx]
        y_va = y_all.iloc[val_idx]

        # XGBoost
        xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42, n_jobs=-1,
                                       use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        xgb_auc = roc_auc_score(y_va, xgb_model.predict_proba(X_va)[:, 1])
        cv_scores['xgb'].append(xgb_auc)
        oof_probs['xgb'].append(xgb_model.predict_proba(X_va)[:, 1])

        # LightGBM
        lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42, n_jobs=-1, verbose=-1)
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        lgb_auc = roc_auc_score(y_va, lgb_model.predict_proba(X_va)[:, 1])
        cv_scores['lgb'].append(lgb_auc)
        oof_probs['lgb'].append(lgb_model.predict_proba(X_va)[:, 1])

        # CatBoost
        if HAS_CATBOOST:
            cat_model = CatBoostClassifier(
                iterations=xgb_params.get('n_estimators', 100),
                depth=min(10, xgb_params.get('max_depth', 6)),
                learning_rate=xgb_params.get('learning_rate', 0.1),
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            )
            cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
            cat_auc = roc_auc_score(y_va, cat_model.predict_proba(X_va)[:, 1])
            cv_scores['cat'].append(cat_auc)
            oof_probs['cat'].append(cat_model.predict_proba(X_va)[:, 1])

        oof_targets.append(y_va.values)

        print(f"  XGB AUC: {xgb_auc:.4f}, LGB AUC: {lgb_auc:.4f}", end="")
        if HAS_CATBOOST:
            print(f", CAT AUC: {cat_auc:.4f}")
        else:
            print()

    # Print CV results
    print("\n--- Cross-Validation Results ---")
    for name, scores in cv_scores.items():
        if scores:
            print(f"{name.upper()}: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.2f}%)")

    # Final training on all training data
    print("\n--- Final Model Training ---")

    # Use purge between train and test
    X_train_final = X_all.iloc[:-24]  # Leave 24h gap before test
    y_train_final = y_all.iloc[:-24]

    # Train final models
    xgb_final = xgb.XGBClassifier(**xgb_params, random_state=42, n_jobs=-1,
                                   use_label_encoder=False, eval_metric='logloss')
    xgb_final.fit(X_train_final, y_train_final, verbose=False)

    lgb_final = lgb.LGBMClassifier(**lgb_params, random_state=42, n_jobs=-1, verbose=-1)
    lgb_final.fit(X_train_final, y_train_final)

    cat_final = None
    if HAS_CATBOOST:
        cat_final = CatBoostClassifier(
            iterations=xgb_params.get('n_estimators', 100),
            depth=min(10, xgb_params.get('max_depth', 6)),
            learning_rate=xgb_params.get('learning_rate', 0.1),
            random_seed=42,
            verbose=False,
            allow_writing_files=False
        )
        cat_final.fit(X_train_final, y_train_final)

    # LSTM (optional)
    lstm_final = None
    if use_lstm and HAS_KERAS:
        print("Training LSTM...")
        lookback = 12

        # Prepare sequences
        X_seq, y_seq = prepare_lstm_sequences(X_train_final, y_train_final, lookback)
        X_test_seq, y_test_seq = prepare_lstm_sequences(X_test, y_test, lookback)

        if len(X_seq) > 100:
            lstm_final = build_lstm_model((lookback, X_seq.shape[2]))
            early_stop = EarlyStopping(patience=10, restore_best_weights=True)

            lstm_final.fit(
                X_seq, y_seq,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )

    # Create ensemble with weighted averaging
    models = {'xgb': xgb_final, 'lgb': lgb_final}
    weights = {
        'xgb': np.mean(cv_scores['xgb']) if cv_scores['xgb'] else 0.5,
        'lgb': np.mean(cv_scores['lgb']) if cv_scores['lgb'] else 0.5,
    }

    if cat_final:
        models['cat'] = cat_final
        weights['cat'] = np.mean(cv_scores['cat']) if cv_scores['cat'] else 0.4

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}

    ensemble = EnsembleClassifier(models, weights)

    # Fit calibration models on OOF ensemble probabilities
    calibration_path = MODELS_DIR / f"{coin}_calibration.joblib"
    calibration_meta = {}
    try:
        oof_y = np.concatenate(oof_targets)
        oof_xgb = np.concatenate(oof_probs['xgb']) if oof_probs['xgb'] else None
        oof_lgb = np.concatenate(oof_probs['lgb']) if oof_probs['lgb'] else None
        oof_cat = np.concatenate(oof_probs['cat']) if oof_probs['cat'] else None

        oof_ens = 0
        weight_sum = 0
        if oof_xgb is not None:
            oof_ens += weights.get('xgb', 0) * oof_xgb
            weight_sum += weights.get('xgb', 0)
        if oof_lgb is not None:
            oof_ens += weights.get('lgb', 0) * oof_lgb
            weight_sum += weights.get('lgb', 0)
        if oof_cat is not None:
            oof_ens += weights.get('cat', 0) * oof_cat
            weight_sum += weights.get('cat', 0)

        oof_ens = oof_ens / max(weight_sum, 1e-8)

        platt = LogisticRegression(solver="lbfgs", random_state=42)
        platt.fit(oof_ens.reshape(-1, 1), oof_y)

        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(oof_ens, oof_y)

        joblib.dump({"platt": platt, "isotonic": isotonic}, calibration_path)

        calibration_meta = {
            "calibration_path": str(calibration_path),
            "brier_raw": float(brier_score_loss(oof_y, oof_ens)),
            "brier_platt": float(brier_score_loss(oof_y, platt.predict_proba(oof_ens.reshape(-1, 1))[:, 1])),
            "brier_isotonic": float(brier_score_loss(oof_y, isotonic.predict(oof_ens))),
            "calibration_set": "oof_cv"
        }
    except Exception as e:
        print(f"Warning: Calibration fit failed: {e}")

    # Evaluate on test set
    print("\n--- TEST SET RESULTS ---")

    results = {}

    # Individual models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        print(f"{name.upper():4}: Accuracy={acc*100:.2f}%, AUC={auc:.4f}")
        results[f'{name}_accuracy'] = acc
        results[f'{name}_auc'] = auc

    # Ensemble
    y_pred_ens = ensemble.predict(X_test)
    y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
    acc_ens = accuracy_score(y_test, y_pred_ens)
    auc_ens = roc_auc_score(y_test, y_prob_ens)
    print(f"{'ENS':4}: Accuracy={acc_ens*100:.2f}%, AUC={auc_ens:.4f}")

    results['ensemble_accuracy'] = acc_ens
    results['ensemble_auc'] = auc_ens

    # LSTM evaluation
    if lstm_final and len(X_test_seq) > 0:
        lstm_prob = lstm_final.predict(X_test_seq, verbose=0).flatten()
        lstm_pred = (lstm_prob > 0.5).astype(int)
        lstm_acc = accuracy_score(y_test_seq, lstm_pred)
        lstm_auc = roc_auc_score(y_test_seq, lstm_prob)
        print(f"{'LSTM':4}: Accuracy={lstm_acc*100:.2f}%, AUC={lstm_auc:.4f}")
        results['lstm_accuracy'] = lstm_acc
        results['lstm_auc'] = lstm_auc

    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(ensemble, MODELS_DIR / f"{coin}_ensemble_v2.joblib")
    joblib.dump(xgb_final, MODELS_DIR / f"{coin}_xgb_v2.joblib")
    joblib.dump(lgb_final, MODELS_DIR / f"{coin}_lgb_v2.joblib")
    if cat_final:
        joblib.dump(cat_final, MODELS_DIR / f"{coin}_cat_v2.joblib")
    if lstm_final:
        lstm_final.save(MODELS_DIR / f"{coin}_lstm_v2.keras")

    # Save metadata
    meta = {
        "coin": coin,
        "version": "v2",
        "trained_at": datetime.now().isoformat(),
        "features": selected_features,
        "n_features": len(selected_features),
        "include_fgi": True,
        "include_cross_asset": include_cross_asset,
        "include_onchain": coin.lower() == "btc",
        "include_lags": True,
        "include_multitimeframe": True,
        "include_macro": False,
        "include_flows": False,
        "include_orderbook": False,
        "model_type": "ensemble_v2",
        "models": list(models.keys()),
        "weights": weights,
        "xgb_params": xgb_params,
        "lgb_params": lgb_params,
        "calibration": calibration_meta,
        "cv_scores": {k: [float(s) for s in v] for k, v in cv_scores.items() if v},
        "test_accuracy": float(acc_ens),
        "test_auc": float(auc_ens),
        "n_trials": n_trials,
        "n_cv_splits": n_cv_splits,
    }

    with open(MODELS_DIR / f"{coin}_meta_v2.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Also save as the main model
    joblib.dump(ensemble, MODELS_DIR / f"{coin}_ensemble.joblib")
    with open(MODELS_DIR / f"{coin}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved models to {MODELS_DIR}")

    if tracker:
        tracker.log_metrics({
            "ensemble_accuracy": float(acc_ens),
            "ensemble_auc": float(auc_ens),
            "xgb_auc_mean": float(np.mean(cv_scores['xgb'])) if cv_scores['xgb'] else 0.0,
            "lgb_auc_mean": float(np.mean(cv_scores['lgb'])) if cv_scores['lgb'] else 0.0
        })
        if calibration_meta:
            tracker.log_metrics({
                "brier_raw": calibration_meta.get("brier_raw", 0.0),
                "brier_platt": calibration_meta.get("brier_platt", 0.0),
                "brier_isotonic": calibration_meta.get("brier_isotonic", 0.0)
            })
        tracker.log_artifact(MODELS_DIR / f"{coin}_meta_v2.json")
        tracker.end()

    return results


def train_all_v2(n_trials: int = 200, tracker: Optional[ExperimentTracker] = None, **kwargs):
    """Train all coins with v2 improvements."""
    all_results = {}

    for coin in SYMBOLS.keys():
        try:
            results = train_with_purged_cv(coin, n_trials=n_trials, tracker=tracker, **kwargs)
            all_results[coin] = results
        except Exception as e:
            print(f"\nError training {coin}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING V2 SUMMARY")
    print("=" * 60)

    for coin, results in all_results.items():
        print(f"\n{coin.upper()}:")
        print(f"  Ensemble: {results.get('ensemble_accuracy', 0)*100:.2f}% (AUC: {results.get('ensemble_auc', 0):.4f})")

    # Average accuracy
    avg_acc = np.mean([r.get('ensemble_accuracy', 0) for r in all_results.values()])
    avg_auc = np.mean([r.get('ensemble_auc', 0) for r in all_results.values()])
    print(f"\nAVERAGE: {avg_acc*100:.2f}% (AUC: {avg_auc:.4f})")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train models v2 with improvements")
    parser.add_argument("--coin", type=str, default="all", help="Coin to train (btc, eth, doge, zec, or 'all')")
    parser.add_argument("--trials", type=int, default=200, help="Optuna trials per model")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--top-k", type=int, default=35, help="Top K features to select")
    parser.add_argument("--no-feature-selection", action="store_true", help="Disable feature selection")
    parser.add_argument("--no-lstm", action="store_true", help="Disable LSTM")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--track-provider", type=str, help="Tracking provider (mlflow|wandb|none)")
    parser.add_argument("--track-project", type=str, help="Tracking project name")

    args = parser.parse_args()

    cfg = load_config(args.config)
    training_cfg = cfg.get("training", {})
    tracking_cfg = cfg.get("tracking", {})

    if training_cfg:
        args.coin = training_cfg.get("coin", args.coin)
        args.trials = training_cfg.get("trials", args.trials)
        args.cv_splits = training_cfg.get("cv_splits", args.cv_splits)
        args.top_k = training_cfg.get("top_k", args.top_k)
        if training_cfg.get("feature_selection") is False:
            args.no_feature_selection = True
        if training_cfg.get("use_lstm") is False:
            args.no_lstm = True

    provider = args.track_provider or tracking_cfg.get("provider", "none")
    project = args.track_project or tracking_cfg.get("project")
    tracker = ExperimentTracker(provider=provider, project=project)

    if args.coin == "all":
        train_all_v2(
            n_trials=args.trials,
            n_cv_splits=args.cv_splits,
            feature_selection=not args.no_feature_selection,
            top_k_features=args.top_k,
            use_lstm=not args.no_lstm,
            tracker=tracker if provider != "none" else None
        )
    else:
        train_with_purged_cv(
            args.coin,
            n_trials=args.trials,
            n_cv_splits=args.cv_splits,
            feature_selection=not args.no_feature_selection,
            top_k_features=args.top_k,
            use_lstm=not args.no_lstm,
            tracker=tracker if provider != "none" else None
        )
