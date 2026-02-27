"""
Ensemble model training with XGBoost, LightGBM, and CatBoost.
Combines predictions from multiple models for more stable results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

import xgboost as xgb
import lightgbm as lgb

# Try to import CatBoost (optional)
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed. Using XGBoost + LightGBM only.")

from data_fetch import load_data, SYMBOLS
from features import prepare_features, get_feature_columns

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_tuned_params(coin: str) -> dict:
    """Load Optuna-tuned hyperparameters if available."""
    params_file = MODELS_DIR / f"{coin}_best_params.json"
    if params_file.exists():
        with open(params_file) as f:
            data = json.load(f)
            return data.get("params", {})
    return {}


def train_xgboost(X_train, y_train, X_val, y_val, params: dict = None):
    """Train XGBoost classifier."""
    default_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, params: dict = None):
    """Train LightGBM classifier."""
    default_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        # Map XGBoost params to LightGBM
        if "min_child_weight" in params:
            default_params["min_child_samples"] = max(1, int(params["min_child_weight"] * 2))
        for key in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]:
            if key in params:
                default_params[key] = params[key]

    model = lgb.LGBMClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    return model


def train_catboost(X_train, y_train, X_val, y_val, params: dict = None):
    """Train CatBoost classifier."""
    if not HAS_CATBOOST:
        return None

    default_params = {
        "iterations": 100,
        "depth": 5,
        "learning_rate": 0.1,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    if params:
        if "n_estimators" in params:
            default_params["iterations"] = params["n_estimators"]
        if "max_depth" in params:
            default_params["depth"] = min(16, params["max_depth"])  # CatBoost max is 16
        if "learning_rate" in params:
            default_params["learning_rate"] = params["learning_rate"]

    model = CatBoostClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
    )
    return model


class EnsembleClassifier:
    """Ensemble of XGBoost, LightGBM, and optionally CatBoost."""

    def __init__(self, models: dict, weights: dict = None):
        """
        Args:
            models: Dict of {"xgb": model, "lgb": model, "cat": model}
            weights: Dict of model weights for averaging (default: equal)
        """
        self.models = models
        self.weights = weights or {name: 1.0 / len(models) for name in models}

    def predict_proba(self, X):
        """Average prediction probabilities across models."""
        probs = []
        total_weight = 0

        for name, model in self.models.items():
            if model is not None:
                weight = self.weights.get(name, 1.0)
                probs.append(model.predict_proba(X) * weight)
                total_weight += weight

        if not probs:
            raise ValueError("No models available for prediction")

        avg_probs = np.sum(probs, axis=0) / total_weight
        return avg_probs

    def predict(self, X):
        """Predict class labels."""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


def walk_forward_split(df: pd.DataFrame, train_months: int = 8, val_months: int = 2):
    """
    Split data using walk-forward validation.

    Args:
        df: DataFrame with datetime index
        train_months: Months for training
        val_months: Months for validation

    Returns:
        train_df, val_df, test_df
    """
    df = df.sort_index()

    total_rows = len(df)
    train_end = int(total_rows * (train_months / 12))
    val_end = int(total_rows * ((train_months + val_months) / 12))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def train_ensemble(coin: str, include_fgi: bool = True, include_onchain: bool = False) -> dict:
    """
    Train ensemble model for a coin.

    Args:
        coin: Coin to train
        include_fgi: Include Fear & Greed features
        include_onchain: Include on-chain metrics (BTC only)

    Returns:
        Dict with metrics and models
    """
    print(f"\n{'='*60}")
    print(f"TRAINING ENSEMBLE FOR {coin.upper()}")
    print(f"{'='*60}")

    # Load data
    df = load_data(coin)

    # For altcoins, load BTC for cross-asset features
    btc_df = None
    include_cross_asset = coin.lower() != "btc"
    if include_cross_asset:
        btc_df = load_data("btc")

    # Prepare features
    df = prepare_features(
        df,
        lookahead=1,
        btc_df=btc_df,
        include_fgi=include_fgi,
        include_onchain=include_onchain and coin.lower() == "btc"
    )

    print(f"Total samples: {len(df)}")

    # Get feature columns
    feature_cols = get_feature_columns(
        include_fgi=include_fgi,
        include_cross_asset=include_cross_asset,
        include_onchain=include_onchain and coin.lower() == "btc"
    )

    # Filter to available features
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"Features: {len(available_features)}")

    # Split data
    train_df, val_df, test_df = walk_forward_split(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    X_train = train_df[available_features]
    y_train = train_df["target"]
    X_val = val_df[available_features]
    y_val = val_df["target"]
    X_test = test_df[available_features]
    y_test = test_df["target"]

    # Load tuned params
    params = load_tuned_params(coin)

    # Train individual models
    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, params)

    print("Training LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, params)

    cat_model = None
    if HAS_CATBOOST:
        print("Training CatBoost...")
        cat_model = train_catboost(X_train, y_train, X_val, y_val, params)

    # Create ensemble
    models = {"xgb": xgb_model, "lgb": lgb_model}
    if cat_model:
        models["cat"] = cat_model

    ensemble = EnsembleClassifier(models)

    # Evaluate on test set
    print("\n" + "-" * 40)
    print("TEST SET RESULTS")
    print("-" * 40)

    results = {}

    # Individual model results
    for name, model in models.items():
        if model:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            print(f"{name.upper():4}: Accuracy={acc*100:.1f}%, AUC={auc:.4f}")
            results[f"{name}_accuracy"] = acc
            results[f"{name}_auc"] = auc

    # Ensemble results
    y_pred_ens = ensemble.predict(X_test)
    y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
    acc_ens = accuracy_score(y_test, y_pred_ens)
    auc_ens = roc_auc_score(y_test, y_prob_ens)
    print(f"{'ENS':4}: Accuracy={acc_ens*100:.1f}%, AUC={auc_ens:.4f}")

    results["ensemble_accuracy"] = acc_ens
    results["ensemble_auc"] = auc_ens

    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(ensemble, MODELS_DIR / f"{coin}_ensemble.joblib")
    joblib.dump(xgb_model, MODELS_DIR / f"{coin}_xgb.joblib")  # Keep XGB for backward compat
    joblib.dump(lgb_model, MODELS_DIR / f"{coin}_lgb.joblib")
    if cat_model:
        joblib.dump(cat_model, MODELS_DIR / f"{coin}_cat.joblib")

    # Save metadata
    meta = {
        "coin": coin,
        "trained_at": datetime.now().isoformat(),
        "features": available_features,
        "include_fgi": include_fgi,
        "include_cross_asset": include_cross_asset,
        "include_onchain": include_onchain and coin.lower() == "btc",
        "include_kalman": True,
        "include_multitimeframe": True,
        "include_macro": False,
        "include_flows": False,
        "include_orderbook": False,
        "model_type": "ensemble",
        "models": list(models.keys()),
        "test_accuracy": acc_ens,
        "test_auc": auc_ens,
    }

    with open(MODELS_DIR / f"{coin}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved ensemble model to {MODELS_DIR / f'{coin}_ensemble.joblib'}")

    return results


def train_all_ensembles(include_fgi: bool = True, include_onchain: bool = True):
    """Train ensemble models for all coins."""
    all_results = {}

    for coin in SYMBOLS.keys():
        try:
            results = train_ensemble(
                coin,
                include_fgi=include_fgi,
                include_onchain=include_onchain
            )
            all_results[coin] = results
        except Exception as e:
            print(f"\nError training {coin}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("ENSEMBLE TRAINING SUMMARY")
    print("=" * 60)

    for coin, results in all_results.items():
        print(f"\n{coin.upper()}:")
        print(f"  XGBoost:  {results.get('xgb_accuracy', 0)*100:.1f}% (AUC: {results.get('xgb_auc', 0):.4f})")
        print(f"  LightGBM: {results.get('lgb_accuracy', 0)*100:.1f}% (AUC: {results.get('lgb_auc', 0):.4f})")
        if "cat_accuracy" in results:
            print(f"  CatBoost: {results.get('cat_accuracy', 0)*100:.1f}% (AUC: {results.get('cat_auc', 0):.4f})")
        print(f"  Ensemble: {results.get('ensemble_accuracy', 0)*100:.1f}% (AUC: {results.get('ensemble_auc', 0):.4f})")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ensemble models")
    parser.add_argument("--coin", type=str, default="all", help="Coin to train (btc, eth, doge, zec, or 'all')")
    parser.add_argument("--no-fgi", action="store_true", help="Disable Fear & Greed features")
    parser.add_argument("--onchain", action="store_true", help="Include on-chain metrics (BTC only)")

    args = parser.parse_args()

    if args.coin == "all":
        train_all_ensembles(
            include_fgi=not args.no_fgi,
            include_onchain=args.onchain
        )
    else:
        train_ensemble(
            args.coin,
            include_fgi=not args.no_fgi,
            include_onchain=args.onchain
        )
