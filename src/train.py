"""
Model training with walk-forward validation.
Uses Fear & Greed Index, cross-asset features, and optionally tuned hyperparameters.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from data_fetch import load_data, fetch_historical_data, save_data
from features import prepare_features, get_feature_columns

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_best_params(coin: str) -> dict:
    """Load tuned params if available."""
    params_path = MODELS_DIR / f"{coin}_best_params.json"
    if params_path.exists():
        with open(params_path) as f:
            data = json.load(f)
            return data.get("params", {})
    return None


def walk_forward_split(df: pd.DataFrame, train_months: int = 8, val_months: int = 2):
    """
    Split data chronologically for walk-forward validation.
    Updated for 12-month data: 8 train, 2 val, 2 test.
    """
    train_end = df.index.min() + pd.Timedelta(days=train_months * 30)
    val_end = train_end + pd.Timedelta(days=val_months * 30)

    train_df = df[df.index < train_end]
    val_df = df[(df.index >= train_end) & (df.index < val_end)]
    test_df = df[df.index >= val_end]

    print(f"Train: {len(train_df)} samples ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"Val:   {len(val_df)} samples ({val_df.index.min().date()} to {val_df.index.max().date()})")
    print(f"Test:  {len(test_df)} samples ({test_df.index.min().date()} to {test_df.index.max().date()})")

    return train_df, val_df, test_df


def train_model(
    coin: str,
    months: int = 12,
    force_fetch: bool = False,
    use_tuned_params: bool = True,
    include_fgi: bool = True
) -> dict:
    """
    Train XGBoost model for a coin with all Phase 2 features.

    Args:
        coin: Coin name (btc, eth, doge, zec)
        months: Months of historical data (default 12)
        force_fetch: Force fresh data fetch
        use_tuned_params: Use Optuna-tuned params if available
        include_fgi: Include Fear & Greed features

    Returns:
        Dict with model, metrics, and feature importance
    """
    print(f"\n{'='*60}")
    print(f"Training model for {coin.upper()}")
    print(f"{'='*60}")

    # Load or fetch data
    try:
        if force_fetch:
            raise FileNotFoundError("Forcing fresh fetch")
        df = load_data(coin)
        print(f"Loaded cached data: {len(df)} rows")
    except FileNotFoundError:
        print("Fetching fresh data...")
        df = fetch_historical_data(coin, months=months)
        save_data(df, coin)

    # Load BTC data for cross-asset features (if not BTC)
    btc_df = None
    if coin.lower() != "btc":
        try:
            btc_df = load_data("btc")
            print(f"Loaded BTC data for cross-asset features")
        except FileNotFoundError:
            print("Fetching BTC data for cross-asset features...")
            btc_df = fetch_historical_data("btc", months=months)
            save_data(btc_df, "btc")

    # Feature engineering with FGI and cross-asset
    print("\nEngineering features (FGI + cross-asset)...")
    df = prepare_features(df, lookahead=1, btc_df=btc_df, include_fgi=include_fgi)
    print(f"After feature engineering: {len(df)} samples")

    # Split data
    print("\nSplitting data (walk-forward)...")
    train_df, val_df, test_df = walk_forward_split(df, train_months=8, val_months=2)

    # Prepare X, y
    include_cross_asset = coin.lower() != "btc"
    features = get_feature_columns(include_fgi=include_fgi, include_cross_asset=include_cross_asset)

    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    print(f"Using {len(available_features)} features")

    X_train = train_df[available_features]
    y_train = train_df["target"]
    X_val = val_df[available_features]
    y_val = val_df["target"]
    X_test = test_df[available_features]
    y_test = test_df["target"]

    # Get model params
    if use_tuned_params:
        tuned_params = load_best_params(coin)
        if tuned_params:
            print(f"\nUsing tuned hyperparameters")
            model_params = {**tuned_params, "random_state": 42, "eval_metric": "logloss"}
        else:
            print(f"\nNo tuned params found, using defaults")
            model_params = {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "logloss",
            }
    else:
        model_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss",
        }

    # Train XGBoost
    print("\nTraining XGBoost...")
    model = XGBClassifier(**model_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate
    print("\nEvaluating...")

    def evaluate(X, y, name):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, y_prob)

        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  AUC:       {auc:.4f}")

        return {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc}

    val_metrics = evaluate(X_val, y_val, "Validation")
    test_metrics = evaluate(X_test, y_test, "Test")

    # Feature importance
    importance = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 10 Features:")
    print(importance.head(10).to_string(index=False))

    # Save model and metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{coin}_xgb.joblib"
    joblib.dump(model, model_path)

    # Save feature list for prediction
    meta_path = MODELS_DIR / f"{coin}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "features": available_features,
            "include_fgi": include_fgi,
            "include_cross_asset": include_cross_asset,
            "include_kalman": True,
            "include_multitimeframe": True,
            "include_macro": False,
            "include_flows": False,
            "include_orderbook": False,
            "train_date": datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nModel saved to {model_path}")

    # Baseline comparison
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"\n{'='*60}")
    print(f"Baseline (always predict majority): {baseline_acc:.4f}")
    print(f"Model accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Improvement: {test_metrics['accuracy'] - baseline_acc:+.4f}")
    print(f"{'='*60}")

    return {
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_importance": importance,
        "model_path": model_path
    }


def train_all_coins(months: int = 12, force_fetch: bool = False, use_tuned_params: bool = True) -> dict:
    """Train models for all coins."""
    results = {}
    for coin in ["btc", "eth", "doge", "zec"]:
        results[coin] = train_model(
            coin,
            months=months,
            force_fetch=force_fetch,
            use_tuned_params=use_tuned_params
        )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train crypto prediction models")
    parser.add_argument("--coin", type=str, default="btc", help="Coin to train (btc, eth, doge, zec, or 'all')")
    parser.add_argument("--months", type=int, default=12, help="Months of historical data")
    parser.add_argument("--force-fetch", action="store_true", help="Force fresh data fetch")
    parser.add_argument("--no-tuned", action="store_true", help="Don't use tuned params")

    args = parser.parse_args()

    if args.coin == "all":
        train_all_coins(
            months=args.months,
            force_fetch=args.force_fetch,
            use_tuned_params=not args.no_tuned
        )
    else:
        train_model(
            args.coin,
            months=args.months,
            force_fetch=args.force_fetch,
            use_tuned_params=not args.no_tuned
        )
