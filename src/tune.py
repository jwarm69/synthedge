"""
Hyperparameter tuning with Optuna.
Walk-forward validation to prevent lookahead bias.
"""

import json
import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from data_fetch import load_data, fetch_historical_data, save_data
from features import prepare_features, get_feature_columns

MODELS_DIR = Path(__file__).parent.parent / "models"


def walk_forward_split(df: pd.DataFrame, train_months: int = 8, val_months: int = 2):
    """Split data chronologically for walk-forward validation."""
    train_end = df.index.min() + pd.Timedelta(days=train_months * 30)
    val_end = train_end + pd.Timedelta(days=val_months * 30)

    train_df = df[df.index < train_end]
    val_df = df[(df.index >= train_end) & (df.index < val_end)]
    test_df = df[df.index >= val_end]

    return train_df, val_df, test_df


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function: maximize validation AUC."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "eval_metric": "logloss",
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)

    return auc


def tune_coin(
    coin: str,
    n_trials: int = 50,
    months: int = 12,
    force_fetch: bool = False
) -> dict:
    """
    Tune hyperparameters for a single coin.

    Args:
        coin: Coin name (btc, eth, doge, zec)
        n_trials: Number of Optuna trials
        months: Months of historical data
        force_fetch: Force fresh data fetch

    Returns:
        Dict with best params and best AUC
    """
    print(f"\n{'='*60}")
    print(f"Tuning {coin.upper()} with {n_trials} trials")
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

    # Load BTC for cross-asset features (if not BTC)
    btc_df = None
    if coin.lower() != "btc":
        try:
            btc_df = load_data("btc")
        except FileNotFoundError:
            btc_df = fetch_historical_data("btc", months=months)
            save_data(btc_df, "btc")

    # Feature engineering
    print("Engineering features...")
    df = prepare_features(df, lookahead=1, btc_df=btc_df, include_fgi=True)
    print(f"After feature engineering: {len(df)} samples")

    # Split data
    print("Splitting data (walk-forward)...")
    train_df, val_df, test_df = walk_forward_split(df, train_months=8, val_months=2)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Prepare X, y
    include_cross_asset = coin.lower() != "btc"
    features = get_feature_columns(include_fgi=True, include_cross_asset=include_cross_asset)

    # Filter to only available features
    available_features = [f for f in features if f in df.columns]
    print(f"Using {len(available_features)} features")

    X_train = train_df[available_features]
    y_train = train_df["target"]
    X_val = val_df[available_features]
    y_val = val_df["target"]

    # Run Optuna
    print(f"\nRunning {n_trials} Optuna trials...")
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Results
    best_params = study.best_params
    best_auc = study.best_value

    print(f"\n{'='*60}")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")
    print(f"{'='*60}")

    # Save best params
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    params_path = MODELS_DIR / f"{coin}_best_params.json"
    with open(params_path, "w") as f:
        json.dump({"params": best_params, "auc": best_auc}, f, indent=2)
    print(f"Saved best params to {params_path}")

    return {"best_params": best_params, "best_auc": best_auc}


def tune_all_coins(n_trials: int = 50, months: int = 12) -> dict:
    """Tune hyperparameters for all coins."""
    results = {}
    for coin in ["btc", "eth", "doge", "zec"]:
        results[coin] = tune_coin(coin, n_trials=n_trials, months=months)
    return results


def load_best_params(coin: str) -> dict:
    """Load best params for a coin."""
    params_path = MODELS_DIR / f"{coin}_best_params.json"
    if not params_path.exists():
        return None
    with open(params_path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune crypto prediction models")
    parser.add_argument("--coin", type=str, default="btc", help="Coin to tune (btc, eth, doge, zec, or 'all')")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--months", type=int, default=12, help="Months of historical data")
    parser.add_argument("--force-fetch", action="store_true", help="Force fresh data fetch")

    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.coin == "all":
        tune_all_coins(n_trials=args.trials, months=args.months)
    else:
        tune_coin(args.coin, n_trials=args.trials, months=args.months, force_fetch=args.force_fetch)
