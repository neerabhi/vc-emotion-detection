# src/model/model_building.py

import os
import pickle
import logging
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# ── Logging configuration ────────────────────────────────────────────────────
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
# ─────────────────────────────────────────────────────────────────────────────


def get_root_directory() -> str:
    """Project root = two levels up."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '../../'))


def load_params(params_path: str) -> dict:
    """Load YAML and return the `model_building` dict."""
    try:
        full = yaml.safe_load(open(params_path, 'r'))
        params = full.get('model_building', {})
        logger.debug("Loaded params from %s: %s", params_path, params)
        return params
    except Exception as e:
        logger.error("Error loading params: %s", e)
        raise


def load_data(csv_path: str) -> pd.DataFrame:
    """Read CSV and fill any NaNs."""
    try:
        df = pd.read_csv(csv_path).fillna(0)  # assume numeric TF-IDF features
        logger.debug("Loaded data from %s (shape=%s)", csv_path, df.shape)
        return df
    except Exception as e:
        logger.error("Error loading CSV %s: %s", csv_path, e)
        raise


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict
) -> GradientBoostingClassifier:
    """Train and return a GradientBoostingClassifier."""
    try:
        clf = GradientBoostingClassifier(
            # learning_rate=params['learning_rate'],
            # n_estimators=params['n_estimators'],
            # max_depth=params.get('max_depth', 3),
            # random_state=params.get('random_state', 42)
            learning_rate=0.1,
            n_estimators=100
        )
        clf.fit(X_train, y_train)
        logger.debug("GradientBoostingClassifier trained")
        return clf
    except Exception as e:
        logger.error("Error training model: %s", e)
        raise


def save_model(model, output_path: str) -> None:
    """Pickle the model to disk."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug("Model saved to %s", output_path)
    except Exception as e:
        logger.error("Error saving model: %s", e)
        raise


def main():
    try:
        root = get_root_directory()

        # ─ Load params ─────────────────────────────────────────────────────────
        params_path = os.path.join(root, 'params.yaml')
        params = load_params(params_path)
        # Example:
        # test_tfidf.to_csv('data/processed/test_tfidf.csv', index=False)

        # ─ Load processed bow CSV ───────────────────────────────────────────
        data_path = os.path.join(root, 'data', 'processed', 'test_bow.csv')
        df = load_data(data_path)

        # ─ Split into X / y ─────────────────────────────────────────────────────
        #    all columns except last → features, last column → target
        X = df.iloc[:, :-1].values
        y = df.iloc[:,  -1].values

        # ─ Train ────────────────────────────────────────────────────────────────
        clf = train_model(X, y, params)

        # ─ Save ────────────────────────────────────────────────────────────────
        out_path = os.path.join(root, 'models', 'model.pkl')
        save_model(clf, out_path)

        logger.info("✓ Model training pipeline completed; saved to %s", out_path)

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
