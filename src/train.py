from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.preprocessing import preprocess_data
from src.genetic_optimizer import genetic_algorithm, run_ga_experiments
from src.monitoring import (
    get_logger,
    PerformanceTracker,
    resource_monitor,
    export_metrics,
)

logger = get_logger("train")

RANDOM_STATE = 42

GA_SGD_EXPERIMENTS = [
    {"population_size": 20, "generations": 20, "mutation_rate": 0.10},
    {"population_size": 30, "generations": 25, "mutation_rate": 0.20},
    {"population_size": 40, "generations": 30, "mutation_rate": 0.30},
]


def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
    }


def train_baseline_models(X_train, y_train):
    with PerformanceTracker("train.baseline_models", logger):
        log_reg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        log_reg.fit(X_train, y_train)

        tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)

        sgd = SGDClassifier(loss="log_loss", max_iter=2000, random_state=RANDOM_STATE)
        sgd.fit(X_train, y_train)

    return {
        "Logistic Regression (Original)": log_reg,
        "Decision Tree (Original)": tree,
        "SGD Classifier (Original)": sgd,
    }


def train_search_optimized_models(X_train, y_train):
    with PerformanceTracker("train.search_optimized_models", logger):
        log_reg_grid = GridSearchCV(
            LogisticRegression(max_iter=2000),
            {"C": [0.01, 0.1, 1, 10]},
            scoring="recall",
            cv=5,
            n_jobs=-1,
        )
        log_reg_grid.fit(X_train, y_train)

        tree_random = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
            },
            n_iter=10,
            scoring="recall",
            cv=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        tree_random.fit(X_train, y_train)

    return {
        "Logistic Regression (GridSearch)": log_reg_grid.best_estimator_,
        "Decision Tree (RandomSearch)": tree_random.best_estimator_,
    }


def train_ga_optimized_models(X_train, y_train, X_val, y_val):
    with PerformanceTracker("train.ga_models", logger):
        _, best_sgd = run_ga_experiments(
            model_type="sgd",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            experiment_configs=GA_SGD_EXPERIMENTS,
            selection_metric="recall",
            verbose=False,
        )

    return {
        "SGD Classifier (Genetic Algorithm - Best of 3)": best_sgd["model"]
    }


def train_models():
    logger.info("=== ML Pipeline started ===")

    with PerformanceTracker("train.full_pipeline", logger):
        X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data()

        models = {
            **train_baseline_models(X_train, y_train),
            **train_search_optimized_models(X_train, y_train),
            **train_ga_optimized_models(X_train, y_train, X_val, y_val),
        }

        results_val = {}

        for name, model in models.items():
            results_val[name] = evaluate_model(model, X_val, y_val)

        # 🔥 Seleção baseada em RECALL (VAL)
        best_model_name = max(results_val, key=lambda m: results_val[m]["recall"])
        best_model = models[best_model_name]

        # 🔥 Avaliação FINAL (TEST)
        test_metrics = evaluate_model(best_model, X_test, y_test)

        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Test metrics: {test_metrics}")

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "test_metrics": test_metrics,
        "val_metrics": results_val[best_model_name],
        "scaler": _
    }


if __name__ == "__main__":
    with resource_monitor(interval_seconds=10):
        result = train_models()

    path = export_metrics()
    print("Metrics exported to:", path)