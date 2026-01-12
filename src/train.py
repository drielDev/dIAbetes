from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import preprocess_data


def evaluate_model(model, X, y):
    """
    Evaluate classification model using multiple metrics.
    """
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    }


def print_metrics(model_name: str, metrics: dict):
    print(f"\nModel: {model_name}")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")


def train_models():
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _
    ) = preprocess_data()

    # =========================
    # Logistic Regression (Grid Search)
    # =========================
    log_reg = LogisticRegression(max_iter=2000)

    log_reg_params = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }

    log_reg_grid = GridSearchCV(
        log_reg,
        log_reg_params,
        scoring="recall",
        cv=5,
        n_jobs=-1
    )

    log_reg_grid.fit(X_train, y_train)

    best_log_reg = log_reg_grid.best_estimator_

    # =========================
    # Decision Tree (Random Search)
    # =========================
    tree = DecisionTreeClassifier(random_state=42)

    tree_params = {
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10]
    }

    tree_random = RandomizedSearchCV(
        tree,
        tree_params,
        n_iter=20,
        scoring="recall",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    tree_random.fit(X_train, y_train)

    best_tree = tree_random.best_estimator_

    # =========================
    # SGD Classifier
    # =========================
    sgd = SGDClassifier(
        loss="log_loss",
        max_iter=2000,
        random_state=42
    )

    sgd.fit(X_train, y_train)

    models = {
        "Logistic Regression (GridSearch)": best_log_reg,
        "Decision Tree (RandomSearch)": best_tree,
        "SGD Classifier": sgd
    }

    results = {}

    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val)
        results[name] = metrics
        print_metrics(name, metrics)

    return results, models, (X_test, y_test)


if __name__ == "__main__":
    results, models, _ = train_models()
    print("\nTraining completed with hyperparameter optimization.")
