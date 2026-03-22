from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import preprocess_data
from genetic_optimizer import genetic_algorithm, run_ga_experiments
# Importação do sistema de monitoramento para rastrear desempenho do pipeline completo
from monitoring import (
    get_logger,
    PerformanceTracker,
    resource_monitor,
    export_metrics,
)

# Logger para o módulo de treinamento
logger = get_logger("train")


RANDOM_STATE = 42
GA_SGD_EXPERIMENTS = [
    {"population_size": 20, "generations": 20, "mutation_rate": 0.10},
    {"population_size": 30, "generations": 25, "mutation_rate": 0.20},
    {"population_size": 40, "generations": 30, "mutation_rate": 0.30}
]


def evaluate_model(model, X, y):
    """
    Evaluate classification model using multiple metrics.
    """
    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    }
    # Registro das métricas calculadas (modo debug)
    logger.debug(f"Evaluation metrics: {metrics}")
    return metrics


def print_metrics(model_name: str, metrics: dict):
    print(f"\nModel: {model_name}")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")


def train_baseline_models(X_train, y_train):
    # Medição de performance do treinamento dos modelos baseline
    with PerformanceTracker("train.baseline_models", logger):
        log_reg_original = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE
        )
        log_reg_original.fit(X_train, y_train)
        # Confirmação de treinamento do modelo
        logger.info("Trained Logistic Regression (Original)")

        tree_original = DecisionTreeClassifier(random_state=RANDOM_STATE)
        tree_original.fit(X_train, y_train)
        logger.info("Trained Decision Tree (Original)")

        sgd_original = SGDClassifier(
            loss="log_loss",
            max_iter=2000,
            random_state=RANDOM_STATE
        )
        sgd_original.fit(X_train, y_train)
        logger.info("Trained SGD Classifier (Original)")

    return {
        "Logistic Regression (Original)": log_reg_original,
        "Decision Tree (Original)": tree_original,
        "SGD Classifier (Original)": sgd_original
    }


def train_search_optimized_models(X_train, y_train):
    # Rastreamento de performance dos modelos com otimização de hiperparâmetros
    with PerformanceTracker("train.search_optimized_models", logger):
        log_reg_grid = GridSearchCV(
            LogisticRegression(solver="lbfgs", max_iter=2000),
            {"C": [0.01, 0.1, 1, 10]},
            scoring="recall",
            cv=5,
            n_jobs=-1
        )
        log_reg_grid.fit(X_train, y_train)
        # Log dos melhores parâmetros encontrados pelo GridSearch
        logger.info(
            f"GridSearch best C={log_reg_grid.best_params_['C']}, "
            f"best recall={log_reg_grid.best_score_:.4f}"
        )

        tree_random = RandomizedSearchCV(
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth": [None, 5, 10, 20, 30],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10]
            },
            n_iter=20,
            scoring="recall",
            cv=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        tree_random.fit(X_train, y_train)
        # Registro do resultado do RandomSearch
        logger.info(
            f"RandomSearch best params={tree_random.best_params_}, "
            f"best recall={tree_random.best_score_:.4f}"
        )

    return {
        "Logistic Regression (GridSearch)": log_reg_grid.best_estimator_,
        "Decision Tree (RandomSearch)": tree_random.best_estimator_
    }


def train_ga_optimized_models(X_train, y_train, X_val, y_val):
    # Medição de performance dos modelos otimizados com algoritmo genético
    with PerformanceTracker("train.ga_optimized_models", logger):
        sgd_experiments, best_sgd_experiment = run_ga_experiments(
            model_type="sgd",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            experiment_configs=GA_SGD_EXPERIMENTS,
            selection_metric="recall",
            verbose=True
        )

        print("\nResumo dos 3 experimentos de AG (SGD):")
        for experiment in sgd_experiments:
            metrics = experiment["val_metrics"]
            cfg = experiment["config"]
            print(
                f"Exp {experiment['experiment_id']} | "
                f"pop={cfg['population_size']} gens={cfg['generations']} "
                f"mut={cfg['mutation_rate']:.2f} | "
                f"acc={metrics['accuracy']:.4f} "
                f"recall={metrics['recall']:.4f} "
                f"f1={metrics['f1_score']:.4f} "
                f"fitness={experiment['fitness']:.4f}"
            )

        log_reg_ga, _, _, _ = genetic_algorithm(
            model_type="logistic_regression",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            population_size=20,
            generations=20,
            mutation_rate=0.2,
            verbose=True
        )

        tree_ga, _, _, _ = genetic_algorithm(
            model_type="decision_tree",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            population_size=20,
            generations=20,
            mutation_rate=0.2,
            verbose=True
        )

    return {
        "Logistic Regression (Genetic Algorithm)": log_reg_ga,
        "Decision Tree (Genetic Algorithm)": tree_ga,
        "SGD Classifier (Genetic Algorithm - Best of 3)": best_sgd_experiment["model"]
    }


def train_models():
    # Marcação do início do pipeline completo
    logger.info("=== ML Pipeline started ===")

    # Rastreamento de performance do pipeline completo (todas as etapas)
    with PerformanceTracker("train.full_pipeline", logger):
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            _
        ) = preprocess_data()

        models = {
            **train_baseline_models(X_train, y_train),
            **train_search_optimized_models(X_train, y_train),
            **train_ga_optimized_models(X_train, y_train, X_val, y_val)
        }

        results = {}

        # Medição do tempo gasto na avaliação de todos os modelos
        with PerformanceTracker("train.validation_evaluation", logger):
            for name, model in models.items():
                metrics = evaluate_model(model, X_val, y_val)
                results[name] = metrics
                print_metrics(name, metrics)
                # Log das métricas de validação de cada modelo
                logger.info(f"Validation — {name}: {metrics}")

        # =========================
        # Final evaluation on test set
        # =========================
        
        best_model_name = max(
            results,
            key=lambda m: results[m]["recall"]
        )

        best_model = models[best_model_name]
        # Registro do modelo com melhor desempenho
        logger.info(f"Best model selected: {best_model_name}")

        test_metrics = evaluate_model(best_model, X_test, y_test)

        print("\nFinal evaluation on test set:")
        print_metrics(best_model_name, test_metrics)
        # Log das métricas finais no conjunto de teste
        logger.info(f"Test set — {best_model_name}: {test_metrics}")

    return results, models, (X_test, y_test)



if __name__ == "__main__":
    # Monitoramento de recursos do sistema (CPU, memória) a cada 10 segundos em background
    with resource_monitor(interval_seconds=10):
        results, models, _ = train_models()

    # Exportação de todas as métricas coletadas para arquivo JSON
    metrics_path = export_metrics()
    logger.info(f"Run metrics exported to {metrics_path}")
    print(f"\nTraining completed. Metrics exported to: {metrics_path}")
    
    
