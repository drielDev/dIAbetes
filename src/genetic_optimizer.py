import random
import numpy as np

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

from preprocessing import preprocess_data


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DEFAULT_METRIC_WEIGHTS = {"accuracy": 0.2, "recall": 0.5, "f1_score": 0.3}
VALID_SELECTION_METRICS = {"accuracy", "recall", "f1_score", "fitness"}


PARAM_SPACES = {
    "logistic_regression": {
        "C": {"type": "float", "min": 0.001, "max": 100.0, "scale": "log"},
        "solver": {"type": "categorical", "values": ["lbfgs", "liblinear"]},
        "max_iter": {"type": "int", "min": 300, "max": 3000},
    },
    "decision_tree": {
        "max_depth": {"type": "int_or_none", "min": 3, "max": 40, "none_prob": 0.15},
        "min_samples_split": {"type": "int", "min": 2, "max": 30},
        "min_samples_leaf": {"type": "int", "min": 1, "max": 15},
        "criterion": {"type": "categorical", "values": ["gini", "entropy"]},
    },
    "sgd": {
        "alpha": {"type": "float", "min": 1e-6, "max": 1e-1, "scale": "log"},
        "penalty": {"type": "categorical", "values": ["l1", "l2"]},
        "max_iter": {"type": "int", "min": 300, "max": 5000},
        "eta0": {"type": "float", "min": 1e-5, "max": 1.0, "scale": "log"},
    },
}

MODEL_BUILDERS = {
    "logistic_regression": lambda p: LogisticRegression(
        C=p["C"],
        solver=p["solver"],
        max_iter=p["max_iter"],
        random_state=SEED,
    ),
    "decision_tree": lambda p: DecisionTreeClassifier(
        max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        min_samples_leaf=p["min_samples_leaf"],
        criterion=p["criterion"],
        random_state=SEED,
    ),
    "sgd": lambda p: SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        alpha=p["alpha"],
        penalty=p["penalty"],
        max_iter=p["max_iter"],
        eta0=p["eta0"],
        random_state=SEED,
    ),
}


def evaluate_metrics(model, X, y):
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "recall": recall_score(y, preds),
        "f1_score": f1_score(y, preds),
    }


def calculate_fitness(metrics, weights=None):
    weights = weights or DEFAULT_METRIC_WEIGHTS
    return (
        weights["accuracy"] * metrics["accuracy"]
        + weights["recall"] * metrics["recall"]
        + weights["f1_score"] * metrics["f1_score"]
    )


def sample_gene(spec):
    if spec["type"] == "categorical":
        return random.choice(spec["values"])

    if spec["type"] == "int":
        return random.randint(spec["min"], spec["max"])

    if spec["type"] == "int_or_none":
        if random.random() < spec.get("none_prob", 0.1):
            return None
        return random.randint(spec["min"], spec["max"])

    if spec["type"] == "float":
        if spec.get("scale") == "log":
            low = np.log10(spec["min"])
            high = np.log10(spec["max"])
            return float(10 ** random.uniform(low, high))
        return random.uniform(spec["min"], spec["max"])

    raise ValueError(f"Unsupported gene type: {spec['type']}")


def create_individual(param_space):
    return {param: sample_gene(spec) for param, spec in param_space.items()}


def create_population(size, param_space):
    return [create_individual(param_space) for _ in range(size)]


def build_model(model_type, params):
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return MODEL_BUILDERS[model_type](params)


def get_data_splits(X_train, y_train, X_val, y_val):
    if any(v is None for v in [X_train, y_train, X_val, y_val]):
        X_train, X_val, _, y_train, y_val, _, _ = preprocess_data()
    return X_train, y_train, X_val, y_val


def fitness(model_type, params, X_train, y_train, X_val, y_val, weights=None):
    model = build_model(model_type, params)
    model.fit(X_train, y_train)
    metrics = evaluate_metrics(model, X_val, y_val)
    return calculate_fitness(metrics, weights=weights), metrics


def selection(population, scores, k=5):
    ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
    return [individual for individual, _ in ranked[:k]]


def crossover(parent1, parent2):
    return {key: random.choice([parent1[key], parent2[key]]) for key in parent1}


def mutate(individual, param_space, rate=0.2):
    mutated = individual.copy()
    for param, spec in param_space.items():
        if random.random() < rate:
            mutated[param] = sample_gene(spec)
    return mutated


def validate_model_type(model_type):
    if model_type not in PARAM_SPACES:
        raise ValueError(f"model_type must be one of {list(PARAM_SPACES.keys())}")

def genetic_algorithm(
    model_type="sgd",
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
    population_size=20,
    generations=30,
    mutation_rate=0.2,
    elite_k=5,
    metric_weights=None,
    verbose=True
):
    X_train, y_train, X_val, y_val = get_data_splits(X_train, y_train, X_val, y_val)
    validate_model_type(model_type)

    param_space = PARAM_SPACES[model_type]
    population = create_population(population_size, param_space)

    best_individual = None
    best_score = float("-inf")
    best_metrics = None

    for gen in range(generations):
        evaluations = [
            fitness(
                model_type,
                ind,
                X_train,
                y_train,
                X_val,
                y_val,
                weights=metric_weights
            )
            for ind in population
        ]

        scores = [score for score, _ in evaluations]
        for ind, (score, metrics) in zip(population, evaluations):
            if score > best_score:
                best_score = score
                best_individual = ind.copy()
                best_metrics = metrics.copy()

        elite_size = max(2, min(elite_k, population_size))
        selected = selection(population, scores, k=elite_size)

        new_population = selected.copy()
        while len(new_population) < population_size:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            child = mutate(child, param_space, rate=mutation_rate)
            new_population.append(child)
        population = new_population

        if verbose:
            print(f"[{model_type}] Geração {gen} - Melhor fitness: {best_score:.4f}")

    best_model = build_model(model_type, best_individual)
    best_model.fit(X_train, y_train)
    return best_model, best_individual, best_score, best_metrics


def run_ga_experiments(
    model_type="sgd",
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
    experiment_configs=None,
    metric_weights=None,
    selection_metric="recall",
    verbose=True
):
    X_train, y_train, X_val, y_val = get_data_splits(X_train, y_train, X_val, y_val)
    validate_model_type(model_type)

    if experiment_configs is None:
        experiment_configs = [
            {"population_size": 20, "generations": 20, "mutation_rate": 0.10},
            {"population_size": 30, "generations": 25, "mutation_rate": 0.20},
            {"population_size": 40, "generations": 30, "mutation_rate": 0.30},
        ]

    experiment_results = []

    for idx, cfg in enumerate(experiment_configs, start=1):
        if verbose:
            print(
                f"\nExperimento {idx} ({model_type}): "
                f"pop={cfg['population_size']}, gens={cfg['generations']}, "
                f"mutation={cfg['mutation_rate']}"
            )

        model, params, fitness_score, metrics = genetic_algorithm(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            population_size=cfg["population_size"],
            generations=cfg["generations"],
            mutation_rate=cfg["mutation_rate"],
            elite_k=min(5, cfg["population_size"]),
            metric_weights=metric_weights,
            verbose=verbose
        )

        experiment_results.append(
            {
                "experiment_id": idx,
                "config": cfg,
                "model": model,
                "best_params": params,
                "fitness": fitness_score,
                "val_metrics": metrics,
            }
        )

    if selection_metric not in VALID_SELECTION_METRICS:
        raise ValueError(
            "selection_metric must be one of: accuracy, recall, f1_score, fitness"
        )

    if selection_metric == "fitness":
        best_experiment = max(experiment_results, key=lambda x: x["fitness"])
    else:
        best_experiment = max(
            experiment_results,
            key=lambda x: x["val_metrics"][selection_metric]
        )

    return experiment_results, best_experiment


if __name__ == "__main__":
    _, best_experiment = run_ga_experiments(model_type="sgd", verbose=True)
    best_params = best_experiment["best_params"]
    best_fitness = best_experiment["fitness"]
    best_metrics = best_experiment["val_metrics"]
    print("\nMelhores hiperparâmetros encontrados:")
    print(best_params)
    print(
        "Melhores métricas de validação "
        f"(acc={best_metrics['accuracy']:.4f}, "
        f"recall={best_metrics['recall']:.4f}, "
        f"f1={best_metrics['f1_score']:.4f})"
    )
    print(f"Melhor fitness de validação: {best_fitness:.4f}")
