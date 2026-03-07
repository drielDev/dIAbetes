import random
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import recall_score

from preprocessing import preprocess_data


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def fitness(params, X_train, y_train, X_val, y_val):

    model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        alpha=params["alpha"],
        penalty=params["penalty"],
        max_iter=params["max_iter"],
        eta0=params["eta0"],
        random_state=SEED
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    score = recall_score(y_val, preds)

    return score

def create_individual():

    return {
        "alpha": random.uniform(0.00001, 0.01),
        "penalty": random.choice(["l1", "l2"]),
        "max_iter": random.randint(500, 2000),
        "eta0": random.uniform(0.0001, 0.1)
    }
    
def create_population(size):

    return [create_individual() for _ in range(size)]

def selection(population, scores, k=5):

    sorted_pop = sorted(
        zip(population, scores),
        key=lambda x: x[1],
        reverse=True
    )

    selected = [ind for ind, score in sorted_pop[:k]]

    return selected

def crossover(parent1, parent2):

    child = {}

    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])

    return child

def mutate(individual, rate=0.2):

    if random.random() < rate:
        individual["alpha"] *= random.uniform(0.5, 1.5)
        individual["alpha"] = min(max(individual["alpha"], 1e-6), 1e-1)

    if random.random() < rate:
        individual["max_iter"] += random.randint(-200, 200)
        individual["max_iter"] = min(max(individual["max_iter"], 200), 5000)

    if random.random() < rate:
        individual["eta0"] *= random.uniform(0.5, 1.5)
        individual["eta0"] = min(max(individual["eta0"], 1e-5), 1.0)

    if random.random() < rate:
        individual["penalty"] = random.choice(["l1", "l2"])

    return individual

def genetic_algorithm(
    X_train=None,
    y_train=None,
    X_val=None,
    y_val=None,
    population_size=20,
    generations=30
):

    if any(v is None for v in [X_train, y_train, X_val, y_val]):
        X_train, X_val, _, y_train, y_val, _, _ = preprocess_data()

    population = create_population(population_size)

    best_individual = None
    best_score = float("-inf")

    for gen in range(generations):

        scores = [
            fitness(ind, X_train, y_train, X_val, y_val)
            for ind in population
        ]

        for ind, score in zip(population, scores):
            if score > best_score:
                best_score = score
                best_individual = ind.copy()

        selected = selection(population, scores)

        new_population = selected.copy()

        while len(new_population) < population_size:

            p1, p2 = random.sample(selected, 2)

            child = crossover(p1, p2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

        print(f"Geração {gen} - Melhor recall: {best_score:.4f}")

    best_model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        random_state=SEED,
        **best_individual
    )

    best_model.fit(X_train, y_train)

    return best_model, best_individual, best_score


if __name__ == "__main__":
    best_model, best_params, best_recall = genetic_algorithm()
    print("\nMelhores hiperparâmetros encontrados:")
    print(best_params)
    print(f"Melhor recall de validação: {best_recall:.4f}")
