from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

from train import train_models


def evaluate_on_test():
    """
    Evaluate the best trained model on the test set.
    """
    results, models, test_data = train_models()
    X_test, y_test = test_data

    # Escolha do melhor modelo com base no recall
    best_model_name = max(results, key=lambda x: results[x]["recall"])
    best_model = models[best_model_name]

    print(f"\nBest model selected: {best_model_name}")

    y_pred = best_model.predict(X_test)

    print("\nFinal Test Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


if __name__ == "__main__":
    evaluate_on_test()
