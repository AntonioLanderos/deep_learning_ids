import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(model, test_features, test_labels):
    test_loss, test_accuracy = model.evaluate(test_features, test_labels)

    print("Test accuracy:", test_accuracy)
    print("Test loss:", test_loss)

    return test_loss, test_accuracy


def evaluate_predictions(model, test_features, test_labels, label_encoder):
    predictions = model.predict(test_features)
    predicted_labels = np.argmax(predictions, axis=1)

    print("Classification Report:")
    print(classification_report(
        test_labels,
        predicted_labels,
        target_names=label_encoder.classes_
    ))

    cm = confusion_matrix(test_labels, predicted_labels)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    disp.plot(cmap=plt.cm.Reds, xticks_rotation=90)
    plt.title("Confusion Matrix")
    plt.show()

    return predictions, predicted_labels, cm


def plot_training_curves(history):
    acc_train = history.history["acc"]
    acc_val = history.history["val_acc"]
    loss_train = history.history["loss"]
    loss_val = history.history["val_loss"]

    epochs = range(1, len(acc_train) + 1)

    plt.figure()
    plt.plot(epochs, acc_train, "o-", label="Train accuracy")
    plt.plot(epochs, acc_val, "s--", label="Validation accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, loss_train, "o-", label="Train loss")
    plt.plot(epochs, loss_val, "s--", label="Validation loss")
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
