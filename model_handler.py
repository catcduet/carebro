import numpy as np

from keras.models import model_from_json
from sklearn.metrics import classification_report, confusion_matrix

from dataset import Dataset
from constants import *


def evaluate_model(model, X_test, Y_test):
    print("Evaluating...")
    loss, accuracy = model.evaluate(X_test, Y_test)
    print('\nloss: {} - accuracy: {}'.format(loss, accuracy))


def save_model(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + ".h5")
    print("Saved model to disk")


def load_model(model_name):
    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + ".h5")
    loaded_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    print("Loaded model from disk")
    return loaded_model


def g_confusion_matrix(model, X_test, Y_test):
    y_pred = model.predict_classes(X_test)
    print(y_pred)

    p = model.predict_proba(X_test)  # to predict probability

    target_names = [str(i) for i in range(273)]
    print('\n' + classification_report(np.argmax(Y_test, axis=1),
                                       y_pred, target_names=target_names))
    print(confusion_matrix(np.argmax(Y_test, axis=1), y_pred))


if __name__ == "__main__":
    # change 3 string parameters to test different sets and models
    d = Dataset("pkl_dataset/12345_400k_30_5_13/", "train_val_set_400000")
    X_test, Y_test = d.get_val_dataset()

    m = load_model('trained_models/12345_400k_30_5_13')
    g_confusion_matrix(m, X_test, Y_test)
