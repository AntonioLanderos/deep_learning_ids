import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

def calculate_class_weights(train_labels):
  classes = np.unique(train_labels)

  class_weights = compute_class_weight(
      class_weight="balanced",
      classes=classes,
      y=train_labels
  )

  class_weights = np.sqrt(class_weights)

  class_weights = np.minimum(class_weights, 5)

  class_weights_dict = dict(zip(classes, class_weights))

  return class_weights_dict

def save_model(experiment_name):
  checkpoint = ModelCheckpoint(filepath = f'./../models/{experiment_name}/model.keras',
                             save_freq = "epoch",
                             verbose = 1)
  return checkpoint

def train_model(model, train_features, train_labels, validation_features, validation_labels, class_weights, epochs, batch_size, checkpoint):
  history = model.fit(
      train_features,
      train_labels,
      epochs=epochs,
      batch_size=batch_size,
      validation_data=(validation_features, validation_labels),
      class_weight=class_weights,
      callbacks=[checkpoint]
  )

  return history
