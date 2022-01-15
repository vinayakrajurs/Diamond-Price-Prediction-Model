import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from tensorflow import feature_column
from tensorflow.keras import layers
import tempfile
import os
from matplotlib import pyplot as plt
from tensorflow.python.ops.gen_array_ops import shape

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

tf.keras.backend.set_floatx('float32')

column_names = ['carat', 'x', 'y', 'z', 'price']
diamonds_dataset_raw = pd.read_csv(
    'diamonds.csv', usecols=column_names, skipinitialspace=True)
dataset = diamonds_dataset_raw.copy()
dataset = dataset.reindex(np.random.permutation(dataset.index))
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
# Scale the labels
scale_factor = 1000.0
# Scale the training set's label.
train_dataset["price"] /= scale_factor

# Scale the test set's label
test_dataset["price"] /= scale_factor


def create_model(my_learning_rate, feature_layer):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(feature_layer)

    model.add(tf.keras.Input(shape=(16,)))

    # Add one linear layer to the model to yield a simple linear regressor.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(4,)))

    # Construct the layers into a model that TensorFlow can execute.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_absolute_error",
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    """Feed a dataset into the model in order to train it."""

    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["mean_absolute_error"]

    return epochs, rmse


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.94, rmse.max() * 1.05])
    plt.show()


resolution_in_degrees = 0.7

# Create a new empty list that will eventually hold the generated feature column.
feature_columns = []

# Create a bucket feature column for x.
x_as_a_numeric_column = tf.feature_column.numeric_column("x")
x_boundaries = list(np.arange(int(min(train_dataset['x'])), int(
    max(train_dataset['x'])), resolution_in_degrees))
x = tf.feature_column.bucketized_column(x_as_a_numeric_column, x_boundaries)

# Create a bucket feature column for y.
y_as_a_numeric_column = tf.feature_column.numeric_column("y")
y_boundaries = list(np.arange(int(min(train_dataset['y'])), int(
    max(train_dataset['y'])), resolution_in_degrees))
y = tf.feature_column.bucketized_column(y_as_a_numeric_column, y_boundaries)

# Create a bucket feature column for z.
z_as_a_numeric_column = tf.feature_column.numeric_column("z")
z_boundaries = list(np.arange(int(min(train_dataset['z'])), int(
    max(train_dataset['z'])), resolution_in_degrees))
z = tf.feature_column.bucketized_column(z_as_a_numeric_column, z_boundaries)

# Create a feature cross of x,y and z.
combined_x_y_z = tf.feature_column.crossed_column(
    [x, y, z], hash_bucket_size=100)
crossed_feature_1 = tf.feature_column.indicator_column(combined_x_y_z)
feature_columns.append(crossed_feature_1)

# Create a bucket feature column for carat.
carat_as_a_numeric_column = tf.feature_column.numeric_column("carat")
carat_boundaries = list(np.arange(int(min(train_dataset['carat'])), int(
    max(train_dataset['carat'])), resolution_in_degrees))
carat = tf.feature_column.bucketized_column(
    carat_as_a_numeric_column, carat_boundaries)

# Create a feature cross of x and carat.
combined_x_carat = tf.feature_column.crossed_column(
    [x, carat], hash_bucket_size=100)
crossed_feature_2 = tf.feature_column.indicator_column(combined_x_carat)
feature_columns.append(crossed_feature_2)

# Convert the list of feature columns into a layer that will later be fed into
# the model.
feature_cross_feature_layer = layers.DenseFeatures(feature_columns)

# The following variables are the hyperparameters.
learning_rate = 0.04
batch_size = 100
epochs = 10
label_name = 'price'

# Build the model, this time passing in the feature_cross_feature_layer:
my_model = create_model(learning_rate, feature_cross_feature_layer)

# Train the model on the training set.
epochs, rmse = train_model(my_model, train_dataset,
                           epochs, batch_size, label_name)

# plot_the_loss_curve(epochs, rmse)

# print("\n: Evaluate the new model against the test set:")
# test_features = {name: np.array(value) for name, value in test_dataset.items()}
# test_label = np.array(test_features.pop(label_name))
# my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)

# saving the model as a sav file
# file_name = 'diamonds_final.sav'
# joblib.dump(my_model, file_name)

my_model.save('diamonds_final')
# pickle.dump(my_model, open('finalised_model.pkl', 'wb'))
# model = pickle.load('finalised_model.pkl', 'rb')
