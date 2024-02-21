# Import necessary libraries
import tensorflow as tf
print(tf.__version__)
import tensorflow_federated as tff
print(tff.__version__)
import pandas as pd
print(pd.__version__)


# Function to read leader info
def read_leader_info(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        cell_id = eval(lines[0].split(': ')[1].strip())
        leader_id = lines[2].split(': ')[1].strip()
        return cell_id, leader_id

# Load the data
vehicle_data = pd.read_csv('vehicle_data.csv')

# Read the leader info
cell_id, leader_id = read_leader_info('leader_info.txt')

# Preprocess the data
def preprocess_data(dataframe, cell_id):
    dataframe = dataframe[dataframe['cell_id'] == cell_id].copy()
    labels = dataframe.pop('group_leader')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.batch(32)
    return ds

# Create a Keras model
def create_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create the federated data
federated_data = [preprocess_data(vehicle_data, cell_id)]

# Create a TFF model from a Keras model
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=federated_data[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])

# Define a function for client update with noise addition
def client_update(model, dataset):
    # Perform model training on client data
    # ... calculate gradients

    # Add Gaussian noise to gradients
    noise_stddev = l2_norm_clip * noise_multiplier
    gradients = gradients + tf.random.normal(tf.shape(gradients), stddev=noise_stddev)

    return model.update_gradients(gradients)

# Create a standard optimizer (no need for DP-specific one)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# Create the federated averaging process
iterative_process = tff.learning.build_model(
    model_fn,
    client_update=client_update,  # Pass the modified client_update function
    server_update=tff.learning.build_federated_averaging_process(optimizer),
)

# Initialize the process
state = iterative_process.initialize()

# Train the model for a few rounds
for _ in range(10):  # You can change this to the number of rounds you want to train for
    state, metrics = iterative_process.next(state, federated_data)
    print('metrics:', metrics)