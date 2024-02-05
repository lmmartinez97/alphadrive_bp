import datetime
import json
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import seaborn as sns
import tensorflow as tf

from rich import print
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Convolution2D, AvgPool2D, MaxPooling2D, Convolution2DTranspose, Rescaling
from tensorflow.keras.layers import RandomFlip, RandomRotation
from tensorflow.keras.layers import Dropout, BatchNormalization, Flatten, Input, SpatialDropout2D

class AutoEncoder:
    def __init__(self, train_config = None, model_config = None, encoder_config = None, decoder_config = None):
        """
        Initialize an AutoEncoder instance.

        Args:
            train_config (dict): Configuration for training the model.
            model_config (dict): Configuration for the autoencoder model.
            encoder_config (dict): Configuration for the encoder part of the autoencoder.
            decoder_config (dict): Configuration for the decoder part of the autoencoder.
        """
        self.train_config = train_config
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.model_config = model_config

        self.encoder = None
        self.decoder = None
        self.model = None
        self.layer_mapping = {
            'Input': None,
            'RandomFlip': RandomFlip,
            'RandomRotation': RandomRotation,
            'LSTM': LSTM,
            'GRU': GRU,
            'Conv2D': Convolution2D,
            'Deconv2D': Convolution2DTranspose,
            'MaxPooling2D': MaxPooling2D,
            'AvgPooling2D': AvgPool2D,
            'Dense': Dense,
            'Flatten': Flatten,
            'Dropout': Dropout,
            'SpatialDropout2D': SpatialDropout2D,
            'BatchNormalization': BatchNormalization,
            'Rescaling': Rescaling
        }
    def buildModel(self, arch_config = None):
        """
        Build a Keras model based on the provided architecture configuration.

        Args:
            arch_config (dict): Configuration for the architecture of the model.

        Returns:
            tf.keras.models.Sequential: The constructed Keras model.
        """
        model = Sequential()
        for layer in arch_config.keys():
            layer_identifier = re.split('_', layer)[0]
            arch_config[layer]['name'] = layer
            
            layer_class = self.layer_mapping.get(layer_identifier)
            if layer_class:
                x = layer_class(**arch_config[layer])
                model.add(x)
        return model
    
    def buildAutoencoder(self):
        """
        Build the autoencoder model using the encoder and decoder configurations.
        """
        self.encoder = self.buildModel(arch_config=self.encoder_config)
        self.decoder = self.buildModel(arch_config=self.decoder_config)

        input_tensor = Input(shape=self.encoder_config['Input_1']['shape'])
        latent_vector = self.encoder(input_tensor)
        output = self.decoder(latent_vector)
        self.model = Model(input_tensor, output)
        self.compileModel()
        
    def compileModel(self):
        """
        Compile the autoencoder model with the specified loss function and optimizer.
        """
        loss_fun = self.model_config['loss']
        opt = self.model_config['opt']
        name = self.model_config['name']
        metrics = self.model_config['metrics']
        initial_learning_rate = self.model_config['initial_learning_rate']
        decay_rate = self.model_config['decay_rate']

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=decay_rate,
            staircase=True
        )

        if opt == 'Adam':
            opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        elif opt == 'SGD':
            opt = tf.keras.optimizers.legacy.SGD(learning_rate=lr_schedule)
        elif opt == 'Aadamax':
            opt = tf.keras.optimizers.legacy.Adamax(learning_rate=lr_schedule)

        print("Loss, optimizer and metric set up")
        self.model._name = name
        self.model.compile(loss = loss_fun, optimizer = opt, metrics = metrics)
        print("Model created:")
        self.model.summary()
        print("Model has been built")

    def trainModel(self, train_data, val_data, log_dir):
        """
        Train the autoencoder model using the provided training data.

        Args:
            train_data (numpy.ndarray): Training data.
            val_data (numpy.ndarray): Validation data.
            log_dir (str): Directory for TensorBoard logs.

        Returns:
            dict: Training history.
        """
        batch_size = self.train_config['batch_size']
        epochs = self.train_config['epochs']
        shuffle = self.train_config['shuffle']
        patience = self.train_config['ES_patience']
        min_delta = self.train_config['ES_min_delta']
        verbose = self.train_config['verbose']
        start_from = self.train_config['start_from_epoch']
        
        checkpoint_path = os.path.join(log_dir, "model_checkpoint.h5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        verbose=1,
        min_delta = min_delta,
        start_from_epoch = start_from,
        restore_best_weights = True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=4)

        history = self.model.fit(
            x=train_data,
            y=train_data,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(val_data, val_data),
            shuffle=shuffle,
            callbacks=[checkpoint, early_stopping, tensorboard_callback],
            steps_per_epoch=None,
            validation_freq=1,
        )
        self.history = history.history

        return self.history

    def predict(self, test_set):
        """
        Generate predictions using the trained autoencoder model.

        Args:
            test_set (numpy.ndarray): Input data for making predictions.

        Returns:
            numpy.ndarray: Predicted output.
        """
        predictions = self.model.predict(test_set)
        return predictions
    
    def compare(self, test_set, num_plots = 3):
        """
        Compare original input data with reconstructed data and display the results.

        Args:
            test_set (numpy.ndarray): Test data for comparison.
            num_plots (int): Number of plots to generate for comparison.
        """
        idxs = np.random.choice(range(len(test_set)), size=num_plots)
        print(idxs)
        test_samples = np.take(test_set, idxs, axis = 0)
        predictions = self.predict(test_samples)
        
        for (img, prediction, index) in zip(test_samples, predictions, idxs):
            fig, (ax1, ax2, ax3) = plt.subplots(3,1)
            fig.set_figheight(6)
            fig.set_figwidth(8)
            # ax1 = fig.add_subplot(1, 3, 1)
            pl1 = ax1.imshow(img, cmap = 'viridis', aspect = 'auto')
            ax1.set_title("Original field")
            ax1.set_xlabel("Longitudinal coordinate")
            ax1.set_ylabel("Transversal coordinate")
            ax1.axis('Off')
            fig.colorbar(pl1, orientation = 'vertical', pad = 0.1)

            # ax2 = fig.add_subplot(1, 3, 2)
            pl2 = ax2.imshow(prediction, cmap = 'viridis', aspect = 'auto')
            ax2.set_title("Reconstructed field")
            ax2.set_xlabel("Longitudinal coordinate")
            ax2.set_ylabel("Transversal coordinate")
            ax2.axis('Off')
            fig.colorbar(pl2, orientation = 'vertical', pad = 0.1)

            # ax3 = fig.add_subplot(1, 3, 3)
            pl3 = ax3.imshow(np.subtract(img, prediction), cmap = 'seismic', aspect = 'auto')
            ax3.set_title("Reconstruction error")
            ax3.set_xlabel("Longitudinal coordinate")
            ax3.set_ylabel("Transversal coordinate")
            ax3.axis('Off')
            fig.colorbar(pl3, orientation = 'vertical', pad = 0.1)

            fig.suptitle("Images and predictions from test set - Index {}".format(index))
            fig.tight_layout()
            plt.show()

    def save_model(self, directory, file_name):
        """
        Save the trained model and configuration dictionaries to files in the specified directory.

        Args:
            directory (str): Directory path for saving files.
            file_name (str): Base name for saved files (without extension). If None, a default name will be generated.
        """
        def is_jsonable(x):
            try:
                json.dumps(x)
                return True
            except (TypeError, OverflowError):
                return False
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Creating...")
            os.makedirs(directory)

        if file_name is None:
            # Generate a default name based on the current date and time
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            file_name = f"autoencoder_{timestamp}"
            
        # Save the model to a keras file
        model_file_path = os.path.join(directory, f"{file_name}.h5")
        self.model.save(model_file_path)
        self.encoder.save(os.path.join(directory, f"{file_name}_encoder.h5"))
        self.decoder.save(os.path.join(directory, f"{file_name}_decoder.h5"))

        # Combine all dictionaries into a single dictionary
        all_data = {
            "model_config": self.model_config,
            "encoder_config": self.encoder_config,
            "decoder_config": self.decoder_config,
            "train_config": self.train_config,
            "history": self.history if self.history else None
        }

        #Convert everything to string so that its json serializable
        for outer_key, outer_value in all_data.items():
            for inner_key, inner_value in outer_value.items():
                if not (is_jsonable(inner_value)):
                    all_data[outer_key][inner_key] = str(inner_value)

        # Save the combined dictionary as a JSON file
        configs_file = os.path.join(directory, f"{file_name}_configs.json")
        with open(configs_file, 'w') as f:
            json.dump(all_data, f, indent=4)

        print(f"Model and configs saved to {directory}")
      
        
    def plot_history(self, smoothing = 0.7):
        """
        Plot the training and validation loss, RMSE, and MAE over epochs.
        """
        def smooth(scalars, weight):  # Weight between 0 and 1
            last = scalars[0]  # First value in the plot (first timestep)
            smoothed = list()
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
                smoothed.append(smoothed_val)                        # Save it
                last = smoothed_val                                  # Anchor the last smoothed value
            return smoothed
        epochs = list(range(len(self.history['loss'])))
        loss = self.history['loss']
        val_loss = self.history['val_loss']
        mae = self.history.get('mean_absolute_error', None)
        val_mae = self.history.get('val_mean_absolute_error', None)
        rmse = self.history.get('root_mean_squared_error', None)
        val_rmse = self.history.get('val_root_mean_squared_error', None)
        binary_crossentropy = self.history.get('binary_crossentropy', None)
        val_binary_crossentropy = self.history.get('val_binary_crossentropy', None)

        plt.figure(figsize=(15,8))
        plt.subplot(2, 2, 1)
        plt.title('Loss - MSE')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(epochs ,loss, label='Training loss', linestyle='--', alpha = 0.5, color = 'purple')
        plt.plot(epochs, val_loss, label='Validation loss', linestyle='--', alpha = 0.5, color = 'orange')
        plt.plot(epochs, smooth(scalars=loss, weight=smoothing), label = 'Smoothed training loss', color = 'purple')
        plt.plot(epochs, smooth(scalars=val_loss, weight=smoothing), label = 'Smoothed validation loss', color = 'orange')
        plt.legend()
        plt.grid(linestyle='--', linewidth=1, alpha=0.5)

        if mae is not None:
            plt.subplot(2, 2, 2)
            plt.title('Mean Absolute Error (MAE)')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.plot(epochs, mae, label='Training loss', linestyle='--', alpha = 0.5, color = 'purple')
            plt.plot(epochs, val_mae, label='Validation loss', linestyle='--', alpha = 0.5, color = 'orange')
            plt.plot(epochs, smooth(scalars=mae, weight=smoothing), label = 'Smoothed training loss', color = 'purple')
            plt.plot(epochs, smooth(scalars=val_mae, weight=smoothing), label = 'Smoothed validation loss', color = 'orange')
            plt.legend()
            plt.grid(linestyle='--', linewidth=1, alpha=0.5)
        if rmse is not None:
            plt.subplot(2, 2, 3)
            plt.title('Root Mean Squared Error (RMSE)')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.plot(epochs, rmse, label='Training loss', linestyle='--', alpha = 0.5, color = 'purple')
            plt.plot(epochs, val_rmse, label='Validation loss', linestyle='--', alpha = 0.5, color = 'orange')
            plt.plot(epochs, smooth(scalars=rmse, weight=smoothing), label = 'Smoothed training loss', color = 'purple')
            plt.plot(epochs, smooth(scalars=val_rmse, weight=smoothing), label = 'Smoothed validation loss', color = 'orange')
            plt.legend()
            plt.grid(linestyle='--', linewidth=1, alpha=0.5)
        if binary_crossentropy is not None:
            plt.subplot(2, 2, 4)
            plt.title('Binary Crossentropy (BC)')
            plt.xlabel('Epoch')
            plt.ylabel('BC')
            plt.plot(epochs, binary_crossentropy, label='Training loss', linestyle='--', alpha = 0.5, color = 'purple')
            plt.plot(epochs, val_binary_crossentropy, label='Validation loss', linestyle='--', alpha = 0.5, color = 'orange')
            plt.plot(epochs, smooth(scalars=binary_crossentropy, weight=smoothing), label = 'Smoothed training loss', color = 'purple')
            plt.plot(epochs, smooth(scalars=val_binary_crossentropy, weight=smoothing), label = 'Smoothed validation loss', color = 'orange')
            plt.legend()
            plt.grid(linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        plt.show()

def load_model(model_name, directory='./saved_models'):
    """
    Load a trained model, configuration dictionaries, and training history from a directory.

    Args:
        model_name (str): Base name of the saved files (without extension).
        directory (str): Directory path where the model and configuration files are located.

    Returns:
        AutoEncoder: An AutoEncoder instance with the loaded model and configurations.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Unable to load model, configurations, and training history.")
        return None

    # Load the model from the h5 file
    model_file_path = os.path.join(directory, f"{model_name}.h5")
    if not os.path.exists(model_file_path):
        print(f"Model file {model_file_path} does not exist. Unable to load the model.")
        return None
    loaded_model = keras.models.load_model(model_file_path, compile=False)
    loaded_encoder = keras.models.load_model(os.path.join(directory, f"{model_name}_encoder.h5"), compile=False)
    loaded_decoder = keras.models.load_model(os.path.join(directory, f"{model_name}_decoder.h5"), compile=False)

    # Load the configuration dictionaries and training history from the JSON file
    configs_file = os.path.join(directory, f"{model_name}_configs.json")
    if not os.path.exists(configs_file):
        print(f"Configurations file {configs_file} does not exist. Unable to load the configurations.")
        return None

    with open(configs_file, 'r') as f:
        all_data = json.load(f)

    # Create an AutoEncoder instance with the loaded model and configurations
    loaded_autoencoder = AutoEncoder()
    loaded_autoencoder.model = loaded_model
    loaded_autoencoder.encoder = loaded_encoder
    loaded_autoencoder.decoder = loaded_decoder
    loaded_autoencoder.model_config = all_data["model_config"]
    loaded_autoencoder.encoder_config = all_data["encoder_config"]
    loaded_autoencoder.decoder_config = all_data["decoder_config"]
    loaded_autoencoder.train_config = all_data["train_config"]
    loaded_autoencoder.history = all_data["history"]

    # #convert history from string to list of floats
    # for key, value in loaded_autoencoder.history.items():
    #     loaded_autoencoder.history[key] = eval(value)

    print(f"Model, configurations, and training history loaded from {directory}/{model_name}")
    return loaded_autoencoder
