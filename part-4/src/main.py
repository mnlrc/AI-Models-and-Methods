import sys
import numpy as np
from utils import get_dataset
from auto_encoder import AutoEncoder
import matplotlib.pyplot as plt
from time import time
import os


INPUT_DIMENSION = 784
ENCODED_DIMENSION = 32
LEARNING_RATE = 0.03
EPOCHS = 10
RENDER = True


def plot_losses_over_parameter(train_losses: list[float], test_losses: list[float], parameter_value: any, parameter_name: str) -> None:
    """
    Plots the training and testing losses over a specific parameter.
    Parameters:
        - train_losses (list[float]): the list of training losses recorded at each epoch.
        - test_losses (list[float]): the list of testing losses recorded at each epoch.
        - parameter_value (any): the value of the parameter being analyzed.
        - parameter_name (str): the name of the parameter being analyzed.
    """
    plt.clf()

    if parameter_name == "Epochs":
        epochs = parameter_value
    else:
        epochs = EPOCHS

    plt.plot(range(epochs), train_losses, label="Training Loss")
    plt.plot(range(epochs), test_losses, label="Testing Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title(f"Mean Squared Error over {parameter_name} = {parameter_value}")
    plt.grid(visible=True)
    if not os.path.exists("doc/graphs"):
        os.makedirs("doc/graphs")
    plt.savefig(f"doc/graphs/auto_encoder_losses_{parameter_name}={parameter_value}.jpg")
    if RENDER:
        plt.show()


def visualize_reconstructed_images(test_output: np.ndarray, labels: np.ndarray, parameter_value: any, parameter_name: str) -> None:
    """
    Visualizes the reconstructed images from the auto-encoder.
    Parameters:
        - test_output (np.ndarray): the reconstructed test output after training.
        - labels (np.ndarray): the labels corresponding to the test dataset.
    """
    if RENDER:
        
        plt.clf()

        for i in range(len(labels)):
            img = test_output[i].reshape(28, 28)
            plt.imshow(img, cmap="gray")
            plt.title(f"Reconstructed Image of Digit {labels[i]} with {parameter_name} = {parameter_value}")
            plt.axis("off")
            plt.show()

            inpt = str(input("Press Enter to see the next image, press 's' to save image or 'q' to quit: ")).lower()

            if inpt == "":
                continue
            elif inpt == "s":
                if not os.path.exists("doc/reconstructed_images"):
                    os.makedirs("doc/reconstructed_images")
                plt.imsave(f"doc/reconstructed_images/entry{i + 2}_reconstructed_digit_{labels[i]}_{parameter_name}={parameter_value}.jpg", img, cmap="gray")
            elif inpt == "q":
                break
            else:
                break


def analyze_encoded_dimension_size_impact(training_data: np.ndarray, test_data: np.ndarray, labels: np.ndarray,  encoded_dimension: int):
    # init auto-encoder
    auto_encoder = AutoEncoder(input_dim=INPUT_DIMENSION, encoded_dim=encoded_dimension, learning_rate=LEARNING_RATE)
    # train auto-encoder
    print("Starting training...")
    start = time()
    training_losses, test_output, test_losses = auto_encoder.train(x_train=training_data, 
                                                                   x_test=test_data, 
                                                                   epochs=EPOCHS)
    end = time()
    total = end - start
    print("="*50)
    print(f"Training and testing completed in {total:.2f} seconds for encoded dimension size.")

    # plot losses
    plot_losses_over_parameter(train_losses=training_losses, 
                               test_losses=test_losses, 
                               parameter_value=encoded_dimension,
                               parameter_name="Encoded Dimension Vector Size")
    
    # visualize reconstructed images
    visualize_reconstructed_images(test_output=test_output,
                                   labels=labels,
                                   parameter_value=encoded_dimension,
                                   parameter_name="Encoded Dimension Vector Size")


def analyze_learning_rate_impact(training_data: np.ndarray, test_data: np.ndarray, labels: np.ndarray, learning_rate: float):
    # init auto-encoder
    auto_encoder = AutoEncoder(input_dim=INPUT_DIMENSION, encoded_dim=ENCODED_DIMENSION, learning_rate=learning_rate)
    # train auto-encoder
    print("Starting training...")
    start = time()
    training_losses, test_output, test_losses = auto_encoder.train(x_train=training_data, 
                                                                   x_test=test_data, 
                                                                   epochs=EPOCHS)
    end = time()
    total = end - start
    print("="*50)
    print(f"Training and testing completed in {total:.2f} seconds for learning rate.")

    # plot losses
    plot_losses_over_parameter(train_losses=training_losses,
                               test_losses=test_losses, 
                               parameter_value=learning_rate,
                               parameter_name="Learning Rate")
    
    # visualize reconstructed images
    visualize_reconstructed_images(test_output=test_output,
                                   labels=labels,
                                   parameter_value=learning_rate,
                                   parameter_name="Learning Rate")
    

def analyze_epochs_number_impact(training_data: np.ndarray, test_data: np.ndarray, labels: np.ndarray, epochs: int):
    # init auto-encoder
    auto_encoder = AutoEncoder(input_dim=INPUT_DIMENSION, encoded_dim=ENCODED_DIMENSION, learning_rate=LEARNING_RATE)
    # train auto-encoder
    print("Starting training...")
    start = time()
    training_losses, test_output, test_losses = auto_encoder.train(x_train=training_data, 
                                                                   x_test=test_data, 
                                                                   epochs=epochs)
    end = time()
    total = end - start
    print("="*50)
    print(f"Training and testing completed in {total:.2f} seconds for epochs number.")

    # plot losses
    plot_losses_over_parameter(train_losses=training_losses,
                               test_losses=test_losses, 
                               parameter_value=epochs,
                               parameter_name="Epochs")
    
    # visualize reconstructed images
    visualize_reconstructed_images(test_output=test_output,
                                   labels=labels,
                                   parameter_value=epochs,
                                   parameter_name="Epochs")


def main(args: list[str]):

    if len(args) == 1:
        data_directory = args[0]
        if os.path.isdir(data_directory):
            training_data_file_path = os.path.join(data_directory, "mnist_train.csv")
            test_data_file_path = os.path.join(data_directory, "mnist_test.csv")

            data_loading_start = time()
            # retrieve datasets
            print(f"Loading training dataset from {training_data_file_path}...")
            training_data, train_labels = get_dataset(filepath=training_data_file_path)

            print(f"Loading testing dataset from {test_data_file_path}...")
            test_data, test_labels = get_dataset(filepath=test_data_file_path)
            data_loading_total = time() - data_loading_start
            print(f"Total time for loading the files = {data_loading_total}s")


            data_analysis_start = time()
            # analyze impacts of different parameters
            encoded_dimensions = [4, 16, 32, 64, 128, 256, 512]
            for dim in encoded_dimensions:
                print(f"\nAnalyzing impact of encoded dimension size = {dim}...")
                print(f"Parameters: Learning rate = {LEARNING_RATE} | Epochs = {EPOCHS}")
                analyze_encoded_dimension_size_impact(training_data=training_data, 
                                                     test_data=test_data,
                                                     labels=test_labels, 
                                                     encoded_dimension=dim)
                
            learning_rates = [0.001, 0.01, 0.03, 0.1, 0.3]
            for rate in learning_rates:
                print(f"\nAnalyzing impact of learning rate = {rate}...")
                print(f"Parameters: Encoded dimension = {ENCODED_DIMENSION} | Epochs = {EPOCHS}")
                analyze_learning_rate_impact(training_data=training_data, 
                                             test_data=test_data,
                                             labels=test_labels,
                                             learning_rate=rate)
                
            epoch_numbers = [2, 5, 10, 20]
            for num in epoch_numbers:
                print(f"\nAnalyzing impact of number of epochs = {num}...")
                print(f"Parameters: Learning rate = {LEARNING_RATE} | Encoded dimension = {ENCODED_DIMENSION}")
                analyze_epochs_number_impact(training_data=training_data, 
                                             test_data=test_data,
                                             labels=test_labels,
                                             epochs=num)
            
            data_analysis_total = time() - data_analysis_start
            print(f"Total time spend analysing and training neural network = {data_analysis_total}s")
        
        else:
            raise TypeError("The provided path is not a directory")

    else:
        raise Exception("Wrong arguments given")
    

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)