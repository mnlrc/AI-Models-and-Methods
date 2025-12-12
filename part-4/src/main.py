import sys
from utils import get_dataset
from auto_encoder import AutoEncoder
import matplotlib.pyplot as plt
from time import time
import os


INPUT_DIMENSION = 784
ENCODED_DIMENSION = 28
LEARNING_RATE = 0.03
EPOCHS = 20


def plot_losses(train_losses: list[float], test_losses, epochs: int, render: bool = True) -> None:
    """
    Plots the training and testing losses over the epochs.
    Parameters:
        - train_losses (list[float]): the list of training losses recorded at each epoch.
        - test_losses (list[float]): the list of testing losses recorded at each epoch.
        - epochs (int): the number of epochs.
    """
    plt.plot(range(epochs), train_losses, label="Training Loss")
    plt.plot(range(epochs), test_losses, label="Testing Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Mean Squared Error over Epochs")
    plt.grid(visible=True)
    if not os.path.exists("doc/graphs"):
        os.makedirs("doc/graphs")
    plt.savefig(f"doc/graphs/auto_encoder_losses_epochs={epochs}.jpg")
    if render:
        plt.show()



def main(args: list[str]):

    if len(args) == 1:
        data_directory = args[0]
        if os.path.isdir(data_directory):
            training_data_file_path = os.path.join(data_directory, "mnist_train.csv")
            test_data_file_path = os.path.join(data_directory, "mnist_test.csv")

            # retrieve datasets
            print(f"Loading training dataset from {training_data_file_path}...")
            training_data, labels = get_dataset(filepath=training_data_file_path)

            print(f"Loading testing dataset from {test_data_file_path}...")
            test_data, labels = get_dataset(filepath=test_data_file_path)

            # init auto-encoder
            auto_encoder = AutoEncoder(input_dim=INPUT_DIMENSION, encoded_dim=ENCODED_DIMENSION, learning_rate=LEARNING_RATE)

            # train auto-encoder
            print("Starting training...")
            start = time()
            training_losses, test_output, test_losses = auto_encoder.train(x_train=training_data, 
                                                                           x_test=test_data, 
                                                                           epochs=EPOCHS)
            end = time()
            total = end - start
            print("="*50)
            print(f"Training and testing completed in {total:.2f} seconds.")

            # plot results
            plot_losses(train_losses=training_losses, test_losses=test_losses, epochs=EPOCHS)

            # visualize reconstructed image
            for i in range(len(labels)):
                img = test_output[i].reshape(28, 28)
                plt.imshow(img, cmap="gray")
                plt.title(f"Reconstructed Image of Digit {labels[i]}")
                plt.axis("off")
                plt.show()

                inpt = str(input("Press Enter to see the next image, press 's' to save image or 'q' to quit: ")).lower()

                if inpt == "":
                    continue
                elif inpt == "s":
                    if not os.path.exists("doc/reconstructed_images"):
                        os.makedirs("doc/reconstructed_images")
                    plt.imsave(f"doc/reconstructed_images/entry{i}_reconstructed_digit_{labels[i]}.jpg", img, cmap="gray")
                elif inpt == "q":
                    break
                else:
                    break
        
        else:
            raise TypeError("The provided path is not a directory")

    else:
        raise Exception("Too many arguments given")
    

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)