import sys
from utils import get_dataset
from auto_encoder import AutoEncoder


INPUT_DIMENSION = 784
ENCODED_DIMENSION = 28
LEARNING_RATE = 0.003

def main(args: list[str]):
    if len(args) == 1:
        file_path = args[0]
        x_data, y_data = get_dataset(filepath=file_path)
        auto_encoder = AutoEncoder(input_dim=INPUT_DIMENSION, encoded_dim=ENCODED_DIMENSION, learning_rate=LEARNING_RATE)
        auto_encoder.train(x_train=x_data)

    else:
        raise Exception("Too many arguments given")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)