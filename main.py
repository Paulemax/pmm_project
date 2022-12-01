from data_preprocessing import prepare_data
from models import train_models


def main()-> None:
    prepare_data()
    train_models()


if __name__ == "__main__":
    main()
