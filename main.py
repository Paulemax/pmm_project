from data_preprocessing import prepare_data
from models import main


def main()-> None:
    prepare_data()
    main()


if __name__ == "__main__":
    main()
