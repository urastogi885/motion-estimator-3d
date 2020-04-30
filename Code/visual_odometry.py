from sys import argv
from utils.data_prep import *

script, dataset_location = argv


if __name__ == '__main__':
    filenames = extract_locations(str(dataset_location))

