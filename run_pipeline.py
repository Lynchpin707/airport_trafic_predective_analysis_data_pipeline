from src.clean_data import clean_data
from src.predict_prophet import run_prophet

import os

RAW_DATA_PATH = './data/trafic_ma_long.csv'
OUTPUTS_DIR = "./docs/outputs"

if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)

def main():
    # Step 1: Clean the data
    clean_data(RAW_DATA_PATH)
    
    # Step 2: Prophet prediction
    run_prophet()


if __name__ == "__main__":
    main()