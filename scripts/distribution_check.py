import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from loguru import logger

logger.add("../logs/distribution_check.log")

def check_distribution(train_file, test_file, output_dir):
    logger.info(f"Loading data from {train_file} and {test_file}")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    logger.info("Checking promotion distribution in training and test sets...")
    sns.histplot(train_df['Promo'], label='Train', color='blue', kde=False)
    sns.histplot(test_df['Promo'], label='Test', color='orange', kde=False)
    plt.legend()
    plt.title('Promotion Distribution in Train vs Test')
    plt.savefig(f"{output_dir}/promo_distribution_comparison.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Distribution Script")
    parser.add_argument('--train', required=True, help="Path to training CSV file")
    parser.add_argument('--test', required=True, help="Path to test CSV file")
    parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
    args = parser.parse_args()

    check_distribution(args.train, args.test, args.output_dir)