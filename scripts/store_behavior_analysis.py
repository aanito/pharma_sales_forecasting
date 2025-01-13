import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from loguru import logger

logger.add("../logs/store_behavior_analysis.log")

def store_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Analyzing store attributes...")
    sns.boxplot(x='StoreType', y='Sales', data=df)
    plt.title('Sales by Store Type')
    plt.savefig(f"{output_dir}/store_type_sales.png")
    plt.close()

    sns.boxplot(x='Assortment', y='Sales', data=df)
    plt.title('Sales by Assortment Type')
    plt.savefig(f"{output_dir}/assortment_type_sales.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store Behavior Analysis Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
    args = parser.parse_args()

    store_analysis(args.input, args.output_dir)