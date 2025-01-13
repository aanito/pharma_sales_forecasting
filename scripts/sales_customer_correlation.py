import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from loguru import logger

logger.add("../logs/sales_customer_correlation.log")

def correlation_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Analyzing correlation between sales and number of customers...")
    correlation = df[['Sales', 'Customers']].corr()
    sns.heatmap(correlation, annot=True)
    plt.title('Correlation Between Sales and Customers')
    plt.savefig(f"{output_dir}/sales_customer_correlation.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales-Customer Correlation Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
    args = parser.parse_args()

    correlation_analysis(args.input, args.output_dir)