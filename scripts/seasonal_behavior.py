import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from loguru import logger

logger.add("../logs/seasonal_behavior.log")

def seasonal_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    logger.info("Analyzing seasonal purchase behavior...")
    sns.lineplot(data=df, x='Month', y='Sales', hue='Year')
    plt.title('Seasonal Purchase Behavior')
    plt.savefig(f"{output_dir}/seasonal_behavior.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seasonal Behavior Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
    args = parser.parse_args()

    seasonal_analysis(args.input, args.output_dir)