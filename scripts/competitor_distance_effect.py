import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from loguru import logger

logger.add("../logs/competitor_distance_effect.log")

def competitor_distance_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Analyzing effect of competitor distance on sales...")
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
    plt.title('Effect of Competitor Distance on Sales')
    plt.savefig(f"{output_dir}/competitor_distance_effect.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competitor Distance Effect Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
    args = parser.parse_args()

    competitor_distance_analysis(args.input, args.output_dir)