import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from loguru import logger

logger.add("../logs/promo_effect_on_sales.log")

def promo_effect_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Analyzing promo effect on sales...")
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Effect of Promotions on Sales')
    plt.savefig(f"{output_dir}/promo_effect_on_sales.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promo Effect on Sales Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
    args = parser.parse_args()

    promo_effect_analysis(args.input, args.output_dir)