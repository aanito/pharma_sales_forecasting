import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from loguru import logger

logger.add("../logs/holiday_sales_behavior.log")

def holiday_sales_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Analyzing sales behavior around holidays...")
    sns.lineplot(data=df, x='Date', y='Sales', hue='StateHoliday')
    plt.title('Sales Behavior Around Holidays')
    plt.savefig(f"{output_dir}/holiday_sales_behavior.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holiday Sales Behavior Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
    args = parser.parse_args()

    holiday_sales_analysis(args.input, args.output_dir)