import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from loguru import logger

logger.add("../logs/exploratory_analysis.log")

def plot_distributions(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Plotting distributions...")
    sns.histplot(df['Sales'], kde=True)
    plt.title('Sales Distribution')
    plt.savefig(f"{output_dir}/sales_distribution.png")
    plt.close()

    sns.countplot(x='Promo', data=df)
    plt.title('Promo Distribution')
    plt.savefig(f"{output_dir}/promo_distribution.png")
    plt.close()

    logger.info("Distributions plotted and saved.")

def holiday_sales_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Analyzing holiday sales...")
    sns.boxplot(x='StateHoliday', y='Sales', data=df)
    plt.title('Sales During Holidays')
    plt.savefig(f"{output_dir}/holiday_sales_analysis.png")
    plt.close()

def correlation_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Analyzing correlation between sales and customers...")
    correlation = df[['Sales', 'Customers']].corr()
    sns.heatmap(correlation, annot=True)
    plt.title('Correlation Heatmap')
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

def seasonal_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.isocalendar().week
    
    logger.info("Analyzing seasonal trends...")
    sns.lineplot(data=df, x='Week', y='Sales', hue='Year')
    plt.title('Seasonal Sales Trends')
    plt.savefig(f"{output_dir}/seasonal_sales_trends.png")
    plt.close()

def promo_effect_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Analyzing promo effect on sales...")
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Promo Effect on Sales')
    plt.savefig(f"{output_dir}/promo_effect.png")
    plt.close()

def store_analysis(input_file, output_dir):
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logger.info("Analyzing store attributes...")
    sns.boxplot(x='StoreType', y='Sales', data=df)
    plt.title('Sales by Store Type')
    plt.savefig(f"{output_dir}/store_type_sales.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploratory Analysis Script")
    parser.add_argument('--input', required=True, help="Path to input CSV file")
    parser.add_argument('--output_dir', required=True, help="Directory to save output plots")
    args = parser.parse_args()

    plot_distributions(args.input, args.output_dir)
    holiday_sales_analysis(args.input, args.output_dir)
    correlation_analysis(args.input, args.output_dir)
    seasonal_analysis(args.input, args.output_dir)
    promo_effect_analysis(args.input, args.output_dir)
    store_analysis(args.input, args.output_dir)

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from loguru import logger

# logger.add("../logs/eda.log")

# @logger.catch
# def plot_distributions(train, test):
#     logger.info("Plotting distributions...")
#     sns.histplot(train['Sales'], kde=True)
#     plt.title('Sales Distribution')
#     plt.show()
#     sns.countplot(x='Promo', data=train)
#     plt.title('Promo Distribution in Training Set')
#     plt.show()
#     sns.countplot(x='Promo', data=test)
#     plt.title('Promo Distribution in Test Set')
#     plt.show()

# @logger.catch
# def holiday_sales_analysis(train):
#     logger.info("Analyzing holiday sales...")
#     sns.boxplot(x='StateHoliday', y='Sales', data=train)
#     plt.title('Sales During Holidays')
#     plt.show()

# @logger.catch
# def correlation_analysis(train):
#     logger.info("Analyzing correlation between sales and customers...")
#     correlation = train[['Sales', 'Customers']].corr()
#     sns.heatmap(correlation, annot=True)
#     plt.title('Correlation Heatmap')
#     plt.show()