import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch

from src.logging import console, logger

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

def load_data() -> Bunch:
    """
    Load the California Housing dataset.
    
    Returns:
        Bunch: A dictionary-like object containing data, target, and metadata.
    """
    return fetch_california_housing(as_frame=True)

def visualise_data(data: Bunch, plotting: bool = False) -> None:
    """
    Visualise the dataset and save plots to the designated plots directory.

    Args:
        data (Bunch): A dictionary-like object containing data, target, and metadata.
        plotting (bool, optional): Whether to generate plots. Defaults to False.
    
    Returns:
        None
    """
    console.section("Data Description")
    console.print(data.DESCR)

    console.section("Data Features Overview")
    console.print(data.data.head())
    
    console.section("Data Target Overview")
    console.print(data.target.head())

    console.section("Data Features Statistics")
    console.print(data.data.describe())

    console.section("Data Target Statistics")
    console.print(data.target.describe())

    if not plotting:
        return
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    pairwise_plot_data = data.frame.copy()

    pairwise_plot_data["MedHouseVal"] = pd.qcut(
        pairwise_plot_data["MedHouseVal"], 6, retbins=False
    ).apply(lambda x: x.mid)

    console.section("Generating Pairwise Plots...")
    _ = sns.pairplot(data=pairwise_plot_data, hue="MedHouseVal", palette="viridis")
    plt.savefig(PLOTS_DIR / "california_housing_pairplot.png")
    plt.close()
    logger.info("Pairwise plot saved to %s", PLOTS_DIR / "california_housing_pairplot.png")
    
    console.section("Generating Histograms...")
    data.frame.hist(bins=50, figsize=(20, 15))
    plt.savefig(PLOTS_DIR / "california_housing_histogram.png")
    plt.close()
    logger.info("Histogram saved to %s", PLOTS_DIR / "california_housing_histogram.png")

def main() -> None:
    """
    Main function.

    Args:
        None    
    Returns:
        None
    """
    console.header("Data Visualisation - California Housing")
    california_housing = load_data()
    visualise_data(data=california_housing)

if __name__ == "__main__":
    main()