import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from rich.table import Table
from rich import box
from rich.align import Align

from src.linear_regression.linear_regression import LinearRegression
from src.linear_regression.weights_rebasing import LinearRegressionStandardScaleWeightsRebasing
from src.scalers.standard_scaler import StandardScaler
from src.config import config
from src.logging import logger, console

TOLERANCE = 1e-5

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the California Housing dataset.
    
    Args:
        None
    Returns:
        pd.DataFrame: The input data
        pd.Series: The target data
    """
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target

def preprocess_data(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the data by handling skewness and engineering features.

    Args:
        X (pd.DataFrame): The input data
        y (pd.Series): The target data

    Returns:
        pd.DataFrame: The preprocessed input data
        pd.Series: The preprocessed target data
    """
    X = X.copy()
    skewed_cols = ["AveRooms", "AveBedrms", "Population", "AveOccup"]
    for col in skewed_cols:
        cap_value = X[col].quantile(0.99)
        X[col] = np.clip(X[col], a_min=None, a_max=cap_value)
        X[col] = np.log1p(X[col])

    sf_coords = (37.7749, -122.4194)
    la_coords = (34.0522, -118.2437)

    X["Distance_to_SF"] = np.sqrt((X["Latitude"] - sf_coords[0])**2 + (X["Longitude"] - sf_coords[1])**2)
    X["Distance_to_LA"] = np.sqrt((X["Latitude"] - la_coords[0])**2 + (X["Longitude"] - la_coords[1])**2)
    X = X.drop(columns=["Latitude", "Longitude"])

    return X, y

def fit_retain_model(
    X_retain: pd.DataFrame,
    y_retain: pd.Series
) -> tuple[LinearRegression, StandardScaler, float]:
    """
    Fit a model on the retained data.
    
    Args:
        X_retain (pd.DataFrame): The input data
        y_retain (pd.Series): The target data
        
    Returns:
        LinearRegression: The fitted model
        StandardScaler: The fitted scaler
        float: The time taken to fit the model
    """
    console.section("Training Model on Retain Dataset")

    start = time.perf_counter()
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X_retain)
    model = LinearRegression()
    model.fit(X_transformed, y_retain)
    end = time.perf_counter()

    model_time = (end - start) * 1000
    logger.info(f"Step completed in {model_time:.2f} ms")
    return model, scaler, model_time

def fit_full_model(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[LinearRegression, StandardScaler, float]:
    """
    Fit a model on the full data.
    
    Args:
        X (pd.DataFrame): The input data
        y (pd.Series): The target data
        
    Returns:
        LinearRegression: The fitted model
        StandardScaler: The fitted scaler
        float: The time taken to fit the model
    """
    console.section("Training Model on Full Dataset")

    start = time.perf_counter()
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_transformed, y)
    end = time.perf_counter()

    model_time = (end - start) * 1000
    logger.info(f"Step completed in {model_time:.2f} ms")
    return model, scaler, model_time

def fit_forget_model(
    X_forget: pd.DataFrame,
    y_forget: pd.Series,
    full_scaler: StandardScaler,
    full_model: LinearRegression
) -> tuple[LinearRegression, StandardScaler, float]:
    """
    Fit a model on the forget data with proper weights rebasing.
    
    Args:
        X_forget (pd.DataFrame): The input data
        y_forget (pd.Series): The target data
        full_scaler (StandardScaler): The scaler for the full data
        full_model (LinearRegression): The model for the full data
        
    Returns:
        LinearRegression: The fitted model
        StandardScaler: The fitted scaler
        float: The time taken to fit the model
    """
    console.section("Training Model on Forget Dataset")

    start = time.perf_counter()
    scaler = full_scaler.forget(X_forget)
    X_transformed = full_scaler.transform(X_forget)
    model = full_model.forget(X_transformed, y_forget)
    
    model = model.rebase_weights(
        func=LinearRegressionStandardScaleWeightsRebasing(
            source_scaler=full_scaler,
            target_scaler=scaler
    ))
    end = time.perf_counter()

    model_time = (end - start) * 1000
    logger.info(f"Step completed in {model_time:.2f} ms")
    return model, scaler, model_time

def fit_all_models(
    X_retain: pd.DataFrame,
    y_retain: pd.Series,
    X_forget: pd.DataFrame,
    y_forget: pd.Series,
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[
    LinearRegression,
    StandardScaler,
    LinearRegression,
    StandardScaler,
    LinearRegression,
    StandardScaler,
    float,
    float,
    float
]:
    """
    Fit all models.

    Args:
        X_retain (pd.DataFrame): The input data for the retain dataset
        y_retain (pd.Series): The target data for the retain dataset
        X_forget (pd.DataFrame): The input data for the forget dataset
        y_forget (pd.Series): The target data for the forget dataset
        X (pd.DataFrame): The input data for the full dataset
        y (pd.Series): The target data for the full dataset

    Returns:
        tuple[
            LinearRegression,
            StandardScaler,
            LinearRegression,
            StandardScaler,
            LinearRegression,
            StandardScaler,
            float,
            float,
            float
        ]: The fitted models and the time taken to fit them
    """

    model_retain, scaler_retain, model_retain_time = fit_retain_model(X_retain, y_retain)
    model_full, scaler_full, model_full_time = fit_full_model(X, y)
    model_forget, scaler_forget, model_forget_time = fit_forget_model(X_forget, y_forget, scaler_full, model_full)

    return model_retain, scaler_retain, model_full, scaler_full, model_forget, scaler_forget, model_retain_time, model_full_time, model_forget_time

def handle_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load and preprocess the data.
    
    Args:
        None
        
    Returns:
        pd.DataFrame: The input data for the retain dataset
        pd.Series: The target data for the retain dataset
        pd.DataFrame: The input data for the forget dataset
        pd.Series: The target data for the forget dataset
        pd.DataFrame: The input data for the full dataset
        pd.Series: The target data for the full dataset
    """

    console.section("Loading and Preprocessing Data")

    start = time.perf_counter()
    X, y = load_data()
    X, y = preprocess_data(X, y)
    end = time.perf_counter()

    X_retain, X_forget, y_retain, y_forget = train_test_split(
        X, y,
        test_size=config.TRAIN_TEST_SPLIT,
        random_state=config.SEED
    )
    logger.info(f"X_retain shape: {X_retain.shape}, y_retain shape: {y_retain.shape}")
    logger.info(f"X_forget shape: {X_forget.shape}, y_forget shape: {y_forget.shape}")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Data preparation completed in {(end - start) * 1000:.2f} ms")

    return X_retain, y_retain, X_forget, y_forget, X, y

def parameter_overview(
    retain_model: LinearRegression,
    full_model: LinearRegression,
    forget_model: LinearRegression
) -> None:
    """
    Print the parameter overview of the models.
    
    Args:
        retain_model (LinearRegression): The retain model
        full_model (LinearRegression): The full model
        forget_model (LinearRegression): The forget model
        
    Returns:
        None
    """
    console.section("Model Parameter Overview")
    logger.info(f"◈ Retain Model Weights:\n  {retain_model.weights}")
    logger.info(f"◈ Full Model Weights:\n  {full_model.weights}")
    logger.info(f"◈ Forget Model Weights:\n  {forget_model.weights}")

def model_weight_assertions(
    retain_model: LinearRegression,
    retain_scaler: StandardScaler,
    forget_model: LinearRegression,
    forget_scaler: StandardScaler,
    X: pd.DataFrame
) -> None:
    """
    Perform mathematical parity assertions on the model weights.
    
    Args:
        retain_model (LinearRegression): The retain model
        full_model (LinearRegression): The full model
        forget_model (LinearRegression): The forget model
        
    Returns:
        None
    """
    console.section("Mathematical Parity Assertions - Model Weights")
    
    try:
        np.testing.assert_allclose(forget_model.weights, retain_model.weights, rtol=TOLERANCE, atol=TOLERANCE)
        logger.info(f"Weight Alignment (tolerance: {TOLERANCE}): [bold green]PASSED[/]")
    except AssertionError as e:
        logger.error(f"Weight Alignment (tolerance: {TOLERANCE}): [bold red]FAILED[/]\n{e}")

    X_transformed_retain = retain_scaler.transform(X)
    X_transformed_forget = forget_scaler.transform(X)

    preds_retain = retain_model.predict(X_transformed_retain)
    preds_forget = forget_model.predict(X_transformed_forget)

    try:
        np.testing.assert_allclose(preds_forget, preds_retain, rtol=TOLERANCE, atol=TOLERANCE)
        logger.info(f"Prediction Invariance (tolerance: {TOLERANCE}): [bold green]PASSED[/]")
    except AssertionError as e:
        logger.error(f"Prediction Invariance (tolerance: {TOLERANCE}): [bold red]FAILED[/]\n{e}")
    print()

def scaler_assertions(
    forget_scaler: StandardScaler,
    retain_scaler: StandardScaler
) -> None:
    """
    Perform mathematical parity assertions on the scalers.
    
    Args:
        forget_scaler (StandardScaler): The forget scaler
        retain_scaler (StandardScaler): The retain scaler
        
    Returns:
        None
    """
    console.section("Mathematical Parity Assertions - Scalers")
    
    try:
        np.testing.assert_allclose(forget_scaler.mean, retain_scaler.mean, rtol=TOLERANCE, atol=TOLERANCE)
        logger.info("Scaler Mean Alignment: [bold green]PASSED[/]")
    except AssertionError as e:
        logger.error(f" Scaler Mean Alignment: [bold red]FAILED[/]\n{e}")

    try:
        np.testing.assert_allclose(forget_scaler.std, retain_scaler.std, rtol=TOLERANCE, atol=TOLERANCE)
        logger.info(f"Scaler Standard Deviation Alignment (tolerance: {TOLERANCE}): [bold green]PASSED[/]")
    except AssertionError as e:
        logger.error(f"Scaler Standard Deviation Alignment (tolerance: {TOLERANCE}): [bold red]FAILED[/]\n{e}")
    print()


def model_execution_time_overview(
    retain_model_time: float,
    full_model_time: float,
    forget_model_time: float
) -> None:
    """
    Print the execution time overview of the models.
    
    Args:
        retain_model_time (float): The execution time of the retain model
        full_model_time (float): The execution time of the full model
        forget_model_time (float): The execution time of the forget model
        
    Returns:
        None
    """
    console.section("Model Execution Times")

    table = Table(
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        border_style="white",
        pad_edge=False
    )

    table.add_column("Model Configuration", style="white", width=26)
    table.add_column("Execution Time", justify="right", width=18)

    table.add_row("Retain Model", f"[bold green]{retain_model_time:.2f} ms[/]", style="bold")
    table.add_row("Full Model", f"[bold green]{full_model_time:.2f} ms[/]", style="bold")
    table.add_row("Forget Model", f"[bold green]{forget_model_time:.2f} ms[/]", style="bold")

    console.print(Align.center(table))

def main() -> None:
    """
    Main function to run the benchmarking.
    
    Args:
        None
        
    Returns:
        None
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=75)

    console.header("Linear Regression Forgetting Benchmarking")
    
    X_retain, y_retain, X_forget, y_forget, X, y = handle_data()

    retain_model, retain_scaler, full_model, full_scaler, forget_model, forget_scaler, retain_model_time, full_model_time, forget_model_time = fit_all_models(
        X_retain,
        y_retain,
        X_forget,
        y_forget,
        X,
        y
    )
    
    parameter_overview(retain_model, full_model, forget_model)

    model_weight_assertions(retain_model, retain_scaler, forget_model, forget_scaler, X)

    scaler_assertions(forget_scaler, retain_scaler)

    model_execution_time_overview(retain_model_time, full_model_time, forget_model_time)


if __name__ == "__main__":
    main()