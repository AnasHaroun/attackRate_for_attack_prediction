import os
import pandas as pd
import numpy as np

def split_xyz(df: pd.DataFrame, column: str) -> None:
    """
    Splits a column with [x, y, z] format into three separate columns.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        column (str): The name of the column to split.
    """
    if column in df:
        df[[f'{column}_x', f'{column}_y', f'{column}_z']] = pd.DataFrame(df[column].tolist(), index=df.index)
        df.drop(column, axis=1, inplace=True)

def convert_json_to_csv(data_dir: str, columns_to_transform: list) -> None:
    """
    Converts JSON files in a directory to CSV after splitting multi-value columns.

    Args:
        data_dir (str): Path to the directory containing JSON files.
        columns_to_transform (list): List of column names to split into x, y, z.
    """
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            json_path = os.path.join(data_dir, file)
            csv_path = os.path.join(data_dir, file.replace('.json', '.csv'))

            try:
                df = pd.read_json(json_path, lines=True)
                for column in columns_to_transform:
                    split_xyz(df, column)
                df.to_csv(csv_path, index=False)
            except ValueError as e:
                print(f"Error processing {file}: {e}")

def load_dataframes_from_csv(data_dir: str):
    """
    Loads all CSV files in a directory into two lists of DataFrames.

    Args:
        data_dir (str): Path to the directory containing CSV files.

    Returns:
        tuple: (list of regular DataFrames, list of ground truth DataFrames)
    """
    regular_dfs, ground_truth_dfs = [], []

    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            csv_path = os.path.join(data_dir, file)
            df = pd.read_csv(csv_path)
            (ground_truth_dfs if 'GroundTruth' in file else regular_dfs).append(df)

    return regular_dfs, ground_truth_dfs

def display_attack_type_counts(df: pd.DataFrame, column: str = 'type', title: str = "Attack Type Counts") -> None:
    """
    Displays counts and percentages of unique values in the attack type column.

    Args:
        df (pd.DataFrame): The DataFrame containing attack types.
        column (str): The name of the column containing attack identifiers.
        title (str): Title to describe the current analysis section.
    """
    value_counts = df[column].value_counts()
    value_percentages = df[column].value_counts(normalize=True) * 100
    combined = pd.DataFrame({'Count': value_counts, 'Percentage': value_percentages})
    print(f"\n{title}")
    print(combined)

def sort_and_filter_df(df: pd.DataFrame, filter_column: str, filter_value, sort_columns: list) -> pd.DataFrame:
    """
    Filters a DataFrame by value and sorts it by given columns.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        filter_column (str): Column to apply filtering on.
        filter_value: Value to filter by.
        sort_columns (list): Columns to sort by.

    Returns:
        pd.DataFrame: Filtered and sorted DataFrame.
    """
    filtered = df[df[filter_column] == filter_value]
    sorted_df = filtered.sort_values(by=sort_columns)
    return sorted_df

def normalize_attack_labels(df: pd.DataFrame, attack_col: str = 'Attack', valid_attacks: list = [22, 33]) -> pd.DataFrame:
    """
    Normalizes the attack labels in the DataFrame to only retain valid attack codes.

    Args:
        df (pd.DataFrame): The DataFrame with an attack column.
        attack_col (str): The name of the attack column.
        valid_attacks (list): List of valid attack codes to retain.

    Returns:
        pd.DataFrame: DataFrame with normalized attack labels.
    """
    df[attack_col] = df[attack_col].apply(lambda x: x if x in valid_attacks else 0)
    return df
    
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

def add_difference_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Adds difference-based features (first-order derivatives) to a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing position and/or speed columns.
        columns (list): List of column names to compute differences for.

    Returns:
        pd.DataFrame: DataFrame with new difference columns.
    """
    for col in columns:
        diff_col = f'diff_{col}'
        df[diff_col] = df[col].diff().fillna(0)
    return df

def train_and_evaluate_first_attack_model(df: pd.DataFrame, target_col: str = 'Attack') -> dict:
    """
    Trains ML models to detect the first type of attack and evaluates performance.

    Args:
        df (pd.DataFrame): Feature-enhanced DataFrame.
        target_col (str): Name of the target column indicating attack type.

    Returns:
        dict: Trained models and evaluation metrics.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Binary labeling: Treat second attack as benign
    y_binary = y.replace(33, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.5, random_state=42)

    rf_model = RandomForestClassifier()
    gb_model = GradientBoostingClassifier()

    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    rf_preds = rf_model.predict(X_test)
    gb_preds = gb_model.predict(X_test)

    print("\nRandom Forest Classifier Performance:")
    print(classification_report(y_test, rf_preds))

    print("\nGradient Boosting Classifier Performance:")
    print(classification_report(y_test, gb_preds))

    return {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'X_test': X_test,
        'rf_predictions': rf_preds,
        'gb_predictions': gb_preds
    }

def identify_benign_senders(df: pd.DataFrame, attack_label: int = 1) -> pd.DataFrame:
    """
    Identifies senders who never had a specific attack label.

    Args:
        df (pd.DataFrame): Ground truth DataFrame containing 'sender' and 'Attack' columns.
        attack_label (int): The attack value to exclude (e.g., 1 for specific attack type).

    Returns:
        pd.DataFrame: A DataFrame listing benign senders and their occurrence counts.
    """
    # Filter out rows with the specified attack label
    filtered_df = df[df['Attack'] != attack_label]

    # Identify unique benign senders
    benign_senders = set(filtered_df['sender'])

    # Count occurrences in the original ground truth
    occurrences = df[df['sender'].isin(benign_senders)]['sender'].value_counts()

    # Convert to DataFrame
    benign_df = occurrences.reset_index()
    benign_df.columns = ['sender', 'number_of_attacks']

    print("Senders with no attack labeled as '1':")
    print(benign_df)

    return benign_df

def analyze_top_senders(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Placeholder for analyzing top benign senders for potential honeypot logic.

    Args:
        df (pd.DataFrame): DataFrame with sender and number_of_attacks columns.
        top_n (int): Number of top senders to consider.

    Returns:
        pd.DataFrame: DataFrame of top senders (future honeypot candidates).
    """
    # Select top N senders
    top_senders = df.nlargest(top_n, 'number_of_attacks')
    print(f"\nTop {top_n} senders with most benign activity:")
    print(top_senders)
    return top_senders
