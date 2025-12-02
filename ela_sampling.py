from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import os, sys
from pflacco.classical_ela_features import *



def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return its contents as a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The contents of the CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def save_csv(data: pd.DataFrame, file_path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        data (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the output CSV file.
    """
    data.to_csv(file_path, index=False)


def read_x_samples(file_path: str) -> pd.DataFrame:
    """
    Read X samples from a CSV file and return them as a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        np.ndarray: The X samples as a numpy array.
    """
    df = pd.read_csv(file_path)
    return df.values


def get_x_sample_filelist(directory: str) -> List[Path]:
    r"""
    Read multiple X sample files from a directory and return a list of paths.

    Args:
        directory (str): The path to the directory containing CSV files.
    Returns:
        List[Path]: A list of paths to the CSV files in the directory.
    """ 

    dir_path = Path(directory)
    file_list = list(dir_path.rglob("*samples.csv"))
    return file_list

def get_y_sample_filelist(directory: str) -> List[Path]:
    r"""
    Read multiple Y sample files from a directory and return a list of paths.

    Args:
        directory (str): The path to the directory containing CSV files.
    Returns:
        List[Path]: A list of paths to the CSV files in the directory.
    """
    dir_path = Path(directory)
    file_list = list(dir_path.rglob("*evaluations.csv"))
    return file_list

def distill_x_sample_list(file_list: List[Path]) -> List[Tuple[int, int, int, str]]:
    r"""
    Distill a list of X sample file paths into a list of tuples containing
    (dimension, seed, number of samples).

    Args:
    ------------
        file_list (List[Path]): A list of paths to the CSV files.

    Outputs:
    ------------
        - A list[Tuple[int, int, int, str]]: A list of tuples (dimension, seed, n_samples, objective_type).
    """ 
    distilled_list = []
    for file_path in file_list:
        parts = file_path.parts
        try:
            dim_part = [p for p in parts if p.startswith("Dimension_")][0]
            seed_part = [p for p in parts if p.startswith("seed_")][0]
            samples_part = [p for p in parts if p.startswith("Samples_")][0]
            objective_type_part = [p for p in parts if p in ["ELA_extraction","reduction"]][0]

            dim = int(dim_part.split("_")[1])
            seed = int(seed_part.split("_")[1])
            n_samples = int(samples_part.split("_")[1])
            objective_type = objective_type_part

            distilled_list.append((dim, seed, n_samples, objective_type))
        except (IndexError, ValueError):
            print(f"Warning: Could not parse file path '{file_path}'. Skipping.")
            continue
    return distilled_list

def distill_y_sample_list(file_list: List[Path]) -> List[Tuple[int, int, int, str, int, int]]:
    r"""
    Distill a list of Y sample file paths into a list of tuples containing
    (dimension, seed, number of samples, objective_type, function_id, instance_id).

    Args:
    ------------
        file_list (List[Path]): A list of paths to the CSV files.

    Outputs:
    ------------
        - A list[Tuple[int, int, int, str, int, int]]: A list of tuples (dimension, seed, n_samples, objective_type, function_id, instance_id).
    """ 
    distilled_list = []
    for file_path in file_list:
        parts = file_path.parts
        try:
            dim_part = [p for p in parts if p.startswith("Dimension_")][0]
            seed_part = [p for p in parts if p.startswith("seed_")][0]
            samples_part = [p for p in parts if p.startswith("Samples_")][0]
            objective_type_part = [p for p in parts if p in ["ELA_extraction","reduction"]][0]
            function_part = [p for p in parts if p.startswith("f_")][0]
            instance_part = [p for p in parts if p.startswith("id_")][0]

            dim = int(dim_part.split("_")[1])
            seed = int(seed_part.split("_")[1])
            n_samples = int(samples_part.split("_")[1])
            objective_type = objective_type_part
            function_id = int(function_part.split("_")[1])
            instance_id = int(instance_part.split("_")[1])

            distilled_list.append((dim, seed, n_samples, objective_type, function_id, instance_id))
        except (IndexError, ValueError):
            print(f"Warning: Could not parse file path '{file_path}'. Skipping.")
            continue
    return distilled_list

def extract_ela_features(
                         seed:int, 
                         X: np.ndarray, 
                         fX: np.ndarray) -> pd.DataFrame:
    """
    Extract ELA features from given samples and their function values.

    Args:
        seed (int): Random seed used for sampling.
        X (np.ndarray): Input samples of shape (n_samples, dim).
        fX (np.ndarray): Function values at the input samples of shape (n_samples,).

    Returns:
        pd.DataFrame: A DataFrame containing the extracted ELA features.
    """

    # Calculate features using pflacco
    ela_meta = calculate_ela_meta(X, fX)
    ela_distr = calculate_ela_distribution(X, fX)
    nbc = calculate_nbc(X, fX)
    disp = calculate_dispersion(X, fX)
    ic = calculate_information_content(X, fX, seed=seed)
    pca_features = calculate_pca(X, fX)

    # Combine all features into a single pandas DataFrame
    feature_df = pd.DataFrame({**ela_meta, **ela_distr, **nbc, **disp, **ic, **pca_features}, index = [0])

    return feature_df


def main():
    # Get the list of X sample files in the specified directory
    directory = Path(os.getcwd())  # You can change this to any directory you want
    file_x_list = get_x_sample_filelist(directory)
    file_y_list = get_y_sample_filelist(directory)
    
    # Distill the file list to extract dimension, seed, and number of samples
    distilled_x_list = distill_x_sample_list(file_x_list)
    distilled_y_list = distill_y_sample_list(file_y_list)

    # print("Distilled X Sample Files:")
    # for item in distilled_x_list:
    #     print(item)
    
    # print("\nDistilled Y Sample Files:")
    # for item in distilled_y_list:
    #     print(item) 

    for ii, x_file_path in enumerate(file_x_list):
        print(f"Reading: X Sample File {ii+1}: {x_file_path}")

        # Get the corresponding distilled info
        dim, seed, n_samples, objective_type = distilled_x_list[ii]

        print(f"Distilled Info - Dimension: {dim}, Seed: {seed}, Number of Samples: {n_samples}, Objective Type: {objective_type}")

        X_values = read_x_samples(Path(x_file_path))

        for jj, y_file_path in enumerate(file_y_list):
            print(f"Reading: Y Sample File {jj+1}: {y_file_path}")

            # Get the corresponding distilled info
            (y_dim, y_seed, y_n_samples, y_objective_type, function_id, instance_id) = distilled_y_list[jj]

            if (dim == y_dim) and (seed == y_seed) and (n_samples == y_n_samples) and (objective_type == y_objective_type):
                print(f"Match Found for Function ID: {function_id}, Instance ID: {instance_id}")

                Y_df = read_csv(Path(y_file_path))
                fX_values = Y_df['fX'].values

                feature_df = extract_ela_features(seed, X_values, fX_values)

                # Save the features
                save_path = directory.joinpath("ela_features",objective_type,f"Dimension_{dim}",f"seed_{seed}",f"Samples_{n_samples}",f"f_{function_id}",f"id_{instance_id}")
                if save_path.exists() is False:
                    save_path.mkdir(parents=True, exist_ok=True)
                else:
                    continue

                out_file = save_path.joinpath("ela_features.csv")
                save_csv(feature_df, out_file)

                
            
            else:
                continue



    


if __name__ == "__main__":
    main()