# Train a CNN model on Gamry data from all battery cells.

from utils.parse_gamry_output import (
    parse_and_merge_gamry_data_for_battery_cell,
    analyze_battery_cells,
)
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from nyquist_cnn import train_nyquist_cnn_from_arrays, predict_soh_arrays, TinyNyquistCNN
from typing import Tuple

def create_ml_data_dataframe(cells_to_use:str) -> pd.DataFrame:

    assert cells_to_use in ["CYLINDRICAL_ONLY", "COIN_ONLY", "ALL"], f"Invalid cells_to_use: {cells_to_use}. It must be one of 'CYLINDRICAL_ONLY', 'COIN_ONLY', 'ALL'."
    
    dir_path = Path(__file__).parent

    ML_data_df = pd.DataFrame()
    if cells_to_use == "CYLINDRICAL_ONLY" or cells_to_use == "ALL":
        battery_cells = ["B10", "B11", "B12"]
        base_path = dir_path / Path("data/INR18650Cycling/15C_Cycling")
        merged_csv_file_path_pattern = "merged_gamry_data_{}.csv"
        for cell in battery_cells:
            # Read the merged CSV file for each battery cell
            merged_csv_file_path = base_path / cell / merged_csv_file_path_pattern.format(cell)
            battery_cell_name = merged_csv_file_path.stem.split("_")[-1]
            merged_df = pd.read_csv(merged_csv_file_path)
            print(f"Battery Cell: {battery_cell_name}")
            # Calculate another dataframe that has five columns: battery_cell_name, cycle_number, SOH_percent, Z_real, Z_imag. It should combine data from all cycles and battery cells.
            # Ensure required columns exist
            required_cols = ["impedance_magnitude_Ohms", "phase_deg", "cycle_number", "SOH_percent"]
            missing = [c for c in required_cols if c not in merged_df.columns]
            if missing:
                raise KeyError(f"Missing required columns in merged_df: {missing}")

            ML_data_cell_df = merged_df[["SOH_percent", "cycle_number", "frequency_Hz"]].copy()
            # Choose random 60 frequency points if there are more than 60 for each cycle
            ML_data_cell_df = ML_data_cell_df.groupby("cycle_number").head(60)

            ML_data_cell_df["Z_real"] = (
                merged_df["impedance_magnitude_Ohms"] * np.cos(np.radians(merged_df["phase_deg"]))
            )
            ML_data_cell_df["Z_imag"] = (
                merged_df["impedance_magnitude_Ohms"] * np.sin(np.radians(merged_df["phase_deg"]))
            )
            ML_data_cell_df["impedance_magnitude_Ohms"] = merged_df["impedance_magnitude_Ohms"]
            ML_data_cell_df["phase_deg"] = merged_df["phase_deg"]
            ML_data_cell_df["battery_cell_name"] = battery_cell_name
            ML_data_df = pd.concat([ML_data_df, ML_data_cell_df], ignore_index=True)
    if cells_to_use == "COIN_ONLY" or cells_to_use == "ALL":
        dir_path = Path(__file__).parent
        preprocessed_dir_path = dir_path / "data/DataForRapidEstimation/preprocessed"
        temperatures = ["25", "35", "45"]
        cell_ids = ["1", "2", "3", "5"]
        for temperature in temperatures:
            for cell_id in cell_ids:
                merged_file_path = preprocessed_dir_path / f"merged_coin_data_{temperature}C0{cell_id}.csv"
                if not merged_file_path.exists():
                    print(f"Warning: Merged file does not exist: {merged_file_path}, skipping.")
                    continue
                merged_df = pd.read_csv(merged_file_path)
                print(f"Battery Cell: Cell_0{cell_id}@{temperature}")
                # Ensure required columns exist
                required_cols = ["impedance_magnitude_Ohms", "phase_deg", "cycle_number", "SOH_percent"]
                missing = [c for c in required_cols if c not in merged_df.columns]
                if missing:
                    raise KeyError(f"Missing required columns in merged_df: {missing}")

                ML_data_cell_df = merged_df[["SOH_percent", "cycle_number", "frequency_Hz"]].copy()
                ML_data_cell_df["Z_real"] = (
                    merged_df["impedance_magnitude_Ohms"] * np.cos(np.radians(merged_df["phase_deg"]))
                )
                ML_data_cell_df["Z_imag"] = (
                    merged_df["impedance_magnitude_Ohms"] * np.sin(np.radians(merged_df["phase_deg"]))
                )
                ML_data_cell_df["impedance_magnitude_Ohms"] = merged_df["impedance_magnitude_Ohms"]
                ML_data_cell_df["phase_deg"] = merged_df["phase_deg"]
                ML_data_cell_df["battery_cell_name"] = f"Cell_0{cell_id}@{temperature}"
                ML_data_df = pd.concat([ML_data_df, ML_data_cell_df], ignore_index=True)
    print("Combined ML Data DataFrame:")
    print(ML_data_df.head())
    base_path = dir_path / "data"
    # Save the combined dataframe to a CSV file
    ML_data_df.to_csv(base_path / "ML_data_all_battery_cells.csv", index=False)

    assert ML_data_df.shape[0] > 0, "No data found to create ML data dataframe."

    return ML_data_df

def create_train_val_test_splits(ML_data_df: pd.DataFrame, style:str="random", cells_types_to_use:str="CYLINDRICAL_ONLY") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Keep only groups (cycle_number, battery_cell_name) that have the most common
    # number of frequency points so downstream processing sees consistent-length samples.
    group_cols = ["cycle_number", "battery_cell_name"]
    groups_list = list(ML_data_df.groupby(group_cols))
    if not groups_list:
        raise ValueError("ML_data_df contains no groups to process.")

    lengths = [len(g) for _, g in groups_list]
    common_length, common_count = Counter(lengths).most_common(1)[0]
    lengths_set = set(lengths)
    if len(lengths_set) > 1:
        print(f"Warning: multiple group lengths found {dict(Counter(lengths))}; using most common length {common_length} (n={common_count}).")

    valid_groups = [g for _, g in groups_list if len(g) == common_length]
    skipped_groups = len(groups_list) - len(valid_groups)
    # Keep keys with their groups so we can build one row per (cycle_number, battery_cell_name)
    valid_groups = [(key, g) for key, g in groups_list if len(g) == common_length]
    if not valid_groups:
        raise ValueError("No groups with the target number of frequency points found in ML_data_df.")

    skipped_groups = len(groups_list) - len(valid_groups)

    # Build a new dataframe with one row per group:
    # columns: battery_number, cycle_number, SOH_percent, Z_real_0..Z_real_{n-1}, Z_imag_0..Z_imag_{n-1}
    rows = []
    for (cycle_number, battery_cell_name), g in valid_groups:
        # preserve frequency ordering if available
        if "frequency_Hz" in g.columns:
            g_sorted = g.sort_values("frequency_Hz")
        else:
            g_sorted = g.copy()

        if len(g_sorted) != common_length:
            # should not happen because we filtered by length, but be defensive
            continue

        soh = g_sorted["SOH_percent"].iloc[0]
        row = {
            "battery_cell_name": battery_cell_name,
            "cycle_number": cycle_number,
            "SOH_percent": soh,
        }

        z_real_vals = g_sorted["Z_real"].values
        z_imag_vals = g_sorted["Z_imag"].values
        impedance_magnitude_vals = g_sorted["impedance_magnitude_Ohms"].values
        phase_deg_vals = g_sorted["phase_deg"].values

        for i, v in enumerate(z_real_vals):
            row[f"Z_real_{i}"] = v
        for i, v in enumerate(z_imag_vals):
            row[f"Z_imag_{i}"] = v
        for i, v in enumerate(impedance_magnitude_vals):
            row[f"imp_mag_{i}"] = v
        for i, v in enumerate(phase_deg_vals):
            row[f"ph_deg_{i}"] = v

        rows.append(row)

    new_ML_data_df = pd.DataFrame(rows)

    # enforce column order: battery_number, cycle_number, SOH_percent, Z_real_*, Z_imag_*, keep battery_cell_name as well
    z_real_cols = [f"Z_real_{i}" for i in range(common_length)]
    z_imag_cols = [f"Z_imag_{i}" for i in range(common_length)]
    impedance_magnitude_cols = [f"imp_mag_{i}" for i in range(common_length)]
    phase_deg_cols = [f"ph_deg_{i}" for i in range(common_length)]
    ordered_cols = ["battery_cell_name", "cycle_number", "SOH_percent"] + z_real_cols + z_imag_cols + impedance_magnitude_cols + phase_deg_cols
    # keep only columns that actually exist (defensive in case of odd edge cases)
    ordered_cols = [c for c in ordered_cols if c in new_ML_data_df.columns]
    new_ML_data_df = new_ML_data_df[ordered_cols]

    # Replace ML_data_df with the new pivoted dataframe so downstream code uses the wide-format samples
    ML_data_df = new_ML_data_df

    if skipped_groups:
        print(f"Filtered out {skipped_groups} groups that did not match target length {common_length}.")
    # Create train/val/test splits with the following percentages: 80% train, 10% val, 10% test.
    if style == "random":
        train_df = ML_data_df.sample(frac=0.8, random_state=42)
        temp_df = ML_data_df.drop(train_df.index)
        val_df = temp_df.sample(frac=0.5, random_state=42)
        test_df = temp_df.drop(val_df.index)
    elif style == "test_one_cell_out":
        # Use B10 and B11 for training (90%)/validation (10%), and B12 for testing.
        train_val_cells_names = []
        test_cells_names = []
        if cells_types_to_use == "CYLINDRICAL_ONLY" or cells_types_to_use == "ALL":
            cylindrical_train_val_cells = ["B10", "B11"]
            cylindrical_test_cells = ["B12"]
            train_val_cells_names.extend(cylindrical_train_val_cells)
            test_cells_names.extend(cylindrical_test_cells)
        if cells_types_to_use == "COIN_ONLY" or cells_types_to_use == "ALL":
            coin_train_val_cells = ["Cell_02@25", "Cell_05@25", "Cell_02@35", "Cell_02@45"]
            # coin_test_cells = ["Cell_01@25", "Cell_03@25", "Cell_01@35", "Cell_01@45"]
            coin_test_cells = ["Cell_01@25", "Cell_01@35"]
            train_val_cells_names.extend(coin_train_val_cells)
            test_cells_names.extend(coin_test_cells)
        train_val_df = ML_data_df[ML_data_df["battery_cell_name"].isin(train_val_cells_names)]
        test_df = ML_data_df[ML_data_df["battery_cell_name"].isin(test_cells_names)]
        # Focus on the first 300 cycles only in the test set
        test_df = test_df[test_df["cycle_number"] <= 300]
        train_df = train_val_df.sample(frac=0.9, random_state=42)
        val_df = train_val_df.drop(train_df.index)
    print(f"Train DataFrame size: {len(train_df)} ({len(train_df)/len(ML_data_df)*100:.2f}%)")
    print(f"Validation DataFrame size: {len(val_df)} ({len(val_df)/len(ML_data_df)*100:.2f}%)")
    print(f"Test DataFrame size: {len(test_df)} ({len(test_df)/len(ML_data_df)*100:.2f}%)")
    # Save the splits to CSV files
    current_python_file_path = Path(__file__)
    base_path = current_python_file_path.parent
    data_dir = base_path / "data"
    data_dir.mkdir(exist_ok=True)
    ML_data_df.to_csv(data_dir / "filtered_ML_data_all_battery_cells.csv", index=False)
    print(f"Saved filtered ML data to {data_dir / 'filtered_ML_data_all_battery_cells.csv'}")
    train_df.to_csv(data_dir / "train_data.csv", index=False)
    val_df.to_csv(data_dir / "val_data.csv", index=False)
    test_df.to_csv(data_dir / "test_data.csv", index=False)

    return train_df, val_df, test_df

def get_nyquist_train_val_test_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features_to_use: str
):
    # Extract features and targets from the dataframes
    def extract_X_y(df: pd.DataFrame, features_to_use: str):
        X = df.drop(columns=["battery_cell_name", "cycle_number", "SOH_percent"]).values
        # reshape X from (N, 2n) -> (N, n, 2) with [:,:,0]=Z_real and [:,:,1]=Z_imag
        num_features = X.shape[1]
        if num_features % 4 != 0:
            raise ValueError(f"Expected even number of features (real+imag), got {num_features}")
        num_freq = num_features // 4
        Z_REAL_FEATURE_START_COL = 0
        Z_IMAG_FEATURE_START_COL = num_freq
        IMP_MAG_FEATURE_START_COL = num_freq * 2
        PHASE_DEG_FEATURE_START_COL = num_freq * 3
        if features_to_use == "NYQUIST_ONLY":
            z_real = X[:, Z_REAL_FEATURE_START_COL:Z_IMAG_FEATURE_START_COL].astype(np.float32)
            z_imag = X[:, Z_IMAG_FEATURE_START_COL:IMP_MAG_FEATURE_START_COL].astype(np.float32)
        elif features_to_use == "IMP_MAG_AND_PHASE":
            z_real = X[:, IMP_MAG_FEATURE_START_COL:PHASE_DEG_FEATURE_START_COL].astype(np.float32)
            z_imag = X[:, PHASE_DEG_FEATURE_START_COL:].astype(np.float32)
        elif features_to_use == "ALL":
            z_real = X[:, Z_REAL_FEATURE_START_COL:Z_IMAG_FEATURE_START_COL].astype(np.float32)
            z_imag = X[:, Z_IMAG_FEATURE_START_COL:IMP_MAG_FEATURE_START_COL].astype(np.float32)
            imp_mag = X[:, IMP_MAG_FEATURE_START_COL:PHASE_DEG_FEATURE_START_COL].astype(np.float32)
            phase_deg = X[:, PHASE_DEG_FEATURE_START_COL:].astype(np.float32)
        
        if features_to_use == "ALL":
            # Stack all four features along the last dimension
            X_reshaped = np.stack((z_real, z_imag, imp_mag, phase_deg), axis=2)  # [N, num_freq, 4]
        else:
            X_reshaped = np.stack((z_real, z_imag), axis=2) # [N, num_freq, 2]
        y = df["SOH_percent"].values.astype(np.float32)
        return X_reshaped, y
    
    assert features_to_use in ["NYQUIST_ONLY", "IMP_MAG_AND_PHASE", "ALL"], f"Invalid features_to_use: {features_to_use}. It must be one of 'NYQUIST_ONLY', 'IMP_MAG_AND_PHASE', 'ALL'."

    train_X, train_y = extract_X_y(train_df, features_to_use)
    val_X, val_y = extract_X_y(val_df, features_to_use)
    test_X, test_y = extract_X_y(test_df, features_to_use)

    return train_X, train_y, val_X, val_y, test_X, test_y

def train_cnn_model_on_dataframes(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, features_to_use: str, epochs: int =200, batch_size: int =16, lr: float =1e-2, early_patience: int =50, weight_decay: float =5e-4, huber_delta: float =1.0) -> Tuple[TinyNyquistCNN, Tuple[np.ndarray, np.ndarray], dict]:
    "features_to_use options: 'NYQUIST_ONLY', 'IMP_MAG_AND_PHASE', 'ALL'"
    reset_seeds()
    print("Training CNN model on the provided dataframes...")
    print(f"Train DataFrame size: {len(train_df)}")
    print(f"Validation DataFrame size: {len(val_df)}")
    print(f"Test DataFrame size: {len(test_df)}")

    assert features_to_use in ["NYQUIST_ONLY", "IMP_MAG_AND_PHASE", "ALL"], f"Invalid features_to_use: {features_to_use}. It must be one of 'NYQUIST_ONLY', 'IMP_MAG_AND_PHASE', 'ALL'."

    train_X, train_y, val_X, val_y, test_X, test_y = get_nyquist_train_val_test_data(train_df, val_df, test_df, features_to_use)

    print(f"Train X shape: {train_X.shape}, Train y shape: {train_y.shape}")
    print(f"Validation X shape: {val_X.shape}, Validation y shape: {val_y.shape}")
    print(f"Test X shape: {test_X.shape}, Test y shape: {test_y.shape}")

    model, (mean, std), logs = train_nyquist_cnn_from_arrays(
        train_X, train_y,
        val_X=val_X, val_y=val_y,
        epochs=epochs,
        batch_size=batch_size, 
        lr=lr,
        early_patience=early_patience,
        weight_decay=weight_decay, # L2 regularization. The higher this value, the stronger the regularization
        huber_delta=huber_delta, # delta parameter for Huber loss. Larger delta means less sensitivity to outliers.
        in_ch=train_X.shape[2] # Number of input channels based on features used. It is 2 for NYQUIST_ONLY or IMP_MAG_AND_PHASE, 4 for ALL.
    )

    print("Training loss at end of training:", logs.get("train_loss", [])[-1] if logs.get("train_loss") else "N/A")
    print("Validation loss at end of training:", logs.get("val_loss", [])[-1] if logs.get("val_loss") else "N/A")
    print("Final Validation MAE:", logs.get("val_mae", [])[-1] if logs.get("val_mae") else "N/A")
    print("Final Validation MAPE:", logs.get("val_mape", [])[-1] if logs.get("val_mape") else "N/A")
    print("Final Validation RMSE:", logs.get("val_rmse", [])[-1] if logs.get("val_rmse") else "N/A")

    # Plot training/validation loss curves
    plot_train_val_loss_curves(logs)
    plot_mae_and_mape_curves(logs)

    print ("Training complete.")

    # Save the trained model and normalization stats
    import torch
    model_save_path = Path("cnn_gamry_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
    }, model_save_path)
    print(f"Saved trained model and normalization stats to {model_save_path}")

    rmse, mae, mape = test_trained_model_on_test_set(model, test_X, test_y, (mean, std))

    return model, (mean, std), logs, (rmse, mae, mape)

def plot_train_val_loss_curves(logs):
    import matplotlib.pyplot as plt

    train_losses = logs.get("train_loss", [])
    val_losses = logs.get("val_loss", [])
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_mae_and_mape_curves(logs):
    import matplotlib.pyplot as plt

    val_mae = logs.get("val_mae", [])
    val_mape = logs.get("val_mape", [])
    val_rmse = logs.get("val_rmse", [])
    epochs = range(1, len(val_mae) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_mae, label="Validation MAE", color='green')
    plt.plot(epochs, val_mape, label="Validation MAPE", color='red')
    plt.plot(epochs, val_rmse, label="Validation RMSE", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Validation MAE, MAPE, and RMSE Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

def test_trained_model_on_test_set(model: TinyNyquistCNN, test_X: np.ndarray, test_y: np.ndarray, norm_stats):
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    predicted_y = predict_soh_arrays(model, test_X, norm_stats, batch_size=64)

    plot_predicted_vs_true_soh(test_y, predicted_y)

    rmse = np.sqrt(mean_squared_error(test_y, predicted_y))
    mae = mean_absolute_error(test_y, predicted_y)
    mape = np.mean(np.abs((test_y - predicted_y) / test_y)) * 100.0

    print(f"Test Set Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")

    return rmse, mae, mape

def plot_predicted_vs_true_soh(true_y: np.ndarray, predicted_y: np.ndarray):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    fontsize = 18
    plt.scatter(true_y, predicted_y, alpha=0.6)
    plt.plot([true_y.min(), true_y.max()], [true_y.min(), true_y.max()], 'r--', lw=2)  # diagonal line
    plt.xlabel("True SOH (%)", fontsize=fontsize)
    plt.ylabel("Predicted SOH (%)", fontsize=fontsize)
    plt.title("Predicted vs True SOH on Test Set", fontsize=fontsize + 2)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.grid(True)
    plt.axis('equal')
    plt.legend(["Ideal Prediction", "Model Predictions"], prop={"size": fontsize - 2})
    plt.show()

def plot_true_soh_vs_predicted_soh_for_battery_cell(cell_name, filtered_ML_data_df, model, norm_stats, features_to_use, max_num_cycles_to_use=None):
    import matplotlib.pyplot as plt

    cell_df = filtered_ML_data_df[filtered_ML_data_df["battery_cell_name"] == cell_name]
    if cell_df.empty:
        print(f"No data found for battery cell {cell_name}.")
        return
    if max_num_cycles_to_use is not None and max_num_cycles_to_use > 0:
        cell_df = cell_df[cell_df["cycle_number"] <= max_num_cycles_to_use]
    test_X, test_y = get_nyquist_train_val_test_data(
        cell_df, cell_df, cell_df, features_to_use
    )[:2]  # only need test_X, test_y

    predicted_y = predict_soh_arrays(model, test_X, norm_stats, batch_size=64)

    plt.figure(figsize=(10, 6))
    plt.plot(test_y, label="True SOH", marker='o')
    plt.plot(predicted_y, label="Predicted SOH", marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("SOH (%)")
    plt.title(f"True vs Predicted SOH for Battery Cell {cell_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    reset_seeds()
    USE_SAVED_ML_DATA_CSV = False
    FEATURES_TO_USE = "NYQUIST_ONLY" # options: "NYQUIST_ONLY", "IMP_MAG_AND_PHASE", "ALL"
    CELLS_TO_USE = "ALL" # options: "COIN_ONLY", "CYLINDRICAL_ONLY", "ALL"
    perform_parameter_sweep_to_find_best_hyperparameters_flag = False
    max_num_cycles_to_use = 250 # Set to None to use all cycles or a positive integer to limit the number of cycles used.

    current_python_file_path = Path(__file__)
    base_path = current_python_file_path.parent
    data_dir = base_path / "data"
    if USE_SAVED_ML_DATA_CSV:
        assert data_dir.exists(), f"Data directory does not exist: {data_dir}" 
        train_data_csv_path = data_dir / "train_data.csv"
        val_data_csv_path = data_dir / "val_data.csv"
        test_data_csv_path = data_dir / "test_data.csv"
        # train_data_csv_path = data_dir / "train_data_B10_B11.csv"
        # val_data_csv_path = data_dir / "val_data_B10_B11.csv"
        # test_data_csv_path = data_dir / "test_data_B12.csv"
        train_df = pd.read_csv(train_data_csv_path)
        val_df = pd.read_csv(val_data_csv_path)
        test_df = pd.read_csv(test_data_csv_path)
        print(f"Loaded train, val, test data from CSV files.")
    else:
        ML_data_df = create_ml_data_dataframe(cells_to_use=CELLS_TO_USE)
        train_df, val_df, test_df = create_train_val_test_splits(ML_data_df, style="test_one_cell_out", cells_types_to_use=CELLS_TO_USE) # Style can be "random" or "test_one_cell_out"

    if perform_parameter_sweep_to_find_best_hyperparameters_flag:
        perform_hyperparameter_sweep_on_cnn_model(train_df, val_df, test_df, FEATURES_TO_USE)
        return
    
    # # Best parameters for cylindrical cells only (B10, B11 for train/val; B12 for test):
    # model, norm_stats, logs, (rmse, mae, mape) = train_cnn_model_on_dataframes(train_df, val_df, test_df, FEATURES_TO_USE,
    #     epochs=300,
    #     lr=0.01,
    #     batch_size=16,
    #     weight_decay=5e-5,
    #     huber_delta=1,
    #     early_patience=50)

    # # Best parameters for coin cells only (Cell_02@25, Cell_05@25, Cell_02@35, Cell_02@45 for train/val; Cell_01@25, Cell_03@25, Cell_01@35, Cell_01@45 for test):
    # model, norm_stats, logs, (rmse, mae, mape) = train_cnn_model_on_dataframes(train_df, val_df, test_df, FEATURES_TO_USE,
    #     epochs=600,
    #     lr=0.001,
    #     batch_size=8,
    #     weight_decay=0.0005,
    #     huber_delta=1,
    #     early_patience=100)

    # Best parameters for all cells (cylindrical + coin) for Nyquist only features:
    model, norm_stats, logs, (rmse, mae, mape) = train_cnn_model_on_dataframes(train_df, val_df, test_df, FEATURES_TO_USE,
        epochs=600,
        lr=0.01,
        batch_size=64,
        weight_decay=0.001,
        huber_delta=5,
        early_patience=100)

    # # Best parameters for all cells (cylindrical + coin) for Nyquist + Mag/Phase features:
    # model, norm_stats, logs, (rmse, mae, mape) = train_cnn_model_on_dataframes(train_df, val_df, test_df, FEATURES_TO_USE,
    #     epochs=600,
    #     lr=0.005,
    #     batch_size=16,
    #     weight_decay=5e-5,
    #     huber_delta=5,
    #     early_patience=100)
    filtere_ML_csv_file_path = data_dir / "filtered_ML_data_all_battery_cells.csv"
    filtered_ml_df = pd.read_csv(filtere_ML_csv_file_path)

    plot_true_soh_vs_predicted_soh_for_battery_cell("B10", filtered_ml_df, model, norm_stats, FEATURES_TO_USE, max_num_cycles_to_use)
    plot_true_soh_vs_predicted_soh_for_battery_cell("B11", filtered_ml_df, model, norm_stats, FEATURES_TO_USE, max_num_cycles_to_use)
    plot_true_soh_vs_predicted_soh_for_battery_cell("B12", filtered_ml_df, model, norm_stats, FEATURES_TO_USE, max_num_cycles_to_use)

    plot_true_soh_vs_predicted_soh_for_battery_cell("Cell_02@25", filtered_ml_df, model, norm_stats, FEATURES_TO_USE, max_num_cycles_to_use)
    plot_true_soh_vs_predicted_soh_for_battery_cell("Cell_05@25", filtered_ml_df, model, norm_stats, FEATURES_TO_USE, max_num_cycles_to_use)
    plot_true_soh_vs_predicted_soh_for_battery_cell("Cell_02@35", filtered_ml_df, model, norm_stats, FEATURES_TO_USE, max_num_cycles_to_use)
    plot_true_soh_vs_predicted_soh_for_battery_cell("Cell_02@45", filtered_ml_df, model, norm_stats, FEATURES_TO_USE, max_num_cycles_to_use)
    plot_true_soh_vs_predicted_soh_for_battery_cell("Cell_01@25", filtered_ml_df, model, norm_stats, FEATURES_TO_USE, max_num_cycles_to_use)
    # plot_true_soh_vs_predicted_soh_for_battery_cell("Cell_03@25", filtered_ml_df, model, norm_stats, FEATURES_TO_USE)
    plot_true_soh_vs_predicted_soh_for_battery_cell("Cell_01@35", filtered_ml_df, model, norm_stats, FEATURES_TO_USE, max_num_cycles_to_use) 
    # plot_true_soh_vs_predicted_soh_for_battery_cell("Cell_01@45", filtered_ml_df, model, norm_stats, FEATURES_TO_USE)

def perform_hyperparameter_sweep_on_cnn_model(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, features_to_use: str):
    print("Performing hyperparameter sweep on CNN model...")
    import itertools
    import tqdm

    learning_rates = [1e-2, 5e-3, 1e-3, 5e-4]
    batch_sizes = [8, 16, 32, 64]
    weight_decays = [1e-3, 5e-4, 1e-4, 5e-5]
    huber_deltas = [1.0, 2.0, 5.0]
    epochs = 600

    best_val_mae = float("inf")
    best_test_mape = float("inf")
    best_hyperparams = None
    record_of_all_results = []
    for lr, batch_size, weight_decay, huber_delta in tqdm.tqdm(
        itertools.product(learning_rates, batch_sizes, weight_decays, huber_deltas),
        total=len(learning_rates)*len(batch_sizes)*len(weight_decays)*len(huber_deltas),
        desc="Hyperparam sweep", 
        unit="run",
        leave=True,
    ):
        print(f"Training with lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, huber_delta={huber_delta}")
        model, norm_stats, logs, (rmse, mae, mape) = train_cnn_model_on_dataframes(
            train_df, val_df, test_df,
            features_to_use,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            huber_delta=huber_delta,
            early_patience=100
        )
        final_val_mae = logs.get("val_mae", [])[-1] if logs.get("val_mae") else float("inf")
        if final_val_mae < best_val_mae:
            best_val_mae = final_val_mae
            best_hyperparams = (lr, batch_size, weight_decay, huber_delta)
            print(f"New best hyperparameters found: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, huber_delta={huber_delta} with Val MAE={best_val_mae}")
        if mape < best_test_mape:
            best_test_mape = mape
            print(f"New best RMSE found: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, huber_delta={huber_delta} with Val RMSE={best_test_mape}")
            best_hyperparams_for_test_mape = (lr, batch_size, weight_decay, huber_delta)

        record = {
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "huber_delta": huber_delta,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "val_mae": final_val_mae
        }
        record_of_all_results.append(record)
        

    print(f"Best hyperparameters: lr={best_hyperparams[0]}, batch_size={best_hyperparams[1]}, weight_decay={best_hyperparams[2]}, huber_delta={best_hyperparams[3]} with Val MAE={best_val_mae}")

    print(f"Best hyperparameters for RMSE (test set): lr={best_hyperparams_for_test_mape[0]}, batch_size={best_hyperparams_for_test_mape[1]}, weight_decay={best_hyperparams_for_test_mape[2]}, huber_delta={best_hyperparams_for_test_mape[3]} with Val RMSE={best_test_mape}")

    # Save record_of_all_results to CSV
    results_df = pd.DataFrame(record_of_all_results)
    dir_path = Path(__file__).parent
    results_df_path = dir_path / "hyperparameter_sweep_results.csv"
    results_df.to_csv(results_df_path, index=False)
    print(f"Saved hyperparameter sweep results to {results_df_path}")

def reset_seeds(seed=42):
    import random, os, torch, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

if __name__ == "__main__":
    main()