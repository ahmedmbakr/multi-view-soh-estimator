# Train a CNN model on Gamry data from all battery cells.

from utils.parse_gamry_output import (
    parse_and_merge_gamry_data_for_battery_cell,
    analyze_battery_cells,
)
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

def create_ml_data_dataframe():
    battery_cells = ["B10", "B11", "B12"]
    base_path = Path("C:/Users/AhmedBakr/Box/INR18650Cycling/15C_Cycling")
    merged_csv_file_path_pattern = "merged_gamry_data_{}.csv"

    ML_data_df = pd.DataFrame()
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
        ML_data_cell_df["Z_real"] = (
            merged_df["impedance_magnitude_Ohms"] * np.cos(np.radians(merged_df["phase_deg"]))
        )
        ML_data_cell_df["Z_imag"] = (
            merged_df["impedance_magnitude_Ohms"] * np.sin(np.radians(merged_df["phase_deg"]))
        )
        ML_data_cell_df["battery_cell_name"] = battery_cell_name
        ML_data_df = pd.concat([ML_data_df, ML_data_cell_df], ignore_index=True)
    print("Combined ML Data DataFrame:")
    print(ML_data_df.head())
    # Save the combined dataframe to a CSV file
    ML_data_df.to_csv(base_path / "ML_data_all_battery_cells.csv", index=False)

    return ML_data_df

def create_train_val_test_splits(ML_data_df: pd.DataFrame):
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

        for i, v in enumerate(z_real_vals):
            row[f"Z_real_{i}"] = v
        for i, v in enumerate(z_imag_vals):
            row[f"Z_imag_{i}"] = v

        rows.append(row)

    new_ML_data_df = pd.DataFrame(rows)

    # enforce column order: battery_number, cycle_number, SOH_percent, Z_real_*, Z_imag_*, keep battery_cell_name as well
    z_real_cols = [f"Z_real_{i}" for i in range(common_length)]
    z_imag_cols = [f"Z_imag_{i}" for i in range(common_length)]
    ordered_cols = ["battery_cell_name", "cycle_number", "SOH_percent"] + z_real_cols + z_imag_cols
    # keep only columns that actually exist (defensive in case of odd edge cases)
    ordered_cols = [c for c in ordered_cols if c in new_ML_data_df.columns]
    new_ML_data_df = new_ML_data_df[ordered_cols]

    # Replace ML_data_df with the new pivoted dataframe so downstream code uses the wide-format samples
    ML_data_df = new_ML_data_df

    if skipped_groups:
        print(f"Filtered out {skipped_groups} groups that did not match target length {common_length}.")
    # Create train/val/test splits with the following percentages: 80% train, 10% val, 10% test.
    train_df = ML_data_df.sample(frac=0.8, random_state=42)
    temp_df = ML_data_df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)
    print(f"Train DataFrame size: {len(train_df)} ({len(train_df)/len(ML_data_df)*100:.2f}%)")
    print(f"Validation DataFrame size: {len(val_df)} ({len(val_df)/len(ML_data_df)*100:.2f}%)")
    print(f"Test DataFrame size: {len(test_df)} ({len(test_df)/len(ML_data_df)*100:.2f}%)")
    # Save the splits to CSV files
    current_python_file_path = Path(__file__)
    base_path = current_python_file_path.parent
    data_dir = base_path / "data"
    data_dir.mkdir(exist_ok=True)
    ML_data_df.to_csv(data_dir / "filtered_ML_data_all_battery_cells.csv", index=False)
    train_df.to_csv(data_dir / "train_data.csv", index=False)
    val_df.to_csv(data_dir / "val_data.csv", index=False)
    test_df.to_csv(data_dir / "test_data.csv", index=False)

    return train_df, val_df, test_df

def train_cnn_model_on_dataframes(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    print("Training CNN model on the provided dataframes...")
    print(f"Train DataFrame size: {len(train_df)}")
    print(f"Validation DataFrame size: {len(val_df)}")
    print(f"Test DataFrame size: {len(test_df)}")
    
    train_X = train_df.drop(columns=["battery_cell_name", "cycle_number", "SOH_percent"]).values
    # reshape train_X from (N, 122) -> (N, 61, 2) with [:,:,0]=Z_real and [:,:,1]=Z_imag
    num_features = train_X.shape[1]
    if num_features % 2 != 0:
        raise ValueError(f"Expected even number of features (real+imag), got {num_features}")
    num_freq = num_features // 2

    z_real = train_X[:, :num_freq].astype(np.float32)
    z_imag = train_X[:, num_freq:].astype(np.float32)
    train_X = np.stack((z_real, z_imag), axis=2)
    train_y = train_df["SOH_percent"].values
    # ensure train targets are float32
    train_y = train_y.astype(np.float32)

    # prepare validation set (same processing as train_X)
    _val_raw = val_df.drop(columns=["battery_cell_name", "cycle_number", "SOH_percent"]).values
    if _val_raw.shape[1] != num_features:
        raise ValueError(f"Validation features ({_val_raw.shape[1]}) do not match train features ({num_features})")
    _val_z_real = _val_raw[:, :num_freq].astype(np.float32)
    _val_z_imag = _val_raw[:, num_freq:].astype(np.float32)
    val_X = np.stack((_val_z_real, _val_z_imag), axis=2)
    val_y = val_df["SOH_percent"].values.astype(np.float32)

    # prepare test set (same processing as train_X)
    _test_raw = test_df.drop(columns=["battery_cell_name", "cycle_number", "SOH_percent"]).values
    if _test_raw.shape[1] != num_features:
        raise ValueError(f"Test features ({_test_raw.shape[1]}) do not match train features ({num_features})")
    _test_z_real = _test_raw[:, :num_freq].astype(np.float32)
    _test_z_imag = _test_raw[:, num_freq:].astype(np.float32)
    test_X = np.stack((_test_z_real, _test_z_imag), axis=2)
    test_y = test_df["SOH_percent"].values.astype(np.float32)

    print(f"Train X shape: {train_X.shape}, Train y shape: {train_y.shape}")
    print(f"Validation X shape: {val_X.shape}, Validation y shape: {val_y.shape}")
    print(f"Test X shape: {test_X.shape}, Test y shape: {test_y.shape}")

    from nyquist_cnn import train_nyquist_cnn_from_arrays
    model, (mean, std), logs = train_nyquist_cnn_from_arrays(
        train_X, train_y,
        val_X=val_X, val_y=val_y,
        epochs=50,
        batch_size=32,
        lr=1e-3,
    )
    print("Training logs:")
    for key, values in logs.items():
        print(f"{key}: {values}")

    print ("Training complete.")

    

def main():
    USE_SAVED_ML_DATA_CSV = True
    if USE_SAVED_ML_DATA_CSV:
        current_python_file_path = Path(__file__)
        base_path = current_python_file_path.parent
        data_dir = base_path / "data"
        assert data_dir.exists(), f"Data directory does not exist: {data_dir}"
        train_data_csv_path = data_dir / "train_data.csv"
        val_data_csv_path = data_dir / "val_data.csv"
        test_data_csv_path = data_dir / "test_data.csv"
        train_df = pd.read_csv(train_data_csv_path)
        val_df = pd.read_csv(val_data_csv_path)
        test_df = pd.read_csv(test_data_csv_path)
        print(f"Loaded train, val, test data from CSV files.")
    else:
        ML_data_df = create_ml_data_dataframe()
        train_df, val_df, test_df = create_train_val_test_splits(ML_data_df)
    
    train_cnn_model_on_dataframes(train_df, val_df, test_df)





if __name__ == "__main__":
    main()