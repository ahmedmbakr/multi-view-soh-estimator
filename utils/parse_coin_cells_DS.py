import pandas as pd
from pathlib import Path
import numpy as np
from typing import Tuple

def main():
    import os
    dir_path = Path(__file__).parent
    dataset_base_path = dir_path / "../data/DataForRapidEstimation"
    capacity_dir_path = dataset_base_path / "CapacityData"
    eis_data_dir_path = dataset_base_path / "EIS_Data"
    preprocessed_dir_path = dataset_base_path / "preprocessed"

    assert capacity_dir_path.exists(), f"Capacity data directory does not exist: {capacity_dir_path}"
    assert eis_data_dir_path.exists(), f"EIS data directory does not exist: {eis_data_dir_path}"

    # Loop through each cell capacity and corresponding EIS data files
    for temperature in ["25", "35", "45"]:
        for cell_id in range(1, 9):
            capacity_file = capacity_dir_path / f"Data_Capacity_{temperature}C0{cell_id}.txt"
            eis_file = eis_data_dir_path / f"EIS_state_I_{temperature}C0{cell_id}.txt" 

            if not capacity_file.exists() or not eis_file.exists():
                print(f"Skipping Cell 0{cell_id} at {temperature}C due to missing files.")
                continue

            # assert capacity_file.exists(), f"Capacity file does not exist: {capacity_file}"
            # assert eis_file.exists(), f"EIS data file does not exist: {eis_file}"

            # Load capacity data
            capacity_df = pd.read_csv(capacity_file, delim_whitespace=True)
            # print number of columns and their names
            print(f"Capacity Data Columns for Cell 0{cell_id}: {capacity_df.columns.tolist()}")
            capacity_df = capacity_df[["cycle_number", "ox/red", "Capacity/mA.h"]]
            capacity_df.columns = ["cycle_number", "ox_red", "capacity_Ah"]
            capacity_df.astype({
                "cycle_number": int,
                "ox_red": int,
                "capacity_Ah": float
            })


            # Group by cycle number and take the maximum capacity for each cycle when ox_red == 0 (which indicates discharge)
            capacity_df = capacity_df[capacity_df["ox_red"] == 0]
            capacity_df = capacity_df.groupby("cycle_number", as_index=False)["capacity_Ah"].max()
            capacity_df = capacity_df.sort_values(by="cycle_number").reset_index(drop=True)
            capacity_df["SOH_percent"] = (capacity_df["capacity_Ah"] / capacity_df["capacity_Ah"].iloc[0]) * 100.0

            # Print the first few rows of capacity data for verification
            print(f"Capacity Data for Cell 0{cell_id}:")
            print(capacity_df.head())

            # Load EIS data
            eis_df = pd.read_csv(eis_file, delim_whitespace=True, skiprows=1)

            eis_df = eis_df.iloc[:, [1, 2, 3, 4, 5, 6]]
            eis_df.columns = [
                "cycle_number", "frequency_Hz", "Z_real_Ohms", "Z_imag_Ohms", "impedance_magnitude_Ohms", "phase_deg"
            ]

            eis_df.astype({
                "cycle_number": int,
                "frequency_Hz": float,
                "Z_real_Ohms": float,
                "Z_imag_Ohms": float,
                "impedance_magnitude_Ohms": float,
                "phase_deg": float
            })

            # Print the first few rows of EIS data for verification
            print(f"EIS Data for Cell 0{cell_id}@{temperature}:")
            print(eis_df.head())

            merged_df = merge_impedance_with_capacity(eis_df, capacity_df)
            
            from parse_gamry_output import plot_impedance_magnitude_vs_phase_for_battery_cell, plot_nyquist_for_battery_cell

            plot_nyquist_for_battery_cell(
                merged_df,
                battery_cell_name=f"Cell_0{cell_id}@{temperature}",
                cycle_stride=10
            )

            plot_impedance_magnitude_vs_phase_for_battery_cell(
                merged_df,
                battery_cell_name=f"Cell_0{cell_id}@{temperature}",
                cycle_stride=10
            )

            merged_df.to_csv(
                preprocessed_dir_path / f"merged_coin_data_{temperature}C0{cell_id}.csv",
                index=False
            )

            print(f"Saved merged data for Cell 0{cell_id}@{temperature} to preprocessed directory with the name: merged_coin_data_{temperature}C0{cell_id}.csv")

def plot_soh_vs_cycle_number_for_batteries():
    import matplotlib.pyplot as plt

    dir_path = Path(__file__).parent
    preprocessed_dir_path = dir_path / "../data/DataForRapidEstimation/preprocessed"

    batteries_csv_files = [ preprocessed_dir_path / "merged_coin_data_25C01.csv",
                            preprocessed_dir_path / "merged_coin_data_25C02.csv",
                            preprocessed_dir_path / "merged_coin_data_25C03.csv",
                            preprocessed_dir_path / "merged_coin_data_25C05.csv",
                            preprocessed_dir_path / "merged_coin_data_35C01.csv",
                            preprocessed_dir_path / "merged_coin_data_35C02.csv",
                            preprocessed_dir_path / "merged_coin_data_45C01.csv",
                            preprocessed_dir_path / "merged_coin_data_45C02.csv"]

    # for cell_id in range(1, 9):
    #     merged_file = preprocessed_dir_path / f"merged_coin_data_{cell_id}.csv"
    #     assert merged_file.exists(), f"Merged data file does not exist: {merged_file}"
    #     batteries_csv_files.append(merged_file)
    
    from parse_gamry_output import analyze_battery_cells

    analyze_battery_cells(
        batteries_csv_files,
        capacity_column_name="capacity_Ah",
        cycle_to_plot_nyquist=1
    )


        

def merge_impedance_with_capacity(eis_df: pd.DataFrame, capacity_df: pd.DataFrame) -> pd.DataFrame:
    # Merge EIS data with capacity data on cycle_number
    merged_df = pd.merge(eis_df, capacity_df, on="cycle_number", how="inner")

    merged_df = merged_df.sort_values(by=["cycle_number", "frequency_Hz"], ascending=[True, False]).reset_index(drop=True)

    print("Merged Data Sample:")
    print(merged_df.head())

    return merged_df


if __name__ == "__main__":
    main()
    plot_soh_vs_cycle_number_for_batteries()