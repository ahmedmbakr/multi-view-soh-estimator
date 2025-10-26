import pandas as pd
from pathlib import Path
import numpy as np

def parse_gamry_impedance_data_output_file(csv_file_path: Path, number_to_add_to_cycle: int = 0) -> pd.DataFrame:
    """Parses a Gamry impedance data output CSV file into a pandas DataFrame.
    Impedance file's name should start with `4_` and the file should be in CSV format.
    Here are the columns of interest (0-indexed):
    - Column 1: Cycle number
    - Column 4: DC Working Electrode (V)
    - Column 5: DC Current (A)
    - Column 6: Frequency (Hz)
    - Column 7: |Z| (Ohms)
    - Column 8: Phase (deg)
    - Column 12: Amplitude (A)
    - Column 13: Amplitude (V)

    Args:
        - csv_file_path (Path): Path to the Gamry impedance data output CSV file.
        - number_to_add_to_cycle (int, optional): Number to add to each cycle number. Defaults to 0.
    Returns:
        - pd.DataFrame: DataFrame containing the parsed impedance data.
    """
    # Make sure the file name starts with '4_'
    if not csv_file_path.name.startswith("4_"):
        raise ValueError("The file name must start with '4_' to be a valid Gamry impedance data output file.")
    # Make sure the file is a CSV file and that it exists
    if not csv_file_path.suffix.lower() == ".csv" or not csv_file_path.exists():
        raise FileNotFoundError(f"The file {csv_file_path} does not exist or is not a CSV file.")
    
    # Read the CSV file into a DataFrame. The first row of the CSV file is skipped as it contains headers.
    df = pd.read_csv(csv_file_path, skiprows=1, header=None)
    # Select relevant columns and rename them
    df = df.iloc[:, [1, 4, 5, 6, 7, 8, 12, 13]]
    df.columns = [
        "cycle_number",
        "dc_working_electrode_V",
        "dc_current_A",
        "frequency_Hz",
        "impedance_magnitude_Ohms",
        "phase_deg",
        "amplitude_A",
        "amplitude_V",
    ]
    # Convert columns to appropriate data types
    df = df.astype({
        "cycle_number": int,
        "dc_working_electrode_V": float,
        "dc_current_A": float,
        "frequency_Hz": float,
        "impedance_magnitude_Ohms": float,
        "phase_deg": float,
        "amplitude_A": float,
        "amplitude_V": float,
    })

    # Adjust cycle numbers if needed
    if number_to_add_to_cycle != 0:
        df["cycle_number"] += number_to_add_to_cycle

    return df

def __test_parse_gamry_impedance_data_output_file():
    # Test the function with a sample file path
    file_path = Path("C:/Users/AhmedBakr/Box/INR18650Cycling/15C_Cycling/B10/INR18650MJ1_Main_Cycle29_B10/4_INR18650MJ1_Main 20250410 003339.csv")
    impedance_data_df = parse_gamry_impedance_data_output_file(file_path, number_to_add_to_cycle=0)
    print(impedance_data_df.head())

def parse_gamry_cycling_data_output_file(csv_file_path: Path, number_to_add_to_cycle: int=0) -> pd.DataFrame:
    """Parses a Gamry cycling data output CSV file into a pandas DataFrame.
    Cycling file's name should start with `2_` and the file should be in CSV format.
    Here are the columns of interest (0-indexed):
    - Column 1: Cycle number
    - Column 3: Elapsed Time (s)
    - Column 4: Working Electrode (V)
    - Column 7: Absolute discharge current (A)
    - Column 9: cumulative charge (mAh)
    - Column 11: Temperature (C)
    - Column 12: State of Charge (%)
    - Column 13: Depth of Discharge (%)
    - Column 14: dQ/dV (mAh/V)
    - Column 15: Energy (mWh)

    Args:
        - csv_file_path (Path): Path to the Gamry cycling data output CSV file.
        - number_to_add_to_cycle (int, optional): Number to add to each cycle number. Defaults to 0.
    Returns:
        - pd.DataFrame: DataFrame containing the parsed cycling data.
    """
    # Make sure the file name starts with '2_'
    if not csv_file_path.name.startswith("2_"):
        raise ValueError("The file name must start with '2_' to be a valid Gamry cycling data output file.")
    # Make sure the file is a CSV file and that it exists
    if not csv_file_path.suffix.lower() == ".csv" or not csv_file_path.exists():
        raise FileNotFoundError(f"The file {csv_file_path} does not exist or is not a CSV file.")
    
    # Read the CSV file into a DataFrame. The first row of the CSV file is skipped as it contains headers.
    df = pd.read_csv(csv_file_path, skiprows=1, header=None)
    # Select relevant columns and rename them
    df = df.iloc[:, [1, 3, 4, 7, 9, 11, 12, 13, 14, 15]]
    df.columns = [
        "cycle_number",
        "elapsed_time_s",
        "working_electrode_V",
        "absolute_discharge_current_A",
        "cumulative_charge_mAh",
        "temperature_C",
        "state_of_charge_percent",
        "depth_of_discharge_percent",
        "dQ_dV_mAh_per_V",
        "energy_mWh",
    ]
    # Convert columns to appropriate data types
    df = df.astype({
        "cycle_number": int,
        "elapsed_time_s": float,
        "working_electrode_V": float,
        "absolute_discharge_current_A": float,
        "cumulative_charge_mAh": float,
        "temperature_C": float,
        "state_of_charge_percent": float,
        "depth_of_discharge_percent": float,
        "dQ_dV_mAh_per_V": float,
        "energy_mWh": float,
    })

    # Adjust cycle numbers if needed
    if number_to_add_to_cycle != 0:
        df["cycle_number"] += number_to_add_to_cycle

    return df

def __test_parse_gamry_cycling_data_output_file():
    # Test the function with a sample file path
    file_path = Path("C:/Users/AhmedBakr/Box/INR18650Cycling/15C_Cycling/B10/INR18650MJ1_Main_Cycle29_B10/2_INR18650MJ1_Main 20250409 202947.csv")
    cycling_data_df = parse_gamry_cycling_data_output_file(file_path)
    print(cycling_data_df.head())

def merge_impedance_and_cycling_data(
    impedance_df: pd.DataFrame,
    cycling_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges impedance and cycling data DataFrames. We need only one row per cycle number from cycling data.
    The cycle data for each cycle number calculates the following columns: avg_working_electrode_V, avg_temperature_C, total_cumulative_charge_mAh, total_energy_mWh.
    The merged DataFrame will contain all impedance data rows with the corresponding cycling data for each cycle number.

    Args:
        - impedance_df (pd.DataFrame): DataFrame containing impedance data.
        - cycling_df (pd.DataFrame): DataFrame containing cycling data.
    Returns:
        - pd.DataFrame: Merged DataFrame containing both impedance and cycling data.
    """
    # Aggregate cycling data to get one row per cycle number
    cycling_agg_df = cycling_df.groupby("cycle_number").agg({
        "working_electrode_V": "mean",
        "temperature_C": "mean",
        "cumulative_charge_mAh": "max",
        "energy_mWh": "max",
    }).reset_index()
    cycling_agg_df.rename(columns={
        "working_electrode_V": "avg_working_electrode_V",
        "temperature_C": "avg_temperature_C",
        "cumulative_charge_mAh": "total_cumulative_charge_mAh",
        "energy_mWh": "total_energy_mWh",
    }, inplace=True)

    # Merge impedance data with aggregated cycling data on cycle_number
    merged_df = pd.merge(impedance_df, cycling_agg_df, on="cycle_number", how="left")

    return merged_df


def __test_merge_impedance_and_cycling_data():
    # Test the function with sample data
    file_path = Path("C:/Users/AhmedBakr/Box/INR18650Cycling/15C_Cycling/B10/INR18650MJ1_Main_Cycle29_B10/4_INR18650MJ1_Main 20250410 003339.csv")
    impedance_df = parse_gamry_impedance_data_output_file(file_path, number_to_add_to_cycle=0)
    file_path = Path("C:/Users/AhmedBakr/Box/INR18650Cycling/15C_Cycling/B10/INR18650MJ1_Main_Cycle29_B10/2_INR18650MJ1_Main 20250409 202947.csv")
    cycling_df = parse_gamry_cycling_data_output_file(file_path, number_to_add_to_cycle=0)

    merged_df = merge_impedance_and_cycling_data(impedance_df, cycling_df)
    print(merged_df.head())

def parse_and_merge_gamry_data_for_battery_cell(
    main_cycling_data_folder: Path,
    battery_cell_name: str,
    main_cycling_data_folder_name_regex_pattern: str,
    subtract_number_from_cycle_number_offset: int = 0
) -> pd.DataFrame:
    """Parses and merges Gamry impedance and cycling data for a specific battery cell.

    Args:
        - main_cycling_data_folder (Path): Path to the main cycling data folder.
    """
    battery_cell_folder = main_cycling_data_folder / battery_cell_name
    if not battery_cell_folder.exists():
        raise FileNotFoundError(f"The battery cell folder {battery_cell_folder} does not exist.")
    print("Parsing data for battery cell folder:", battery_cell_folder)
    all_impedance_dfs = []
    all_cycling_dfs = []

    for main_cycle_folder in battery_cell_folder.iterdir():
        if main_cycle_folder.is_dir():
            # Extract cycle number from folder name using regex
            import re
            match = re.match(main_cycling_data_folder_name_regex_pattern, main_cycle_folder.name)
            if match:
                cycle_number_offset = int(match.group(1)) - subtract_number_from_cycle_number_offset
            else:
                print(f"Folder name {main_cycle_folder.name} is skipped as it does not match the expected pattern.")
                continue  # Skip folders that do not match the pattern

            # Find impedance and cycling files
            impedance_files = list(main_cycle_folder.glob("4_*.csv"))
            cycling_files = list(main_cycle_folder.glob("2_*.csv"))

            assert len(impedance_files) > 0, f"No impedance files found in {main_cycle_folder}"
            assert len(cycling_files) > 0, f"No cycling files found in {main_cycle_folder}"

            assert len(impedance_files) == len(cycling_files) == 1, f"Number of impedance files and cycling files do not match in {main_cycle_folder} or are not equal to 1."

            for imp_file in impedance_files:
                imp_df = parse_gamry_impedance_data_output_file(imp_file, number_to_add_to_cycle=cycle_number_offset)
                all_impedance_dfs.append(imp_df)

            for cyc_file in cycling_files:
                cyc_df = parse_gamry_cycling_data_output_file(cyc_file, number_to_add_to_cycle=cycle_number_offset)
                all_cycling_dfs.append(cyc_df)
            print("Parsed data from folder:", main_cycle_folder.name)

    # Concatenate all dataframes
    full_impedance_df = pd.concat(all_impedance_dfs, ignore_index=True)
    full_cycling_df = pd.concat(all_cycling_dfs, ignore_index=True)

    # Merge impedance and cycling data
    merged_df = merge_impedance_and_cycling_data(full_impedance_df, full_cycling_df)
    
    # Sort merged_df by cycle_number and frequency_Hz. Ascending order for cycle_number and descending order for frequency_Hz
    merged_df = merged_df.sort_values(by=["cycle_number", "frequency_Hz"], ascending=[True, False]).reset_index(drop=True)
    print("Completed parsing and merging data for battery cell:", battery_cell_name)

    return merged_df

def __test_parse_and_merge_gamry_data_for_battery_cell():
    main_cycling_data_folder = Path("C:/Users/AhmedBakr/Box/INR18650Cycling/15C_Cycling")
    battery_cell_name = "B12"
    main_cycling_data_folder_name_regex_pattern = r".*_Main_Cycle(\d+)_.*"
    output_csv_file_path = Path(f"C:/Users/AhmedBakr/Box/INR18650Cycling/15C_Cycling/{battery_cell_name}/merged_gamry_data_{battery_cell_name}.csv")

    import time
    start_time = time.time()
    merged_df = parse_and_merge_gamry_data_for_battery_cell(
        main_cycling_data_folder,
        battery_cell_name,
        main_cycling_data_folder_name_regex_pattern,
        subtract_number_from_cycle_number_offset=29
    )
    end_time = time.time()
    print(f"Time taken to parse and merge data: {end_time - start_time:.2f} seconds")
    print(f"Number of rows in merged DataFrame: {len(merged_df)}")
    print(merged_df.head())
    # Write merged_df to a CSV file
    merged_df.to_csv(output_csv_file_path, index=False)

def plot_nyquist_for_battery_cell(merged_df: pd.DataFrame, battery_cell_name: str, cycle_stride: int = 0):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    cycle_numbers = merged_df["cycle_number"].unique()
    min_cycle_number = cycle_numbers.min()
    max_cycle_number = cycle_numbers.max()
    if cycle_stride != 0:
        # Apply stride to cycle numbers
        cycle_numbers = cycle_numbers[::cycle_stride]
    for cycle_number in cycle_numbers:
        cycle_data = merged_df[merged_df["cycle_number"] == cycle_number]
        Z_real = cycle_data["impedance_magnitude_Ohms"] * np.cos(np.radians(cycle_data["phase_deg"]))
        Z_imag = cycle_data["impedance_magnitude_Ohms"] * np.sin(np.radians(cycle_data["phase_deg"]))
        plt.plot(Z_real, -Z_imag, label=f'Cycle {cycle_number}')

    plt.title(f'Nyquist Plot for Battery Cell {battery_cell_name}')
    plt.xlabel('Z\' (Ohms)')
    plt.ylabel('-Z\'\' (Ohms)')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

def __test_plot_nyquist_for_battery_cell():
    main_cycling_data_folder = Path("C:/Users/AhmedBakr/Box/INR18650Cycling/15C_Cycling")
    battery_cell_name = "B10"
    main_cycling_data_folder_name_regex_pattern = r".*_Main_Cycle(\d+)_.*"
    cycle_stride = 10 # The stride between cycles to plot after the first one

    merged_df = parse_and_merge_gamry_data_for_battery_cell(
        main_cycling_data_folder,
        battery_cell_name,
        main_cycling_data_folder_name_regex_pattern,
        subtract_number_from_cycle_number_offset=29
    )
    plot_nyquist_for_battery_cell(merged_df, battery_cell_name, cycle_stride=cycle_stride)



if __name__ == "__main__":
    # __test_parse_gamry_impedance_data_output_file()
    # __test_parse_gamry_cycling_data_output_file()
    # __test_merge_impedance_and_cycling_data()
    # __test_parse_and_merge_gamry_data_for_battery_cell()
    __test_plot_nyquist_for_battery_cell()
