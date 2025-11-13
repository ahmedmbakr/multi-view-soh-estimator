from pathlib import Path
import torch
import pandas as pd
import numpy as np
from train_cnn_on_gamry import create_ml_data_dataframe, create_train_val_test_splits, test_trained_model_on_test_set, get_nyquist_train_val_test_data, reset_seeds
from nyquist_cnn import TinyNyquistCNN, predict_soh_arrays
import nyquist_cnn

cell_type_to_model_name_dict = {"COIN_ONLY": "coin",
                                     "CYLINDRICAL_ONLY": "cyl",
                                     "ALL": "both"}
feature_to_use_to_model_name_dict = {"NYQUIST_ONLY": "Nyquist",
                                            "IMP_MAG_AND_PHASE": "MAG_PH",
                                            "ALL": "both"}
cell_name_to_plot_name_dict = {"B12": "CYL-03@15",
                               "Cell_01@25": "COIN-01@25",
                                "Cell_01@35": "COIN-01@35"}

feature_to_use_to_plot_name_dict = {"NYQUIST_ONLY": "Nyquist",
                                    "IMP_MAG_AND_PHASE": "Mag_Ph",
                                    "ALL": "Nyquist + Mag_Ph"}

def plot_true_vs_predicted_soh_for_paper():
    import matplotlib.pyplot as plt
    dir_path = Path(__file__).parent
    reset_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    fontsize = 24
    
    # Load dataframes based on cell type "ALL"
    ML_data_df = create_ml_data_dataframe(cells_to_use="ALL")
    train_df, val_df, test_df = create_train_val_test_splits(ML_data_df, style="test_one_cell_out", cells_types_to_use="ALL")
    
    # Limit to the first 250 cycles
    test_df = test_df[test_df["cycle_number"] <= 250]
    
    # Get unique cells in test set
    test_cells = test_df["battery_cell_name"].unique()
    num_cells = len(test_cells)
    
    # Feature sets to evaluate
    feature_sets = ["NYQUIST_ONLY", "IMP_MAG_AND_PHASE", "ALL"]
    num_features = len(feature_sets)
    
    # Create figure with subplots: rows = cells, columns = feature sets
    fig, axes = plt.subplots(num_cells, num_features, figsize=(6 * num_features, 6 * num_cells))
    
    # Ensure axes is 2D array even if there's only one row/column
    if num_cells == 1 and num_features == 1:
        axes = np.array([[axes]])
    elif num_cells == 1:
        axes = axes.reshape(1, -1)
    elif num_features == 1:
        axes = axes.reshape(-1, 1)
    
    # Dictionary to store predictions for each feature set
    predictions_dict = {}
    
    for col_idx, feature_to_use in enumerate(feature_sets):
        reset_seeds()
        
        train_X, train_y, val_X, val_y, test_X, test_y = get_nyquist_train_val_test_data(train_df, val_df, test_df, features_to_use=feature_to_use)
        nyquist_cnn.NUMBER_OF_DIMENSIONS = train_X.shape[2]
        print(f"  Using feature set: {feature_to_use}")
        model_name = f"cnn_gamry_model_{cell_type_to_model_name_dict['ALL']}_{feature_to_use_to_model_name_dict[feature_to_use]}.pth"
        model_path = dir_path / model_name
        assert model_path.exists(), f"Model file {model_path} does not exist."
        # Load the trained model from the pth file
        checkpoint = torch.load(model_path, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
        if state_dict is None:
            raise RuntimeError("No state_dict found in checkpoint.")
        model = TinyNyquistCNN(in_ch=train_X.shape[2])
        model.load_state_dict(state_dict)
        mean = checkpoint['mean']
        std = checkpoint['std']
        model.eval().to(device)
        predicted_y = predict_soh_arrays(model, test_X, (mean, std), batch_size=64)
        
        # Store predictions with corresponding cell names
        predictions_dict[feature_to_use] = predicted_y
        
        # Plot for each cell
        for row_idx, cell_name in enumerate(test_cells):
            ax = axes[row_idx, col_idx]
            
            # Get indices for this specific cell
            cell_indices = test_df["battery_cell_name"] == cell_name
            true_y_cell = test_df[cell_indices]["SOH_percent"].values
            predicted_y_cell = predicted_y[cell_indices.values]
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((true_y_cell - predicted_y_cell) ** 2))
            mae = np.mean(np.abs(true_y_cell - predicted_y_cell))
            mape = np.mean(np.abs((true_y_cell - predicted_y_cell) / true_y_cell)) * 100.0
            
            # Create scatter plot: True SOH vs Predicted SOH
            ax.scatter(true_y_cell, predicted_y_cell, alpha=0.6, s=50, color='blue', marker='o', label='Model Predictions')
            
            # Plot ideal prediction line (y=x)
            min_soh = min(true_y_cell.min(), predicted_y_cell.min())
            max_soh = max(true_y_cell.max(), predicted_y_cell.max())
            ax.plot([min_soh, max_soh], [min_soh, max_soh], 'g--', lw=2, label='Ideal Prediction')
            
            # Add text box with metrics
            textstr = f'RMSE: {rmse:.1f}\nMAE  : {mae:.1f}\nMAPE: {mape:.1f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=fontsize - 2,
                   verticalalignment='top', bbox=props)
            
            # Set labels - only show on outer edges
            # Show "True SOH (%)" only on the last row
            if row_idx == num_cells - 1:
                ax.set_xlabel("True SOH (%)", fontsize=fontsize)
            else:
                ax.set_xlabel("")
            
            # Show "Predicted SOH (%)" only on the first column
            if col_idx == 0:
                ax.set_ylabel("Predicted SOH (%)", fontsize=fontsize)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelleft=False)  # Hide left y-axis tick labels
            
            # Get plot name for cell
            plot_cell_name = cell_name_to_plot_name_dict.get(cell_name, cell_name)
            
            # Title: cell name for first row, feature set for top row
            if row_idx == 0:
                title = f"{feature_to_use_to_plot_name_dict[feature_to_use]}\n{plot_cell_name}"
            else:
                title = plot_cell_name
            ax.set_title(title, fontsize=fontsize + 2)
            
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=fontsize - 4)
            ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(dir_path / "true_vs_predicted_scatter_plot_for_paper.pdf", dpi=600, bbox_inches='tight')
    plt.show()
    print(f"True vs Predicted scatter plot saved to {dir_path / 'true_vs_predicted_scatter_plot_for_paper.pdf'}")

def plot_for_paper():
    import matplotlib.pyplot as plt
    dir_path = Path(__file__).parent
    reset_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    fontsize = 20
    
    for cell_type in ["ALL"]:
        reset_seeds()
        print(f"Training and evaluating model for cell type: {cell_type}")
        # Load dataframes based on cell type
        
        ML_data_df = create_ml_data_dataframe(cells_to_use=cell_type)
        train_df, val_df, test_df = create_train_val_test_splits(ML_data_df, style="test_one_cell_out", cells_types_to_use=cell_type)

        # Limit to the first 250 cycles
        test_df = test_df[test_df["cycle_number"] <= 250]
        
        # Get unique cells in test set
        test_cells = test_df["battery_cell_name"].unique()
        num_cells = len(test_cells)
        
        # Feature sets to evaluate
        feature_sets = ["NYQUIST_ONLY", "IMP_MAG_AND_PHASE", "ALL"]
        num_features = len(feature_sets)
        
        # Create figure with subplots: rows = cells, columns = feature sets
        fig, axes = plt.subplots(num_cells, num_features, figsize=(6 * num_features, 6 * num_cells))
        
        # Ensure axes is 2D array even if there's only one row/column
        if num_cells == 1 and num_features == 1:
            axes = np.array([[axes]])
        elif num_cells == 1:
            axes = axes.reshape(1, -1)
        elif num_features == 1:
            axes = axes.reshape(-1, 1)
        
        # Dictionary to store predictions for each feature set
        predictions_dict = {}
        
        for col_idx, feature_to_use in enumerate(feature_sets):
            reset_seeds()
            
            train_X, train_y, val_X, val_y, test_X, test_y = get_nyquist_train_val_test_data(train_df, val_df, test_df, features_to_use=feature_to_use)
            nyquist_cnn.NUMBER_OF_DIMENSIONS = train_X.shape[2]
            print(f"  Using feature set: {feature_to_use}")
            model_name = f"cnn_gamry_model_{cell_type_to_model_name_dict[cell_type]}_{feature_to_use_to_model_name_dict[feature_to_use]}.pth"
            model_path = dir_path / model_name
            assert model_path.exists(), f"Model file {model_path} does not exist."
            # Load the trained model from the pth file
            checkpoint = torch.load(model_path, weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
            if state_dict is None:
                raise RuntimeError("No state_dict found in checkpoint.")
            model = TinyNyquistCNN(in_ch=train_X.shape[2])
            model.load_state_dict(state_dict)
            mean = checkpoint['mean']
            std = checkpoint['std']
            model.eval().to(device)
            predicted_y = predict_soh_arrays(model, test_X, (mean, std), batch_size=64)
            
            # Store predictions with corresponding cell names
            predictions_dict[feature_to_use] = predicted_y
            
            # Plot for each cell
            for row_idx, cell_name in enumerate(test_cells):
                ax = axes[row_idx, col_idx]
                
                # Get indices for this specific cell
                cell_indices = test_df["battery_cell_name"] == cell_name
                true_y_cell = test_df[cell_indices]["SOH_percent"].values
                predicted_y_cell = predicted_y[cell_indices.values]
                
                # Calculate MAPE for each sample
                mape_cell = np.abs((true_y_cell - predicted_y_cell) / true_y_cell) * 100.0
                
                # Create line plots with markers on primary axis
                ax.plot(true_y_cell, label="True SOH", marker='o', color='green')
                ax.plot(predicted_y_cell, label="Predicted SOH", marker='x', color='blue')
                
                # Create secondary y-axis for MAPE
                ax2 = ax.twinx()
                ax2.plot(mape_cell, label="MAPE", color='red', alpha=0.5, linestyle='', marker='o', markerfacecolor='none', markeredgecolor='red', markersize=8)
                
                # Add average MAPE line
                avg_mape = np.mean(mape_cell)
                ax2.axhline(y=avg_mape, color='red', linestyle='--', linewidth=2, label=f'Avg MAPE: {avg_mape:.1f}%')
                
                ax2.set_ylim(0, 100)  # Set MAPE axis range from 0 to 100%
                
                # Set labels - only show on outer edges
                # Show "Cycle number" only on the last row
                if row_idx == num_cells - 1:
                    ax.set_xlabel("Cycle number", fontsize=fontsize)
                else:
                    ax.set_xlabel("")
                
                # Show "SOH (%)" only on the first column
                if col_idx == 0:
                    ax.set_ylabel("SOH (%)", fontsize=fontsize)
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis='y', labelleft=False)  # Hide left y-axis tick labels
                
                # Show "MAPE (%)" only on the last column
                if col_idx == num_features - 1:
                    ax2.set_ylabel("MAPE (%)", fontsize=fontsize, color='red')
                    ax2.tick_params(axis='y', labelcolor='red', labelsize=fontsize - 2)
                else:
                    ax2.set_ylabel("")
                    ax2.tick_params(axis='y', labelsize=0)  # Hide tick labels
                
                # Get plot name for cell
                plot_cell_name = cell_name_to_plot_name_dict.get(cell_name, cell_name)
                
                # Title: cell name for first row, feature set for top row
                if row_idx == 0:
                    title = f"{feature_to_use_to_plot_name_dict[feature_to_use]}\n{plot_cell_name}"
                else:
                    title = plot_cell_name
                ax.set_title(title, fontsize=fontsize + 2)
                
                ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
                ax.grid(True)
                
                # Combine legends from both axes
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                if row_idx == 2:
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop={"size": fontsize - 2})
                else:
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(0.01, 0.35), prop={"size": fontsize - 2}) # 0.01 is the x offset from the left and 0.35 is the y offset
        
        plt.tight_layout()
        plt.savefig(dir_path / "comparison_plot_for_paper.pdf", dpi=600, bbox_inches='tight')
        plt.show()
        print(f"Comparison plot saved to {dir_path / 'comparison_plot_for_paper.png'}")

def plot_nyquist_for_paper():
    import matplotlib.pyplot as plt
    dir_path = Path(__file__).parent
    reset_seeds()
    fontsize = 24
    
    # Define the cells for each row
    cells_layout = [
        ["B10", "B11", "B12"],  # Row 1: Cylindrical cells
        ["Cell_02@25", "Cell_05@25", "Cell_01@25"],  # Row 2: Coin cells at 25C
        ["Cell_02@35", "Cell_02@45", "Cell_01@35"]   # Row 3: Coin cells at other temps
    ]
    
    # Define display names for each cell
    cell_display_names = {
        "B10": "CYL-01@15",
        "B11": "CYL-02@15",
        "B12": "CYL-03@15",
        "Cell_02@25": "COIN-02@25",
        "Cell_05@25": "COIN-05@25",
        "Cell_01@25": "COIN-01@25",
        "Cell_02@35": "COIN-02@35",
        "Cell_02@45": "COIN-02@45",
        "Cell_01@35": "COIN-01@35"
    }
    
    # Load the ML data
    ML_data_df = create_ml_data_dataframe(cells_to_use="ALL")

    # Limit to the first 250 cycles
    ML_data_df = ML_data_df[ML_data_df["cycle_number"] <= 250]
    
    # Create 3x3 subplot grid with colorbar
    fig = plt.figure(figsize=(20, 12))
    
    # Create GridSpec for layout: 3 rows x 4 columns (3 plots + 1 colorbar)
    # Make the colorbar column narrower and reduce space before it
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.05], hspace=0.3, wspace=0.05)
    
    # Create axes for the 3x3 grid
    axes = []
    for row in range(3):
        row_axes = []
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)
    
    # Select 20 evenly spaced cycles for visualization
    num_cycles_to_plot = 20
    
    # Use viridis colormap
    colormap = plt.cm.viridis
    
    for row_idx, row_cells in enumerate(cells_layout):
        for col_idx, cell_name in enumerate(row_cells):
            ax = axes[row_idx][col_idx]
            
            # Filter data for this cell
            cell_data = ML_data_df[ML_data_df["battery_cell_name"] == cell_name]
            
            if cell_data.empty:
                ax.text(0.5, 0.5, f"No data for {cell_name}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(cell_display_names.get(cell_name, cell_name), fontsize=fontsize)
                continue
            
            # Get unique cycles and select evenly spaced ones
            available_cycles = sorted(cell_data["cycle_number"].unique())
            if len(available_cycles) > num_cycles_to_plot:
                cycle_indices = np.linspace(0, len(available_cycles) - 1, num_cycles_to_plot, dtype=int)
                selected_cycles = [available_cycles[i] for i in cycle_indices]
            else:
                selected_cycles = available_cycles
            
            # Plot Nyquist for each selected cycle
            colors = [colormap(i / (len(selected_cycles) - 1)) for i in range(len(selected_cycles))]
            
            for idx, cycle in enumerate(selected_cycles):
                cycle_data = cell_data[cell_data["cycle_number"] == cycle]
                
                # Extract Z_real and Z_imag
                Z_real = cycle_data["Z_real"].values
                Z_imag = cycle_data["Z_imag"].values
                
                # Plot Nyquist (Z_real vs -Z_imag)
                ax.plot(Z_real, -Z_imag, 'o-', color=colors[idx], 
                       markersize=3, linewidth=1, alpha=0.7)
            
            # Set labels
            if row_idx == 2:  # Bottom row
                ax.set_xlabel("Z' (Ω)", fontsize=fontsize)
            if col_idx == 0:  # First column
                ax.set_ylabel("-Z'' (Ω)", fontsize=fontsize)
            
            # Set title
            ax.set_title(cell_display_names.get(cell_name, cell_name), fontsize=fontsize)
            
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
            ax.grid(True, alpha=0.3)
    
    # Add colorbar in the rightmost column spanning all rows
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    cbar_ax = fig.add_subplot(gs[:, 3])  # Span all rows, last column
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Battery Age', fontsize=fontsize, rotation=270, labelpad=0)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['New', 'Aged'], fontsize=fontsize - 2)
    
    plt.savefig(dir_path / "nyquist_plot_for_paper.pdf", dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Nyquist plot saved to {dir_path / 'nyquist_plot_for_paper.pdf'}")

def plot_mag_phase_for_paper():
    import matplotlib.pyplot as plt
    dir_path = Path(__file__).parent
    reset_seeds()
    fontsize = 24
    
    # Define the cells for each row
    cells_layout = [
        ["B10", "B11", "B12"],  # Row 1: Cylindrical cells
        ["Cell_02@25", "Cell_05@25", "Cell_01@25"],  # Row 2: Coin cells at 25C
        ["Cell_02@35", "Cell_02@45", "Cell_01@35"]   # Row 3: Coin cells at other temps
    ]
    
    # Define display names for each cell
    cell_display_names = {
        "B10": "CYL-01@15",
        "B11": "CYL-02@15",
        "B12": "CYL-03@15",
        "Cell_02@25": "COIN-02@25",
        "Cell_05@25": "COIN-05@25",
        "Cell_01@25": "COIN-01@25",
        "Cell_02@35": "COIN-02@35",
        "Cell_02@45": "COIN-02@45",
        "Cell_01@35": "COIN-01@35"
    }
    
    # Load the ML data
    ML_data_df = create_ml_data_dataframe(cells_to_use="ALL")

    # Limit to the first 250 cycles
    ML_data_df = ML_data_df[ML_data_df["cycle_number"] <= 250]
    
    # Create 3x3 subplot grid with colorbar
    fig = plt.figure(figsize=(20, 12))
    
    # Create GridSpec for layout: 3 rows x 4 columns (3 plots + 1 colorbar)
    # Make the colorbar column narrower and reduce space before it
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.05], hspace=0.3, wspace=0.05)
    
    # Create axes for the 3x3 grid
    axes = []
    for row in range(3):
        row_axes = []
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            row_axes.append(ax)
        axes.append(row_axes)
    
    # Select 20 evenly spaced cycles for visualization
    num_cycles_to_plot = 20
    
    # Use viridis colormap
    colormap = plt.cm.viridis
    
    for row_idx, row_cells in enumerate(cells_layout):
        for col_idx, cell_name in enumerate(row_cells):
            ax = axes[row_idx][col_idx]
            
            # Filter data for this cell
            cell_data = ML_data_df[ML_data_df["battery_cell_name"] == cell_name]
            
            if cell_data.empty:
                ax.text(0.5, 0.5, f"No data for {cell_name}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(cell_display_names.get(cell_name, cell_name), fontsize=fontsize)
                continue
            
            # Get unique cycles and select evenly spaced ones
            available_cycles = sorted(cell_data["cycle_number"].unique())
            if len(available_cycles) > num_cycles_to_plot:
                cycle_indices = np.linspace(0, len(available_cycles) - 1, num_cycles_to_plot, dtype=int)
                selected_cycles = [available_cycles[i] for i in cycle_indices]
            else:
                selected_cycles = available_cycles
            
            # Plot Magnitude vs Phase for each selected cycle
            colors = [colormap(i / (len(selected_cycles) - 1)) for i in range(len(selected_cycles))]
            
            for idx, cycle in enumerate(selected_cycles):
                cycle_data = cell_data[cell_data["cycle_number"] == cycle]
                
                # Extract impedance magnitude and phase
                impedance_mag = cycle_data["impedance_magnitude_Ohms"].values
                phase = cycle_data["phase_deg"].values
                
                # Plot Magnitude vs Phase
                ax.plot(impedance_mag, phase, 'o-', color=colors[idx], 
                       markersize=3, linewidth=1, alpha=0.7)
            
            # Set labels
            if row_idx == 2:  # Bottom row
                ax.set_xlabel("Magnitude (Ω)", fontsize=fontsize)
            if col_idx == 0:  # First column
                ax.set_ylabel("Phase (°)", fontsize=fontsize)
            
            # Set title
            ax.set_title(cell_display_names.get(cell_name, cell_name), fontsize=fontsize)
            
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
            ax.grid(True, alpha=0.3)
    
    # Add colorbar in the rightmost column spanning all rows
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    
    cbar_ax = fig.add_subplot(gs[:, 3])  # Span all rows, last column
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Battery Age', fontsize=fontsize, rotation=270, labelpad=0)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['New', 'Aged'], fontsize=fontsize - 2)
    
    plt.savefig(dir_path / "mag_phase_plot_for_paper.pdf", dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Magnitude-Phase plot saved to {dir_path / 'mag_phase_plot_for_paper.pdf'}")

def print_num_parameters_in_cnn():
    model = TinyNyquistCNN(in_ch=4)  # Example with 2 input channels
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters in TinyNyquistCNN: {total_params}")

if __name__ == "__main__":
    print_num_parameters_in_cnn()
    # plot_true_vs_predicted_soh_for_paper()
    plot_for_paper()
    # plot_nyquist_for_paper()
    # plot_mag_phase_for_paper()