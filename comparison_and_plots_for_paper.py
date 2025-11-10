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
cell_name_to_plot_name_dict = {"B12": "CYL-B12@15",
                               "Cell_01@25": "COIN-01@25",
                                "Cell_01@35": "COIN-01@35"}

feature_to_use_to_plot_name_dict = {"NYQUIST_ONLY": "Nyquist",
                                    "IMP_MAG_AND_PHASE": "Mag_Ph",
                                    "ALL": "Nyquist + Mag_Ph"}

def comparison_for_paper():
    dir_path = Path(__file__).parent
    save_results_csv_path = dir_path / "comparison_table_results_for_paper.csv"
    results_df = pd.DataFrame()
    reset_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    for cell_type in ["COIN_ONLY", "CYLINDRICAL_ONLY", "ALL"]:
        reset_seeds()
        print(f"Training and evaluating model for cell type: {cell_type}")
        # Load dataframes based on cell type
        
        ML_data_df = create_ml_data_dataframe(cells_to_use=cell_type)
        train_df, val_df, test_df = create_train_val_test_splits(ML_data_df, style="test_one_cell_out", cells_types_to_use=cell_type) # Style can be "random" or "test_one_cell_out"
        for feature_to_use in ["NYQUIST_ONLY", "IMP_MAG_AND_PHASE", "ALL"]:
            reset_seeds()
            
            train_X, train_y, val_X, val_y, test_X, test_y = get_nyquist_train_val_test_data(train_df, val_df, test_df, features_to_use=feature_to_use)
            nyquist_cnn.NUMBER_OF_DIMENSIONS = train_X.shape[2]
            print(f"  Using feature set: {feature_to_use}")
            model_name = f"cnn_gamry_model_{cell_type_to_model_name_dict[cell_type]}_{feature_to_use_to_model_name_dict[feature_to_use]}.pth"
            model_path = dir_path / model_name
            assert model_path.exists(), f"Model file {model_path} does not exist."
            # Load the trained model from the pth file
            checkpoint = torch.load(model_path, weights_only=False)
            # model = checkpoint['model_state_dict']
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
            if state_dict is None:
                raise RuntimeError("No state_dict found in checkpoint.")
            model = TinyNyquistCNN(in_ch=train_X.shape[2])
            model.load_state_dict(state_dict)
            mean = checkpoint['mean']
            std = checkpoint['std']
            model.eval().to(device)
            rmse, mae, mape = test_trained_model_on_test_set(model, test_X, test_y, (mean, std))
            results_df = pd.concat([results_df, pd.DataFrame({
                "Cell_Type": [cell_type],
                "Feature_Set": [feature_to_use],
                "Test_RMSE": [rmse],
                "Test_MAE": [mae],
                "Test_MAPE": [mape]
            })], ignore_index=True)
            print(f"    Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")
    results_df.to_csv(save_results_csv_path, index=False)
    print(f"Comparison results saved to {save_results_csv_path}")

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
                ax.plot(true_y_cell, label="True SOH", marker='o')
                ax.plot(predicted_y_cell, label="Predicted SOH", marker='x')
                
                # Create secondary y-axis for MAPE
                ax2 = ax.twinx()
                ax2.plot(mape_cell, label="MAPE", color='red', alpha=0.5, linestyle='', marker='o', markerfacecolor='none', markeredgecolor='red', markersize=8)
                
                # Add average MAPE line
                avg_mape = np.mean(mape_cell)
                ax2.axhline(y=avg_mape, color='red', linestyle='--', linewidth=2, label=f'Avg MAPE: {avg_mape:.0f}%')
                
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
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop={"size": fontsize - 2})
        
        plt.tight_layout()
        plt.savefig(dir_path / "comparison_plot_for_paper.png", dpi=600, bbox_inches='tight')
        plt.show()
        print(f"Comparison plot saved to {dir_path / 'comparison_plot_for_paper.png'}")

if __name__ == "__main__":
    # comparison_for_paper()
    plot_for_paper()