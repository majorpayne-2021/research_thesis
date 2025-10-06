# -------------------------------------------------------------------------
# Functions for Binary Classification of Transactions
# -------------------------------------------------------------------------

# Data cleaning and manipulation
import pandas as pd
import numpy as np
import math

# Machine Learning
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score, precision_score,accuracy_score, f1_score
from sklearn.impute import SimpleImputer

# Visualisation Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
class model_preprocessing:
# -------------------------------------------------------------------------

    def __init__(self):
        """
        This class holds different functions to process and clean data to create a feature table for a classification model.
        The default df is the dataframe assigned to the class, else, the input_df argument. 
        """
        
        self.df = pd.DataFrame()

    def handle_missing_values(self, input_df, ignore_cols=None):
        """
        Handle missing values in the dataset by imputing with the median of each column.
        Prints out columns that had missing values before and after imputation for cross-checking.

        Parameters:
            input_df (pd.DataFrame): The full dataset before splitting.
            ignore_cols (list): List of columns to exclude from imputation.

        Returns:
            dataset_imputed (pd.DataFrame): The dataset with missing values imputed.
        """
        if ignore_cols is None:
            ignore_cols = []

        imputer = SimpleImputer(strategy='median')

        # Create a copy to avoid modifying the original dataset
        dataset_imputed = input_df.copy()

        # Identify columns to impute (exclude specified ignored columns)
        feature_cols = input_df.drop(columns=ignore_cols, errors='ignore').select_dtypes(include=['number']).columns

        # Find columns that had missing values before imputation
        missing_cols_before = input_df[feature_cols].isna().sum()
        missing_cols_before = missing_cols_before[missing_cols_before > 0]  # Keep only columns with missing values

        # Print summary of missing values before imputation
        if not missing_cols_before.empty:
            print("\n===== Missing Value Summary Before Imputation =====")
            print(missing_cols_before.to_string())  # Print the count of missing values per column
            print(f"\nTotal columns with missing values: {len(missing_cols_before)}")
        else:
            print("\nNo missing values detected in feature columns before imputation.")

        # Apply imputation only to numeric feature columns
        dataset_imputed[feature_cols] = imputer.fit_transform(input_df[feature_cols])

        # Find columns that still have missing values after imputation (should be empty)
        missing_cols_after = dataset_imputed[feature_cols].isna().sum()
        missing_cols_after = missing_cols_after[missing_cols_after > 0]  # Keep only columns with remaining missing values

        # Print summary of missing values after imputation
        if not missing_cols_after.empty:
            print("\nWARNING: Missing Values Detected After Imputation!")
            print(missing_cols_after.to_string())  # Print the count of remaining missing values per column
            print(f"\nTotal columns still with missing values: {len(missing_cols_after)}")
        else:
            print("\nAll missing values successfully imputed.")

        return dataset_imputed
    
    def split_training_test_data(self, input_df):
        """
        Preprocess the input dataset by encoding the class labels and filtering relevant rows.
        Split the data into training/validation and test sets, ensuring no overlap.
        The test set contains all observations with 'Unknown' class and NaN values,
        while the training/validation set contains only 'Licit' (0) and 'Illicit' (1) classes.
        Also prints summary statistics for validation, including counts of rows with missing values.

        Parameters:
            input_df (pd.DataFrame): The input dataset containing transaction features and labels.

        Returns:
            X (pd.DataFrame): Features for training/validation.
            y (pd.Series): Encoded target labels (0 for Licit, 1 for Illicit).
            X_test (pd.DataFrame): Features for testing (including NaN and Unknown class).
            y_test (pd.Series): Test labels with Unknown as NaN.
            time_step_test (pd.Series): Time step information for the test set.
        """
        # Encode 'class_label': Licit = 0, Illicit = 1, Unknown = NaN
        input_df['class'] = input_df['class_label'].map({'Licit': 0, 'Illicit': 1, 'Unknown': None})

        # Store the total dataset shape before splitting
        total_dataset_shape = input_df.shape[0]

        # Split the dataset into training/validation (Licit and Illicit only) and test (Unknown + NaN)
        train_val_dataset = input_df[input_df['class'].isin([0, 1])].dropna(subset=['class'])
        test_dataset = input_df[~input_df.index.isin(train_val_dataset.index)]  # Everything else

        # Extract training and validation features and target
        X = train_val_dataset.drop(columns=['txId', 'Time step', 'class_label', 'class'])
        y = train_val_dataset['class']

        # Extract test features and target (including NaNs)
        X_test = test_dataset.drop(columns=['txId', 'Time step', 'class_label', 'class'])
        y_test = test_dataset['class']
        time_step_test = test_dataset['Time step']

        # Count rows with missing values in each set
        train_missing_rows = X[X.isna().any(axis=1)].shape[0]
        test_missing_rows = X_test[X_test.isna().any(axis=1)].shape[0]

        # Calculate percentages
        train_val_percentage = (X.shape[0] / total_dataset_shape) * 100
        test_percentage = (X_test.shape[0] / total_dataset_shape) * 100
        class_counts = y.value_counts(dropna=False)
        class_percentages = (class_counts / class_counts.sum()) * 100
        test_counts = y_test.value_counts(dropna=False)
        test_percentages = (test_counts / test_counts.sum()) * 100

        # Print summary statistics
        print("\n===== Dataset Summary =====")
        print(f"Total Dataset Size: {total_dataset_shape:,}")

        print(f"\nTraining/Validation Set: {X.shape[0]:,} ({train_val_percentage:.1f}%)")
        for cls, count in class_counts.items():
            print(f"{cls:.4f}    {count:,} ({class_percentages[cls]:.1f}%)")
        print(f"Total Training/Validation Size: {X.shape[0]:,}")
        print(f"Rows with Missing Values in Training/Validation: {train_missing_rows}")

        print(f"\nTest Set: {X_test.shape[0]:,} ({test_percentage:.1f}%)")
        for cls, count in test_counts.items():
            print(f"{cls}    {count:,} ({test_percentages[cls]:.1f}%)")
        print(f"Total Test Size: {X_test.shape[0]:,}")
        print(f"Rows with Missing Values in Test Set: {test_missing_rows}")

        # Sanity check: Ensure no data leakage (should add up to total dataset size)
        assert total_dataset_shape == (X.shape[0] + X_test.shape[0]), \
            "Mismatch in dataset split: Training/Val + Test does not add up to the total dataset!"

        return X, y, X_test, y_test, time_step_test

    def label_train_val_test(self, input_df, train_val_indices, test_indices):
        """
        Labels the dataset with 'Train/Val' or 'Test' based on index mappings.
        
        Parameters:
            df (pd.DataFrame): The dataset to be labeled.
            train_val_indices (pd.Index): Indices for training and validation data.
            test_indices (pd.Index): Indices for test data.
        
        Returns:
            pd.DataFrame: Dataset with a new column 'data_split' labeled as 'Train/Val' or 'Test'.
        """
        df = input_df.copy()
        df['data_split'] = 'Unlabeled'
        df.loc[train_val_indices, 'data_split'] = 'Train/Val'
        df.loc[test_indices, 'data_split'] = 'Test'
        
        return df

    def optimal_bins_scott(self, input_df, field_to_bin):
        """
        Determines the optimal number of bins for a histogram using Scott's Rule.
        
        Parameters:
            input_df: DataFrame containing the model output.
            field_to_bin: Column in input_df that contains the prediction probability values.

        Returns:
            num_bins: Suggested number of bins (or default 10 if calculation fails).
            bin_width: The calculated optimal bin width (or default NaN if invalid).
        """
        data = input_df[field_to_bin].dropna()  # Drop NaN values

        # Compute standard deviation
        sigma = np.std(data)
        n = len(data)  # Total number of observations

        # Prevent division by zero if standard deviation is too small or data is empty
        if sigma == 0 or n < 2:
            print("Warning: Standard deviation is zero or too few data points. Defaulting to 10 bins.")
            return 10, np.nan  # Default bins and NaN bin width

        bin_width = 3.49 * sigma / (n ** (1/3))  # Scott's Rule

        # Prevent bin_width from being zero or negative
        if bin_width <= 0:
            print("Warning: Bin width is non-positive. Defaulting to 10 bins.")
            return 10, np.nan

        bin_width = round(bin_width, 3)  # Round to 3 decimal places
        range_data = np.max(data) - np.min(data)  # Data range

        # Prevent division by zero in num_bins calculation
        num_bins = max(1, int(range_data / bin_width)) if bin_width > 0 else 10

        return num_bins, bin_width

    def optimal_bins_freedman(self, input_df, field_to_bin):
        """
        Determines the optimal number of bins for a histogram using the Freedman-Diaconis Rule.
        
        Parameters:
            input_df: DataFrame containing the model output.
            field_to_bin: Column in input_df that contains the prediction probability values.

        Returns:
            num_bins: Recommended number of bins (or default 10 if calculation fails).
            bin_width: Computed optimal bin width (or default value if invalid).
        """
        data = input_df[field_to_bin].dropna()  # Drop NaN values

        # Compute the 1st and 3rd quartiles
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25  # Interquartile range (middle 50% of the data)

        # Prevent division by zero if IQR is too small
        if iqr == 0:
            print("Warning: IQR is zero. Defaulting to 10 bins.")
            return 10, np.nan  # Return default bins and NaN bin width

        bin_width = 2 * iqr * (len(data) ** (-1/3))  # Freedman-Diaconis Rule

        # Prevent bin_width from being too small
        if bin_width <= 0:
            print("Warning: Bin width is non-positive. Defaulting to 10 bins.")
            return 10, np.nan

        bin_width = round(bin_width, 3)  # Round to 3 decimal places
        range_data = np.max(data) - np.min(data)  # Data range

        # Prevent division by zero in num_bins calculation
        num_bins = max(1, int(range_data / bin_width)) if bin_width > 0 else 10

        return num_bins, bin_width


    def build_bins_by_width(self, input_df, input_bin_width, field_risk_score):
        """
        Segments a numerical field into bins of a specified width, allowing for float precision 
        in bin ranges.

        Parameters:
            input_df (DataFrame): The DataFrame containing the data.
            input_bin_width (float): The width of each bin.
            field_risk_score (str): The name of the column containing risk scores to be binned.

        Returns:
            df (DataFrame): The updated DataFrame with the newly created bin column.
        """
        
        df = input_df if input_df is not None else self.df

        # Remove existing bin column if it already exists
        if 'pred_bin' in df.columns:
            df.drop(columns=['pred_bin'], inplace=True)

        # Define bin edges using floating-point intervals
        min_score = df[field_risk_score].min()
        max_score = df[field_risk_score].max()
        bin_edges = np.arange(min_score, max_score + input_bin_width, input_bin_width)

        # Ensure the last bin captures the maximum value
        if bin_edges[-1] < max_score:
            bin_edges[-1] = max_score

        # Debugging print to check bin edges
        print("Min and Max Score:", min_score, max_score)
        print("Bin Edges:", bin_edges)

        # Create bin labels with float precision
        bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)]

        # Assign bins using pd.cut(), including the lower bound and excluding the upper bound
        df['pred_bin'] = pd.cut(df[field_risk_score], bins=bin_edges, labels=bin_labels, include_lowest=True, right=False)

        # Add columns for lower and upper bin boundaries
        df[f"pred_bin_LOWERB"] = pd.cut(df[field_risk_score], bins=bin_edges, labels=bin_edges[:-1], include_lowest=True, right=False)
        df[f"pred_bin_UPPERB"] = pd.cut(df[field_risk_score], bins=bin_edges, labels=bin_edges[1:], include_lowest=True, right=False)

        return df
    
    def build_bins_manually(self, input_df, field_risk_score, list_bin_thresholds):
        """
        Creates bins for a numeric column based on manually specified thresholds.

        Parameters:
            input_df (DataFrame): The DataFrame containing the data.
            field_risk_score (str): The column name containing the numeric values to be binned.
            list_bin_thresholds (list): A list of numeric values that define the bin boundaries.

        Returns:
            DataFrame: The updated DataFrame with the new binned column and additional bin boundary columns.
        """
        
        df = input_df if input_df is not None else self.df

        # Remove existing bin column if it already exists
        if 'pred_bin' in df.columns:
            df = df.drop(columns=['pred_bin'])

        # Sort thresholds and remove duplicates
        sorted_thresholds = sorted(set(list_bin_thresholds))

        # Determine the minimum and maximum values from the dataset
        min_value = df[field_risk_score].min()
        max_value = df[field_risk_score].max()

        # Round min and max values to two decimal places
        min_value = math.floor(min_value * 100) / 100.0
        max_value = math.ceil(max_value * 100) / 100.0

        # Extend bin edges to fully encompass the data range
        if min_value < sorted_thresholds[0]:
            sorted_thresholds = [min_value] + sorted_thresholds
        if max_value > sorted_thresholds[-1]:
            sorted_thresholds.append(max_value)

        # Define bin edges based on the adjusted threshold list
        bin_edges = sorted_thresholds

        # Generate human-readable bin labels
        bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]

        # Assign each value to a bin using pd.cut()
        df['pred_bin'] = pd.cut(df[field_risk_score], bins=bin_edges, labels=bin_labels, include_lowest=True, right=False)

        # Create additional columns for lower and upper bin boundaries
        df[f"pred_bin_LOWERB"] = pd.cut(df[field_risk_score], bins=bin_edges, labels=bin_edges[:-1], include_lowest=True, right=False)
        df[f"pred_bin_UPPERB"] = pd.cut(df[field_risk_score], bins=bin_edges, labels=bin_edges[1:], include_lowest=True, right=False)

        return df


# -------------------------------------------------------------------------
class build_model:
# -------------------------------------------------------------------------

    def __init__(self):
        """
        This class holds different functions to build a classification model.
        The default df is the dataframe assigned to the class, else, the input_df argument. 
        """
        
        self.df = pd.DataFrame()

    def calculate_folds(self, input_df, obs_size, num_folds, data_split_field, valid_splits):
        """
        Generates exact fold indices based on available dataset indices, filtering based on a specified field.

        Parameters:
            input_df (pd.DataFrame): The dataset.
            obs_size (int): The number of observations per fold.
            num_folds (int): The number of folds to generate.
            data_split_field (str): Column used to filter dataset for training/validation.
            valid_splits (list): List of values in `data_split_field` to include.
        
        Returns:
            list of lists: Each list contains exact row indices for a fold.
        """
        # Filter dataset based on data_split_field
        filtered_df = input_df[input_df[data_split_field].isin(valid_splits)].copy()
        available_indices = filtered_df.sort_values("Time step").index.to_numpy()  # Ensure sorted order
        total_obs = len(available_indices)
        
        if obs_size > total_obs:
            print(f"Warning: obs_size ({obs_size}) is larger than total dataset size ({total_obs}). Adjusting to total dataset size.")
            obs_size = total_obs
        
        fold_indices = []
        
        for _ in range(num_folds):
            if total_obs - obs_size < 1:
                break
            start_idx = np.random.randint(0, total_obs - obs_size + 1)  # Ensure within bounds
            selected_indices = available_indices[start_idx:start_idx + obs_size]  # Select exact indices
            fold_indices.append(selected_indices)  # Store list of indices
        
        return fold_indices

    def split_temporal_data(self, input_df, indices, timestep_field, train_ratio=0.7):
        """
        Splits the dataset into training and validation periods while preserving temporal order.
        Ensures no overlap of training/validation within the same timestep.
        
        Parameters:
            input_df (pd.DataFrame): The dataset containing indexed rows.
            indices (list): The exact indices to use for the fold.
            timestep_field (str): Column name representing the time step.
            train_ratio (float): Ratio of training data (default is 0.7).
        
        Returns:
            df (pd.DataFrame): Updated dataset with 'training_label' and 'pred_proba' columns.
            summary_df (pd.DataFrame): Summary statistics in one row.
        """
        df = input_df.loc[indices].sort_values(timestep_field).copy()  # Select indices and sort by time step
        unique_timesteps = df[timestep_field].unique()
        train_timesteps = unique_timesteps[:int(len(unique_timesteps) * train_ratio)]
        
        df['training_label'] = 'Val'
        df.loc[df[timestep_field].isin(train_timesteps), 'training_label'] = 'Train'
        
        # Initialize column for predicted probabilities
        df['pred_proba'] = None
        
        # Summary statistics
        train_size = len(df[df['training_label'] == 'Train'])
        val_size = len(df[df['training_label'] == 'Val'])
        summary_df = pd.DataFrame({
            'Total Size': [len(df)],
            'Training Size': [train_size],
            'Val Size': [val_size],
            'Training %': [round((train_size / len(df)) * 100)],
            'Val %': [round((val_size / len(df)) * 100)]
        })
        
        return df, summary_df

    def prepare_train_val_data(self, input_df, drop_training_cols=None, output_column=None):
        """
        Prepares training and validation datasets from the labeled dataframe.
        
        Parameters:
            df (pd.DataFrame): The labeled dataframe with a 'training_label' column.
            drop_training_cols (list, optional): List of columns to drop for training.
            output_column (str, optional): Column to be used as the output target.
        
        Returns:
            X (pd.DataFrame): Training features.
            X_val (pd.DataFrame): Validation features.
            y_train (pd.Series or pd.DataFrame): Training labels.
            y_val (pd.Series or pd.DataFrame): Validation labels.
        """

        df = input_df if input_df is not None else self.df

        # Ensure the required column exists
        if 'training_label' not in df.columns:
            raise ValueError("DataFrame must have a 'training_label' column from split_temporal_data()")

        # Split into training and validation sets
        train_df = df[df['training_label'] == 'Train']
        val_df = df[df['training_label'] == 'Val']

        # Drop specified columns for training
        if drop_training_cols:
            X = train_df.drop(columns=drop_training_cols, errors='ignore')
            X_val = val_df.drop(columns=drop_training_cols, errors='ignore')
        else:
            X = train_df.copy()
            X_val = val_df.copy()

        # Select output column for validation
        if output_column:
            if output_column not in df.columns:
                raise ValueError(f"Output column '{output_column}' not found in DataFrame.")

            y_train = train_df[output_column]
            y_val = val_df[output_column]
            X = X.drop(columns=[output_column], errors='ignore')
            X_val = X_val.drop(columns=[output_column], errors='ignore')
        else:
            y_train, y_val = None, None  # No output column specified

        return X, X_val, y_train, y_val
    
    def normalize_after_split(self, X_train, X_val, columns_to_normalize):
        """
        Normalizes specified features using StandardScaler after splitting the data.
        
        Parameters:
            X_train, X_val (pd.DataFrame): Feature sets for training and validation.
            columns_to_normalize (list): List of columns to normalize.
        
        Returns:
            X_train_scaled, X_val_scaled (pd.DataFrame): Scaled feature sets.
            scaler (StandardScaler): The fitted StandardScaler instance.
        """
        from sklearn.preprocessing import StandardScaler
        
        # Copy data to avoid modifying the original
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        
        # Fit scaler only on training data and transform
        scaler = StandardScaler().fit(X_train[columns_to_normalize])
        X_train_scaled[columns_to_normalize] = scaler.transform(X_train[columns_to_normalize])
        X_val_scaled[columns_to_normalize] = scaler.transform(X_val[columns_to_normalize])
        
        return X_train_scaled, X_val_scaled, scaler

    def evaluate_model_performance(self, input_df, y_true, y_pred_proba, timestep_field, threshold=0.5, fold_name='Fold 1'):
        """
        Evaluates model performance and returns a one-line summary DataFrame.

        Parameters:
            input_df (pd.DataFrame): Original input DataFrame.
            y_true (pd.Series): Actual class labels.
            y_pred_proba (pd.Series): Predicted probabilities.
            timestep_field (str): Column name representing the time step.
            threshold (float): Classification threshold (default = 0.5).
            fold_name (str): Name of the fold (default = 'Fold 1').

        Returns:
            pd.DataFrame: Summary of model performance in one row.
        """
        from sklearn.metrics import roc_auc_score

        # Ensure inputs are valid Pandas Series
        if not isinstance(y_true, pd.Series) or not isinstance(y_pred_proba, pd.Series):
            raise TypeError("y_true and y_pred_proba must be Pandas Series.")

        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Handle edge case where confusion matrix doesn't return a full 2x2 matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(set(y_true)) > 1 else (0, 0, 0, sum(y_true))

        # Calculate AUC-ROC and Gini score
        if len(set(y_true)) > 1:  # Ensure at least two classes exist
            auc_score = roc_auc_score(y_true, y_pred_proba)
            gini_score = 2 * auc_score - 1
        else:
            auc_score = None
            gini_score = None  # Undefined if only one class present

        # Calculate min and max timesteps
        min_timestep = input_df.get(timestep_field, pd.Series()).min()
        max_timestep = input_df.get(timestep_field, pd.Series()).max()

        # Count observations in training and validation
        train_count = len(input_df[input_df['training_label'] == 'Train'])
        val_count = len(input_df[input_df['training_label'] == 'Val'])
        
        # Count class distribution in training and validation
        train_class_counts = input_df[input_df['training_label'] == 'Train']['class'].value_counts().to_dict()
        val_class_counts = input_df[input_df['training_label'] == 'Val']['class'].value_counts().to_dict()
        
        # Count overall class distribution
        overall_class_counts = input_df['class'].value_counts().to_dict()
        class_0_size = overall_class_counts.get(0, 0)
        class_1_size = overall_class_counts.get(1, 0)

        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'Fold': [fold_name],
            'Threshold': [threshold],
            'Total Size': [len(y_true)],
            'Training Size': [train_count],
            'Validation Size': [val_count],
            'Class 0 Size': [class_0_size],
            'Class 1 Size': [class_1_size],
            'Train Class Distribution': [train_class_counts],
            'Val Class Distribution': [val_class_counts],
            'Min Time Step': [min_timestep],
            'Max Time Step': [max_timestep],
            'Accuracy': [accuracy],
            'Recall': [recall],
            'Precision': [precision],
            'F1-Score': [f1],
            'AUC-ROC': [auc_score],  # Added AUC-ROC
            'Gini Score': [gini_score],  # Added Gini score
            'True Positive': [tp],
            'True Negative': [tn],
            'False Positive': [fp],
            'False Negative': [fn]
        })

        return summary_df


    # visualise the split of training and validation data by timestep as a stacked bar chart. 
    def visualize_timestep_distribution(self, input_df, viz_field):
        """
        Visualizes the distribution of Training, Validation, and Test data over time using a stacked bar chart.

        Parameters:
            dataset_imputed (pd.DataFrame): The dataset containing the 'Time step' and viz_field columns.
        """

        df = input_df if input_df is not None else self.df

        # Count the number of occurrences of each label per timestep
        timestep_distribution = df.groupby('Time step')[viz_field].value_counts().unstack().fillna(0)

        # Define colors for each dataset label
        colors = {'Train': 'skyblue', 'Val': 'grey', 'Test': '#FFD700'}  # Butter yellow

        # Plot the stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 3))
        timestep_distribution.plot(kind='bar', stacked=True, color=[colors[label] for label in timestep_distribution.columns], ax=ax, edgecolor='grey')

        # Labels and title
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Count of Observations')
        ax.set_title('Distribution of Data Splits Over Time')

        # Legend
        ax.legend(title='Dataset Label')

        # Display the plot
        plt.show()
    
    def run_cross_validation(self, input_df, timestep_field, drop_training_cols, columns_to_normalize, 
                            data_split_field, valid_splits, model, obs_size=5000, num_folds=5, 
                            train_ratio=0.7, threshold=0.5):
        """
        Runs cross-validation using precomputed folds, normalizes data using normalize_after_split, 
        trains a logistic regression model, evaluates performance, and stores trained models, scalers, 
        and fold indices.

        Returns:
            pd.DataFrame: Summary of evaluation metrics across all folds.
            dict: Dictionary containing trained models for each fold.
            dict: Dictionary containing fitted scalers for each fold.
            dict: Dictionary containing training and validation indices for each fold.
        """
        
        all_eval_dfs = []  # Store evaluation results for all folds
        trained_models = {}  # Store trained models per fold
        scalers = {}  # Store fitted scalers per fold
        fold_indices_dict = {}  # Store training/validation indices per fold
        
        # Generate fold indices based on exact row indices
        fold_indices = self.calculate_folds(input_df, obs_size, num_folds, data_split_field, valid_splits)
        
        fold_number = 1  # Track fold count
        
        for indices in fold_indices:
            df_fold, summary_df = self.split_temporal_data(
                input_df=input_df,
                indices=indices,
                timestep_field=timestep_field,
                train_ratio=train_ratio
            )
            
            print(f"Processing Fold {fold_number}: {len(df_fold)} observations...")
            
            # Visualize fold distribution
            self.visualize_timestep_distribution(input_df=df_fold, viz_field='training_label')

            # Prepare training and validation sets
            X_train, X_val, y_train, y_val = self.prepare_train_val_data(
                input_df=df_fold, 
                drop_training_cols=drop_training_cols, 
                output_column='class'
            )

            # Save the training and validation indexes
            fold_indices_dict[f'Fold {fold_number}'] = {
                "train_indices": X_train.index.tolist(),
                "val_indices": X_val.index.tolist()
            }

            # Normalize data using the dedicated function
            X_train_scaled, X_val_scaled, scaler = self.normalize_after_split(X_train, X_val, columns_to_normalize)

            # Store the fitted scaler in memory
            scalers[f'Fold {fold_number}'] = scaler

            # OLD VERSION WTIHOUT HYPERPARAMETER TUNING
            # Train model
            #model.fit(X_train_scaled, y_train)

            # Store the trained model in memory
            #fold_name = f'Fold {fold_number}'
            #trained_models[fold_name] = model

            # Predict probabilities
            #y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
            #y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

            # NEW VERSION WITH HYPER PARAMETER TUNING
            # --- Clone & fit a fresh estimator for THIS fold ---
            estimator = clone(model)                      # new, unfitted copy (works for RF or search wrappers)
            estimator.fit(X_train_scaled, y_train)        # fit only on this fold's training slice

            # Store the trained estimator in memory (per-fold)
            fold_name = f'Fold {fold_number}'
            trained_models[fold_name] = estimator

            # Use the fold's estimator for predictions
            y_train_pred_proba = estimator.predict_proba(X_train_scaled)[:, 1]
            y_val_pred_proba   = estimator.predict_proba(X_val_scaled)[:, 1]

            # Assign predicted probabilities to df_fold
            df_fold.loc[X_train_scaled.index, "pred_proba"] = y_train_pred_proba
            df_fold.loc[X_val_scaled.index, "pred_proba"] = y_val_pred_proba

            # Evaluate model performance
            eval_df = self.evaluate_model_performance(
                input_df=df_fold,
                y_true=df_fold['class'],
                y_pred_proba=df_fold['pred_proba'],
                timestep_field=timestep_field,
                threshold=threshold,
                fold_name=fold_name
            )

            # Append evaluation results
            all_eval_dfs.append(eval_df)
            
            fold_number += 1  # Increment fold count

        # Combine results from all folds into a single summary DataFrame
        summary_results = pd.concat(all_eval_dfs, ignore_index=True) if all_eval_dfs else pd.DataFrame()
        
        return summary_results, trained_models, scalers, fold_indices_dict

    def select_best_model_and_scaler(self, cross_fold_summary, saved_models, saved_scalers, top_n=5):
        """
        Selects the best model based on Gini Score first, then Recall, and finally Accuracy in case of a tie.
        Returns the best model and its corresponding scaler.

        Parameters:
            cross_fold_summary (pd.DataFrame): DataFrame containing model evaluation metrics.
            saved_models (dict): Dictionary of trained models indexed by fold name.
            saved_scalers (dict): Dictionary of scalers indexed by fold name.
            top_n (int): Number of top models to display (default = 3).

        Returns:
            best_model: The best-trained model.
            best_scaler: The scaler associated with the best model.
        """
        # Sorting by Gini Score first, then Recall, then Accuracy (all descending)
        sorted_summary = cross_fold_summary.sort_values(by=["Gini Score", "Recall", "Accuracy"], ascending=[False, False, False]).head(top_n)

        print("Top Models Ranked:")
        for rank, (_, row) in enumerate(sorted_summary.iterrows(), 1):
            fold_name = row['Fold']
            print(f"{rank}. {fold_name} - Gini Score: {row['Gini Score']:.4f}, Recall: {row['Recall']:.4f}, Accuracy: {row['Accuracy']:.4f}, "
                f"AUC-ROC: {row['AUC-ROC']:.4f}, Precision: {row['Precision']:.4f}, F1-Score: {row['F1-Score']:.4f}")

        # Select the best model (first in sorted list)
        best_fold = sorted_summary.iloc[0]["Fold"]
        best_model = saved_models[best_fold]
        best_scaler = saved_scalers[best_fold]

        print(f"\nBest Model Selected: {best_fold}")
        
        return best_model, best_scaler

    def _get_cv_meta(self,mdl):
        """Extract inner-CV meta: scorer name, CV class, n_splits."""
        scorer = getattr(mdl, "scoring", None)
        cv_obj = getattr(mdl, "cv", None)
        cv_class = cv_obj.__class__.__name__ if cv_obj is not None else None
        # Try to infer n_splits
        n_splits = None
        if cv_obj is not None and hasattr(cv_obj, "n_splits"):
            # TimeSeriesSplit has .n_splits; KFold/StratifiedKFold too
            n_splits = cv_obj.n_splits
        else:
            # Fallback: count splitX columns in cv_results_
            if hasattr(mdl, "cv_results_"):
                cols = [c for c in mdl.cv_results_.keys() if c.startswith("split") and c.endswith("_test_score")]
                if cols:
                    # split0..split{n-1}
                    nums = sorted(set(int(c.split("_")[0].replace("split","")) for c in cols))
                    n_splits = len(nums)
        return scorer, cv_class, n_splits

    def _get_best_idx(self,cvr):
        """Return the row index of the winning candidate in cv_results_."""
        # rank_test_score is 1-based rank; smallest is best
        return int(np.argmin(cvr["rank_test_score"]))

    def build_tuning_params(self,trained_models: dict) -> pd.DataFrame:
        rows = []

        for fold_name, mdl in trained_models.items():
            row = {"fold": fold_name}

            # Case 1: search wrapper (Halving/Randomized)
            if hasattr(mdl, "best_estimator_") and hasattr(mdl, "cv_results_"):
                row["tuning_method"] = mdl.__class__.__name__

                # Inner-CV meta
                scorer, cv_class, n_splits = self._get_cv_meta(mdl)
                row["inner_cv_scoring"] = scorer
                row["inner_cv_class"]   = cv_class
                row["inner_cv_n_splits"]= n_splits

                # Aggregate scores for the winning candidate
                cvr = mdl.cv_results_
                best_idx = self._get_best_idx(cvr)
                row["best_score"]     = float(cvr["mean_test_score"][best_idx])
                row["best_score_std"] = float(cvr["std_test_score"][best_idx])
                row["n_candidates_tested"] = int(len(cvr["params"]))

                # Per-split scores for the winning candidate (split0..split{n-1})
                if n_splits is not None:
                    for s in range(n_splits):
                        key = f"split{s}_test_score"
                        if key in cvr:
                            row[f"best_split{s}_score"] = float(cvr[key][best_idx])

                # Params used (after refit on outer-train) vs tuned subset
                used  = mdl.best_estimator_.get_params()
                tuned = getattr(mdl, "best_params_", {})
                param_keys = [
                "n_estimators","max_depth","min_samples_split","min_samples_leaf",
                "max_features","class_weight","bootstrap","random_state"]

                for k in param_keys:
                    row[f"used__{k}"]  = used.get(k, None)
                    row[f"tuned__{k}"] = tuned.get(k, None)

                # Extra halving diagnostics (if applicable)
                # These attrs exist on Halving* search CVs
                for attr in ["resource", "min_resources_", "max_resources_", "factor"]:
                    row[f"halving__{attr}"] = getattr(mdl, attr, None)

                # Iteration-wise candidates/resources (Halving)
                # n_resources_ and n_candidates_ are sequences (one per iteration) if present
                n_res = getattr(mdl, "n_resources_", None)
                n_cand = getattr(mdl, "n_candidates_", None)
                if n_res is not None:
                    row["halving__n_iterations"] = len(n_res)
                    row["halving__resources_schedule"] = list(map(int, n_res))
                if n_cand is not None:
                    row["halving__candidates_schedule"] = list(map(int, n_cand))

            # Case 2: plain fitted estimator (no search wrapper)
            else:
                row["tuning_method"] = "None"
                row["inner_cv_scoring"] = None
                row["inner_cv_class"]   = None
                row["inner_cv_n_splits"]= None
                row["best_score"]       = np.nan
                row["best_score_std"]   = np.nan
                row["n_candidates_tested"] = np.nan
                est = mdl if hasattr(mdl, "get_params") else None
                if est is not None:
                    used = est.get_params()
                    for k in param_keys:
                        row[f"used__{k}"]  = used.get(k, None)
                        row[f"tuned__{k}"] = used.get(k, None)

            rows.append(row)

        df = (pd.DataFrame(rows)
                .sort_values("fold")
                .reset_index(drop=True))
        return df

# -------------------------------------------------------------------------
class model_reporting:
# -------------------------------------------------------------------------

    def __init__(self):
        """
        This class holds different functions to report on model performance.
        The default df is the dataframe assigned to the class, else, the input_df argument. 
        """
        
        self.df = pd.DataFrame()

    def count_classes_by_bin(self, input_df, class_label, field_txID, field_bin):
        """
        Counts occurrences of class labels within each bin, pivots the data into a table format, 
        and adds a total column for each bin.

        Parameters:
            input_df (DataFrame): The dataset containing the bin and class label information.
            class_label (str): The column representing the class labels.
            field_txID (str): The column containing the transaction ID (used to count occurrences).
            field_bin (str): The column representing the bins for grouping.

        Returns:
            DataFrame: A pivot table with class counts per bin and a total column.
        """
        
        # ---------------- COUNT OBSERVATIONS BY CLASS ----------------
        # Group data by bin and class, then count occurrences
        df_classes = input_df.groupby([field_bin, class_label], observed=True).agg({field_txID: 'count'}).reset_index()

        # Pivot into a table format (similar to a confusion matrix)
        pivot_classes = df_classes.pivot_table(index=[field_bin], columns=class_label, values=field_txID, 
                                            fill_value=0, observed=True)

        # Add a total column summing all class counts per bin
        pivot_classes['TOTAL'] = pivot_classes.sum(axis=1)

        return pivot_classes
    
    def cumulperc_class_by_bin(self, input_pivot):
        """
        Calculate the percentage of total and cumulative percentage of total for all class types.
        Adds the percentage and cumulative percentage columns to the input DataFrame and reorders them.

        Parameters:
            input_pivot (DataFrame): A pivot table where class types are counted by bin.

        Returns:
            DataFrame: Updated pivot table with percentage and cumulative percentage columns, reordered.
        """
        
        pivot = input_pivot.copy()

        # ---------------- CUMULATIVE PERCENTAGE OF CLASS COUNTS BY BIN ----------------
        # Calculate the sum of each column to get the total counts per class type
        column_totals = pivot.sum(axis=0)

        # Calculate percentage of total and cumulative percentage for each class type
        pct_columns = []
        cumpct_columns = []

        for column in pivot.columns:
            # Compute percentage contribution per class
            pct_col = column + '_PCT'
            cumpct_col = column + '_CUMPCT'

            pivot[pct_col] = pivot[column] / column_totals[column]
            pivot[cumpct_col] = pivot[pct_col].cumsum()

            # Track column names for reordering
            pct_columns.append(pct_col)
            cumpct_columns.append(cumpct_col)

        # Format results to 3 decimal places
        pivot = pivot.round(3)

        # ---------------- REORDER COLUMNS ----------------
        # Ensure original class counts, followed by percentages, then cumulative percentages
        ordered_columns = list(pivot.columns[:len(pivot.columns)//3]) + pct_columns + cumpct_columns
        pivot = pivot[ordered_columns]

        return pivot


    def tpfp_class_by_bin(self, input_df, field_bin, field_class_binary, field_txID):
        """
        Computes Illicit to Licit odds and Illicit conversion rate.
        """
        df = input_df.groupby([field_bin, field_class_binary])[field_txID].count().unstack(fill_value=0)
        df.columns = ['Licit_Count', 'Illicit_Count']

        df['Illicit_Licit_ODDS'] = np.where(df['Licit_Count'] == 0, df['Illicit_Count'], df['Illicit_Count'] / df['Licit_Count'])
        df['Illicit_CONVERSION'] = df['Illicit_Count'] / (df['Illicit_Count'] + df['Licit_Count'])

        return df[['Illicit_Licit_ODDS', 'Illicit_CONVERSION']].round(2)

    def append_totals(self, df):
        """
        Calculates the totals for numeric columns and appends a 'Total' row to the DataFrame.

        Parameters:
            df (DataFrame): The DataFrame containing binned prediction data.

        Returns:
            DataFrame: The original DataFrame with an appended totals row, ensuring odds and conversion are correctly recalculated.
        """

        # Compute total sums for all numeric columns
        totals = df.sum(numeric_only=True).to_dict()  # Convert Series to dictionary to allow safe updates

        # Ensure cumulative percentages (CUMPCT) are set to 1 by assigning them individually
        for col in df.columns:
            if 'CUMPCT' in col:
                totals[col] = 1  # Correctly assign scalar values to cumulative percentage columns

        # Recalculate Illicit_Licit_ODDS (Illicit / Licit)
        if 'Licit' in totals and 'Illicit' in totals:
            totals['Illicit_Licit_ODDS'] = totals['Illicit'] / totals['Licit'] if totals['Licit'] != 0 else totals['Illicit']

        # Recalculate Illicit_CONVERSION (Illicit / TOTAL)
        if 'Illicit' in totals and 'TOTAL' in totals:
            totals['Illicit_CONVERSION'] = totals['Illicit'] / totals['TOTAL']

        # Convert totals to a DataFrame and append it to the original DataFrame
        totals_df = pd.DataFrame([totals], index=['Total'])

        return pd.concat([df, totals_df])
    

    def build_binned_pred_report(self, dataset):
        """
        Builds a binned prediction report by aggregating class counts, computing percentages, 
        calculating TP/FP odds, and appending totals.

        Parameters:
            dataset (DataFrame): The dataset containing classification results.
            report (object): The report module/class containing necessary processing functions.

        Returns:
            DataFrame: A comprehensive binned prediction report.
        """

        # Step 1: Count classes by bin
        pivot_counts = self.count_classes_by_bin(dataset, class_label='class_label', field_txID='txId', field_bin='pred_bin')

        # Step 2: Compute cumulative percentages
        pivot_counts = self.cumulperc_class_by_bin(pivot_counts)

        # Step 3: Compute Illicit-Licit odds and conversion
        tpfp_counts = self.tpfp_class_by_bin(dataset, field_bin='pred_bin', field_class_binary='class', field_txID='txId')

        # Step 4: Merge all computed reports
        joined_report = pd.merge(pivot_counts, tpfp_counts, how='left', on='pred_bin')

        # Step 5: Append totals row to the final report
        final_report = self.append_totals(joined_report)

        return final_report

    def calculate_cm_metrics(self, input_df, field_class_binary, field_pred_proba, input_threshold):
        """
        Builds a confusion matrix and calculates classification metrics based on a given threshold.

        Parameters:
            input_df (DataFrame): The dataset containing classification results.
            field_class_binary (str): Column name for the actual class labels (binary 0/1).
            field_pred_proba (str): Column name for the predicted probability scores.
            input_threshold (float): The threshold to classify predictions as 1 or 0.

        Returns:
            metrics_df (DataFrame): DataFrame containing accuracy, precision, recall, and F1-score.
            cm_count_df (DataFrame): DataFrame containing the confusion matrix counts.
        """

        df = input_df.copy()

        # Apply threshold to create a binary prediction column
        df['PRED_BINARY'] = df[field_pred_proba].apply(lambda row: 1 if row >= input_threshold else 0)

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(df[field_class_binary], df['PRED_BINARY']).ravel()

        # Store confusion matrix counts
        cm_counts = {
            'THRESHOLD': input_threshold,
            'TRUE_POSITIVE': tp,
            'TRUE_NEGATIVE': tn,
            'FALSE_POSITIVE': fp,
            'FALSE_NEGATIVE': fn,
            'TOTAL': tn + tp + fn + fp
        }

        cm_count_df = pd.DataFrame([cm_counts])

        # Compute classification performance metrics
        value_accuracy = (tp + tn) / (tp + tn + fp + fn)
        value_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        value_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        value_f1 = (2 * (value_precision * value_recall) / (value_precision + value_recall)) if (value_precision + value_recall) > 0 else 0

        # Store metric formulas
        formula_accuracy = f"{value_accuracy:.4f} = ((TP + TN) / (TP + TN + FP + FN))"
        formula_precision = f"{value_precision:.4f} = (TP / (TP + FP))"
        formula_recall = f"{value_recall:.4f} = (TP / (TP + FN))"
        formula_f1 = f"{value_f1:.4f} = 2 * (Precision * Recall) / (Precision + Recall)"

        # Create a DataFrame for the computed metrics
        metrics_data = {
            'METRIC': ['ACCURACY', 'PRECISION', 'RECALL', 'F1'],
            'THRESHOLD': input_threshold,
            'VALUE': [value_accuracy, value_precision, value_recall, value_f1],
            'FORMULA': [formula_accuracy, formula_precision, formula_recall, formula_f1]
        }

        metrics_df = pd.DataFrame(metrics_data)

        # Round the VALUE column to 2 decimal places
        metrics_df['VALUE'] = metrics_df['VALUE'].round(2)

        # Drop the temporary prediction column
        df.drop(columns=['PRED_BINARY'], inplace=True)

        return metrics_df, cm_count_df


    def calculate_cm_performance(self, input_df, field_class_binary, field_pred_proba):
        """
        Compiles a confusion matrix performance report across different thresholds.

        Parameters:
            input_df (DataFrame): The dataset containing classification results.
            field_class_binary (str): Column name for the actual class labels (binary 0/1).
            field_pred_proba (str): Column name for the predicted probability scores.

        Returns:
            cm_report (DataFrame): Performance metrics across thresholds (Accuracy, Recall, Precision, F1-score).
            cm_results_df (DataFrame): Detailed classification metrics for each threshold.
        """

        df = input_df.copy()

        # Extract unique threshold values from pred_bin_UPPERB
        max_thresholds = df['pred_bin_UPPERB'].drop_duplicates().sort_values().tolist()

        list_cm_results = []  # Stores metrics for each threshold
        list_cm_counts = []   # Stores confusion matrix counts for each threshold

        # Iterate over each threshold and compute confusion matrix metrics
        for threshold in max_thresholds:
            cm_metrics, cm_counts = self.calculate_cm_metrics(
                input_df=df,
                field_class_binary=field_class_binary,
                field_pred_proba=field_pred_proba,
                input_threshold=threshold
            )

            list_cm_results.append(cm_metrics)
            list_cm_counts.append(cm_counts)

        # Convert results lists into DataFrames
        cm_results_df = pd.concat(list_cm_results, ignore_index=True)
        cm_counts_df = pd.concat(list_cm_counts, ignore_index=True)

        # Pivot the metrics DataFrame to structure it by threshold and metric
        cm_report = cm_results_df.pivot_table(index='THRESHOLD', columns='METRIC', values='VALUE', fill_value=0, observed=True)

        # Order metrics in the output report
        cm_report = cm_report[['ACCURACY', 'RECALL', 'PRECISION', 'F1']]

        # Merge performance metrics with confusion matrix counts
        cm_report = pd.merge(cm_report, cm_counts_df, how='left', on='THRESHOLD').set_index('THRESHOLD')

        return cm_report, cm_results_df

    def plot_cm_metrics(self, input_df, plot_x_axis, plot_value, classes):
        """
        Plots performance metrics at different thresholds.

        Parameters:
            input_df (DataFrame): The dataset containing classification metrics.
            plot_x_axis (str): The column name for the x-axis (typically 'THRESHOLD').
            plot_value (str): The column name for the y-axis (typically 'VALUE').
            classes (str): The column name used for different metric categories (typically 'METRIC').
        """

        df = input_df.copy()

        # Set the Seaborn theme
        sns.set_theme(style="whitegrid")

        # Create a line plot
        sns.lineplot(data=df, x=plot_x_axis, y=plot_value, hue=classes, marker='o')

        # Add plot titles and labels
        plt.title('Performance Metrics at Various Thresholds')
        plt.xlabel('Threshold')
        plt.ylabel('Value')
        plt.legend(title='Metric')
        plt.tight_layout()

        # Show the plot
        plt.show()
        plt.close()

    def evaluate_final_model(self, y_true, y_pred, label):
        """
        Evaluates model performance and returns a one-line summary DataFrame.

        Parameters:
            y_true (pd.Series): Actual class labels.
            y_pred (pd.Series): Predicted class labels (0 or 1).

        Returns:
            pd.DataFrame: Summary of model performance in one row.
        """
        # Ensure inputs are valid Pandas Series
        if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
            raise TypeError("y_true and y_pred must be Pandas Series.")

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else None
        gini_score = 2 * auc_score - 1 if auc_score is not None else None

        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(set(y_true)) > 1 else (0, 0, 0, sum(y_true))

        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'Label': [label],
            'Total Size': [len(y_true)],
            'Accuracy': [accuracy],
            'Recall': [recall],
            'Precision': [precision],
            'F1-Score': [f1],
            'AUC-ROC': [auc_score],
            'Gini Score': [gini_score],
            'True Positive': [tp],
            'True Negative': [tn],
            'False Positive': [fp],
            'False Negative': [fn]
        })

        return summary_df



######## END OF CLASS ########
