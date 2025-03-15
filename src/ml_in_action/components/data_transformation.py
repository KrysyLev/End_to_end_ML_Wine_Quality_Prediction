import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from ml_in_action import logger


class DataTransformation:
    def __init__(self, config):
        self.config = config

    def train_test_spliting(self):
        # Load dataset
        data = pd.read_csv(self.config.data_path)

        # Drop the last column (assuming it's named 'Id')
        if "Id" in data.columns:
            data = data.drop(columns=["Id"])

        # Splitting features and target variable
        X = data.iloc[:, :-1]  # All columns except last (features)
        y = data.iloc[:, -1]  # Last column (target)

        # Train-test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Apply feature scaling
        scaler = StandardScaler()  # Use MinMaxScaler() if needed
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert back to DataFrame
        train = pd.DataFrame(X_train_scaled, columns=X.columns)
        train["quality"] = y_train.values  # Add target back

        test = pd.DataFrame(X_test_scaled, columns=X.columns)
        test["quality"] = y_test.values  # Add target back

        # Save processed data
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into training and test sets")
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

        print(f"Train shape: {train.shape}, Test shape: {test.shape}")
