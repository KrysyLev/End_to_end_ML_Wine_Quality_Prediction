import os
from ml_in_action import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class DataTransformation:
    def __init__(self, config):
        self.config = config

    def train_test_spliting(self):
        # Load dataset
        data = pd.read_csv(self.config.data_path)

        # Drop the last column (assuming it's named 'Id')
        if 'Id' in data.columns:
            data = data.drop(columns=['Id'])
        
        # Rename columns to better recall them.
        data.rename(columns = {"fixed acidity": "fixed_acidity",
                            "volatile acidity": "volatile_acidity",
                            "citric acid": "citric_acid",
                            "residual sugar": "residual_sugar",
                            "chlorides": "chlorides",
                            "free sulfur dioxide": "free_sulfur_dioxide",
                            "total sulfur dioxide": "total_sulfur_dioxide"},
                    inplace = True)

        # Replace with the label Bad, Middle, Good
        data = data.replace({'quality': {
            8: 'Good',
            7: 'Good',
            6: 'Middle',
            5: 'Middle',
            4: 'Bad',
            3: 'Bad',
        }})
        
        # Data Regularization
        X_temp = data.drop(columns='quality')
        y = data.quality
        
        scaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(X_temp)
        X = pd.DataFrame(scaler, columns=X_temp.columns)

        # Train-test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert back to DataFrame
        train = pd.DataFrame(X_train, columns=X.columns)
        train['quality'] = y_train.values  # Add target back

        test = pd.DataFrame(X_test, columns=X.columns)
        test['quality'] = y_test.values  # Add target back

        # Save processed data
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into training and test sets")
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")

        print(f"Train shape: {train.shape}, Test shape: {test.shape}")