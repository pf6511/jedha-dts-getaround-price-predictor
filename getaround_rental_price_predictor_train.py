import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score,root_mean_squared_error

import time
import os
import mlflow

from collections import defaultdict
from typing import DefaultDict, List

from sklearn.base import BaseEstimator, TransformerMixin

PRICING_PROJECT_CSV_FILE_PATH = "data/get_around_pricing_project.csv"
TARGET_COLUMN = 'rental_price_per_day'

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

print("Logging to:", mlflow.get_tracking_uri())

def get_columns_by_dtype(dataset:pd.DataFrame) -> DefaultDict[str,List[str]]:
    columns_by_dtype = defaultdict(list)
    for col, dtype in dataset.dtypes.to_dict().items():
        columns_by_dtype[str(dtype)].append(col)
    return columns_by_dtype

# Custom transformer for model_key
class ModelKeyProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, column='model_key'):
        self.column = column
        self.model_key_counts_ = None

    def fit(self, X, y=None):
        df = X.copy()
        df[self.column] = df[self.column].str.lower()
        self.model_key_counts_ = df[self.column].value_counts().to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        df[self.column] = df[self.column].str.lower()
        df['model_key_count'] = df[self.column].map(self.model_key_counts_).fillna(0)
        return df.drop(columns=[self.column])
    
def prepare_dataset() -> pd.DataFrame:
    pricing_dtf = pd.read_csv(PRICING_PROJECT_CSV_FILE_PATH, delimiter=',', encoding="UTF-8", index_col=0)
    # Remove invalid data
    pricing_dtf = pricing_dtf[(pricing_dtf['mileage']>=0.0) & (pricing_dtf['engine_power']>0)]
    pricing_dtf['model_key']= pricing_dtf['model_key'].str.lower()
    return pricing_dtf

def prepare_pipeline(pricing_dtf:pd.DataFrame) -> GridSearchCV:
    columns_by_dtype = get_columns_by_dtype(pricing_dtf)
    categorical_columns = columns_by_dtype['object']
    numerical_columns = columns_by_dtype['int64'] + columns_by_dtype['float64'] 
    binary_columns = columns_by_dtype['bool']

    print('categorical_columns:', categorical_columns)
    print('numerical_columns:', numerical_columns)
    print('binary_columns:', binary_columns)

    TARGET_COLUMN = 'rental_price_per_day'  # adjust if different
    MODEL_KEY_COUNT_FEATURE = 'model_key_count'

    # Prepare feature lists
    num_features = numerical_columns.copy()
    num_features.remove(TARGET_COLUMN)
    num_features.append(MODEL_KEY_COUNT_FEATURE)
    categorical_features = categorical_columns.copy()
    if 'model_key' in categorical_features:
        categorical_features.remove('model_key')

    print('num_features:', num_features)
    print('categorical_features:', categorical_features)

    # Column transformer after model_key is processed & dropped
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", 'passthrough', num_features),
            ("cat", OneHotEncoder(), categorical_features),
            ("binary", 'passthrough', binary_columns)
        ]
    )

    # Preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('model_key_processing', ModelKeyProcessor(column='model_key')),
        ('column_transformer', column_transformer)
    ])

    # Full pipeline: preprocessing + model
    preprocess_and_model_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # Grid search parameters. Pour chaque param, le prefix du nom doit correspondre à la clé du step du type d emodèle sous-jacent
    params = {
        'model__max_depth': [3, 5, 7, 10],
        'model__min_samples_split': [8, 10, 20],
        'model__n_estimators': [50, 100],
        'model__max_features': ['sqrt', 'log2']
    }

    gridsearch = GridSearchCV(
        estimator=preprocess_and_model_pipeline,
        param_grid=params,
        cv=5,
        verbose=2,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    return gridsearch

if __name__ == "__main__":

        # Time execution
    start_time = time.time()

    print("preparing pipeline...")
    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False) # We won't log models right away
    pricing_dtf= prepare_dataset()
    gridsearch = prepare_pipeline(pricing_dtf)

    print("Separating labels from features...")
    Y = pricing_dtf.loc[:, TARGET_COLUMN]
    #X = pricing_dtf.drop({TARGET_COLUMN, 'model_key'}, axis=1)
    X = pricing_dtf.drop({TARGET_COLUMN}, axis=1)

    print("Dividing into train and test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    with mlflow.start_run(run_name="rf_gridsearch_with_pipeline") as run:
        print("training model...")
        gridsearch.fit(X_train, Y_train)
            # Best model
        best_pipeline = gridsearch.best_estimator_
        # Log hyperparameters
        mlflow.log_params(gridsearch.best_params_)
        # Log  / tag experiment version
        mlflow.log_param("preprocessing_model_key_case", "lowercase")
        # Predict and evaluate
        Y_test_pred = best_pipeline.predict(X_test)
        rmse = root_mean_squared_error(Y_test, Y_test_pred)
        r2 = r2_score(Y_test, Y_test_pred)
    
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model
        mlflow.sklearn.log_model(best_pipeline, "model")
        print("Logged model and metrics to MLflow.")
        print("Model trained and logged to:", run.info.artifact_uri)