import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class PollutionModel:

    def __init__(self, path):
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(path)
        self.model = self.create_pipeline()
        self.model = self.train_model()
        return None

    def load_data(self, path):
        # Load and split data
        data = pd.read_csv(path)
        data = data.drop_duplicates()
        y = data.high_level # Target
        X = data.drop(columns='high_level') #Feature
        return train_test_split(X, y, train_size=0.8)

    def create_pipeline(self):
        # Preprocess data
        # Impute then Scale for numerical variables: 
        num_transformer = Pipeline([
            ('num_imputer', SimpleImputer()),
            ('num_scaler', StandardScaler())])

        # Encode categorical variables
        cat_transformer = Pipeline([
            ('cat_imputer',SimpleImputer(strategy = 'most_frequent')),
            ('cat_encoder',OneHotEncoder(handle_unknown='ignore', sparse=False))])
        preprocessor = ColumnTransformer([
            ('num_transformer', num_transformer, self.X_train.select_dtypes(include=np.number).columns),
            ('cat_transformer', cat_transformer, ['state', 'type'])
        ])

        final_pipe = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', LogisticRegression(max_iter=2500))])

        return final_pipe

    def train_model(self):
        # Instanciate grid search
        grid_search = GridSearchCV(
        self.model, 
        param_grid={
        # Access any component of the pipeline, as far back as you want
        'preprocessing__num_transformer__num_imputer__strategy': ['mean', 'median'],
        'classifier__C': [0.1, 0.5, 1, 5, 10]},
        cv=5, scoring="accuracy")

        # Train model with grid search
        grid_search.fit(self.X_train, self.y_train)

        return grid_search.best_estimator_

    def score_model(self):
        # Score final model
        return self.model.score(self.X_test, self.y_test)

    def predict(self, path):
        # Predict (new) samples
        new_samples = pd.read_csv(path)
        pred = self.model.predict(new_samples)
        new_samples['Predictions'] = pred

        return new_samples