import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

def split_data(df, target, test_size=0.2):
    """Split the dataset into training and testing sets."""
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def evaluate_model(model, X_test, y_test):
    """Evaluate the performance of a model."""
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

def grid_search(model, X_train, y_train):
    """Perform grid search for hyperparameter tuning."""
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def evaluate_cross_validation(model, X, y):
    """Evaluate the performance of a model using cross-validation."""
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print("Cross-Validation Scores:", scores)
    print("Average Cross-Validation Score:", np.mean(scores))

def scale_data(X_train, X_test):
    """Scale the data using StandardScaler."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def get_model(name):
    """Return a model based on the given name."""
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100),
        'support_vector_machine': SVC(),
        'decision_tree': DecisionTreeClassifier(),
        'k_nearest_neighbors': KNeighborsClassifier(),
        'ridge_regression': RidgeClassifier(),
        'dummy': DummyClassifier(strategy='stratified')
    }
    
    return models.get(name)

def get_pipeline(name):
    """Return a pipeline based on the given name."""
    models = {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000))
        ]),
        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100))
        ]),
        'support_vector_machine': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC())
        ]),
        'decision_tree': Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier())
        ]),
        'k_nearest_neighbors': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier())
        ]),
        'ridge_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RidgeClassifier())
        ]),
        'dummy': Pipeline([
            ('model', DummyClassifier(strategy='stratified'))
        ])
    }
    
    return models.get(name)