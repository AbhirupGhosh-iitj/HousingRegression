from utils import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def run_tuned_models():
    df = load_data()
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Ridge": (Ridge(), {
            'alpha': [0.1, 1.0, 10.0],
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky']
        }),
        "RandomForest": (RandomForestRegressor(), {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }),
        "SVR": (SVR(), {
            'kernel': ['linear', 'rbf'],
            'C': [1, 10],
            'epsilon': [0.1, 0.2]
        })
    }

    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - Best Params: {grid.best_params_}")
        print(f"{name} - MSE: {mse:.2f}, R²: {r2:.2f}")

if __name__ == "__main__":
    run_tuned_models()

