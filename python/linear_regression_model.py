"""
This file contains a custom-made linear regression model.
The purpose of this project is to deeply understand how a linear regression model works
by implementing it from scratch without using libraries like scikit-learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class Linear_regression:
    """
    A simple linear regression model built from scratch.
    It supports fitting a model, making predictions, and scoring the model using the R² metric.
    """
    x: float = None  # Input feature (optional placeholder)
    y: float = None  # Target feature (optional placeholder)
    m: float = None  # Slope of the regression line
    c: float = None  # Intercept of the regression line
    e: float = None  # Error (not used in this version)
    pred: list[float] = []  # Stores the predicted values

    def __init__(self) -> None:
        pass

    def fit_model(self, predictor_x_train: pd.Series, target_y_train: pd.Series) -> None:
        """
        Fits the linear regression model to the training data using the least squares method.

        Parameters:
            predictor_x_train (pd.Series): The independent variable (X)
            target_y_train (pd.Series): The dependent variable (Y)
        """
        x = predictor_x_train
        y = target_y_train

        # Calculate the means of x and y
        x_mean = x.mean()
        y_mean = y.mean()

        # Calculate the slope (m) using the least squares formula
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        self.m = numerator / denominator

        # Calculate the intercept (c)
        self.c = y_mean - self.m * x_mean

    def predict(self, predictor_x_test: pd.Series) -> pd.DataFrame:
        """
        Predicts target values using the trained model.

        Parameters:
            predictor_x_test (pd.Series): Input values for prediction

        Returns:
            pd.DataFrame: Predicted values as a DataFrame
        """
        self.pred = []

        if self.m is not None and self.c is not None:
            # Apply the line equation: y = mx + c
            predicted_values = self.m * predictor_x_test + self.c
            self.pred = predicted_values.tolist()
            return predicted_values.to_frame()

        # Fallback in case model is not trained
        return pd.DataFrame([])

    def score(self, target_y_test: pd.Series) -> float:
        """
        Evaluates the model performance using the R² (coefficient of determination) metric.

        Parameters:
            target_y_test (pd.Series): Actual target values

        Returns:
            float: R² score indicating model performance (1.0 is perfect)
        """
        row_num = len(target_y_test)
        ssr: float = 0  # Sum of squares of residuals
        sst: float = 0  # Total sum of squares
        mean_y = target_y_test.mean()

        for i in range(row_num):
            ssr += (target_y_test.iloc[i] - self.pred[i]) ** 2
            sst += (target_y_test.iloc[i] - mean_y) ** 2

        r_squared = 1 - (ssr / sst)
        print(f"The R² score of the model is: {r_squared}")
        return r_squared

    def plot_regression(self, x_test: pd.Series, y_test: pd.Series):
        if self.m is None or self.c is None:
            print("Model is not trained yet.")
            return

        # Get predicted values
        y_pred = pd.Series(self.pred, index=x_test.index)

        # Plot actual points
        plt.scatter(x_test, y_test, color='blue', label='Actual')

        # Plot regression line
        line_x = np.linspace(min(x_test), max(x_test), 100)
        line_y = self.m * line_x + self.c
        plt.plot(line_x, line_y, color='red', label='Regression Line')

        # Plot residuals
        for i in range(len(x_test)):
            plt.plot([x_test.iloc[i], x_test.iloc[i]],
                    [y_test.iloc[i], y_pred.iloc[i]],
                    color='green', linestyle='--', linewidth=1)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression with Residuals')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ===========================
# Example usage
# ===========================

data = {
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "y": [2 * x + 1 for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]  # [3, 5, 7, ..., 21]
}

test_data = {
    "x": [1.5, 3.5, 5.5, 7.5, 9.5],
    "y": [2 * x + 1 for x in [1.5, 3.5, 5.5, 7.5, 9.5]]  # [4.0, 8.0, 12.0, ...]
}



# Convert to DataFrames
df_train = pd.DataFrame(data)
df_test = pd.DataFrame(test_data)

# Separate features and labels
x_train = df_train["x"]
y_train = df_train["y"]
x_test = df_test["x"]
y_test = df_test["y"]

# Initialize and train the model
model = Linear_regression()
model.fit_model(x_train, y_train)

# Make predictions
predicted_vals = model.predict(x_test)

print(f"\nexpected values: \n{y_test}")
print(f"\nPredicted Values: \n{predicted_vals}")

# Evaluate the model
model.score(y_test)
model.plot_regression(x_test,y_test)