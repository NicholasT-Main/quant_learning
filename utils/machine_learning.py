import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # Example model - replace with your actual model

class WalkForwardPredictor:
    def __init__(self, model, min_train_size=252, retrain_frequency=1):
        """
        Walk-forward predictor for time series

        Args:
            model: The ML model to use (should have fit() and predict() methods)
            min_train_size: Minimum number of samples to start training
            retrain_frequency: How often to retrain (1 = retrain after every prediction)
        """
        self.model = model
        self.min_train_size = min_train_size
        self.retrain_frequency = retrain_frequency
        self.predictions = []
        self.actual_values = []
        self.prediction_dates = []
        self.errors = []
        self.models_history = []  # Store model states if needed

    def predict_and_learn(self, X, y):
        """
        Perform walk-forward prediction and learning

        Args:
            X: Features dataframe
            y: Target series
        """
        predictions = []
        actual_values = []
        prediction_dates = []
        errors = []

        # Start from min_train_size
        for i in range(self.min_train_size, len(X)):
            # Get training data up to current point
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]

            # Train the model (or retrain based on frequency)
            if i == self.min_train_size or (i - self.min_train_size) % self.retrain_frequency == 0:
                #print(f"Training model at step {i} with {len(X_train)} samples...")
                self.model.fit(X_train, y_train)

            # Predict the next value
            X_current = X.iloc[i:i+1]  # Current features
            prediction = self.model.predict(X_current)[0]

            # Get actual value
            actual = y.iloc[i]

            # Calculate error
            error = actual - prediction

            # Store results
            predictions.append(prediction)
            actual_values.append(actual)
            prediction_dates.append(X.index[i])
            errors.append(error)

            # Optional: Print progress
            #if i % 50 == 0:
            #    print(f"Step {i}: Predicted {prediction:.4f}, Actual {actual:.4f}, Error {error:.4f}")

        # Store results
        self.predictions = np.array(predictions)
        self.actual_values = np.array(actual_values)
        self.prediction_dates = prediction_dates
        self.errors = np.array(errors)

        return self.predictions, self.actual_values, self.prediction_dates

def plot_walkforward_results(predictor, prices, target_stock='ASML'):
    """Plot the walk-forward results"""

    # Get full price series for context
    full_prices = prices

    # Calculate prices for prediction period
    pred_start_idx = predictor.min_train_size
    start_price = full_prices.iloc[pred_start_idx - 1]

    # Convert predictions and actuals to prices
    actual_prices = []
    predicted_prices = []

    current_actual_price = start_price
    current_pred_price = start_price

    for i, (actual_return, pred_return) in enumerate(zip(predictor.actual_values, predictor.predictions)):
        # Update actual prices
        current_actual_price = current_actual_price * (1 + actual_return)
        actual_prices.append(current_actual_price)

        # Update predicted prices
        current_pred_price = current_pred_price * (1 + pred_return)
        predicted_prices.append(current_pred_price)

    # Create plots
    plt.figure(figsize=(15, 12))

    # Plot 1: Full price history with prediction period highlighted
    plt.subplot(3, 1, 1)
    plt.plot(full_prices.index, full_prices.values, label=f'{target_stock} Full History',
             linewidth=2, alpha=0.8, color='blue')

    # Highlight prediction period
    pred_start_date = predictor.prediction_dates[0]
    plt.axvline(x=pred_start_date, color='red', linestyle='--', alpha=0.7,
                label='Walk-Forward Start')

    plt.title(f'{target_stock} - Full Price History with Walk-Forward Period',
              fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Walk-forward predictions vs actual
    plt.subplot(3, 1, 2)
    plt.plot(predictor.prediction_dates, actual_prices,
             label='Actual Prices', color='blue', linewidth=2, marker='o', markersize=2)
    plt.plot(predictor.prediction_dates, predicted_prices,
             label='Predicted Prices', color='red', linewidth=2, linestyle='--', marker='s', markersize=2)

    plt.title('Walk-Forward: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Rolling error analysis
    plt.subplot(3, 1, 3)

    # Plot prediction errors
    plt.plot(predictor.prediction_dates, predictor.errors,
             label='Prediction Errors', color='red', alpha=0.7, linewidth=1)

    # Plot rolling average error
    window = 20
    rolling_error = pd.Series(predictor.errors).rolling(window=window).mean()
    plt.plot(predictor.prediction_dates, rolling_error,
             label=f'{window}-day Rolling Avg Error', color='black', linewidth=2)

    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Return Error')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def analyze_walkforward_performance(predictor):
    """Analyze the performance of walk-forward predictions"""

    # Return-based metrics
    return_mse = mean_squared_error(predictor.actual_values, predictor.predictions)
    return_rmse = np.sqrt(return_mse)
    return_mae = np.mean(np.abs(predictor.errors))

    # Direction accuracy
    actual_directions = np.sign(predictor.actual_values)
    pred_directions = np.sign(predictor.predictions)
    direction_accuracy = np.mean(actual_directions == pred_directions)

    # Price-based analysis
    start_price = 1.0
    actual_final_price = start_price * np.prod(1 + predictor.actual_values)
    pred_final_price = start_price * np.prod(1 + predictor.predictions)

    print("=" * 60)
    print("WALK-FORWARD PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"Number of predictions: {len(predictor.predictions)}")
    print(f"Prediction period: {predictor.prediction_dates[0]} to {predictor.prediction_dates[-1]}")
    print()

    print("RETURN-BASED METRICS:")
    print(f"MSE: {return_mse:.6f}")
    print(f"RMSE: {return_rmse:.6f}")
    print(f"MAE: {return_mae:.6f}")
    print(f"Direction Accuracy: {direction_accuracy:.2%}")
    print()

    print("CUMULATIVE PERFORMANCE:")
    print(f"Actual total return: {(actual_final_price - 1) * 100:.2f}%")
    print(f"Predicted total return: {(pred_final_price - 1) * 100:.2f}%")
    print(f"Return difference: {((pred_final_price - actual_final_price) / actual_final_price) * 100:.2f}%")
    print()

    print("ERROR STATISTICS:")
    print(f"Mean error: {np.mean(predictor.errors):.6f}")
    print(f"Std error: {np.std(predictor.errors):.6f}")
    print(f"Min error: {np.min(predictor.errors):.6f}")
    print(f"Max error: {np.max(predictor.errors):.6f}")

    return {
        'mse': return_mse,
        'rmse': return_rmse,
        'mae': return_mae,
        'direction_accuracy': direction_accuracy,
        'actual_return': (actual_final_price - 1) * 100,
        'predicted_return': (pred_final_price - 1) * 100
    }
