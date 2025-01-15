import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

try:
    # MODULARIZE code functions for organizational purposes.
    
    # Fetch S&P 500 data
    ticker = "^GSPC"  # S&P 500 Index
    data = yf.download(ticker, start="2000-01-01", end="2023-12-31", interval="1d")

    # Fill missing values
    data.ffill(inplace=True)  # Forward-fill missing values
    data.bfill(inplace=True)  # Backward-fill missing values

    # Add new features
    data['Returns'] = data['Close'].pct_change()  # Daily returns
    data['Volatility'] = data['Returns'].rolling(window=10).std()  # 10-day volatility
    data['MA_5'] = data['Close'].rolling(window=5).mean()  # 5-day moving average
    data['MA_10'] = data['Close'].rolling(window=10).mean()  # 10-day moving average

    # Drop rows with NaN after feature creation
    data.dropna(inplace=True)

    # Define the target: Crash = 1 if Returns < -2%, else 0
    data['Crash'] = (data['Returns'] < -0.02).astype(int)

    # Save the cleaned dataset to a CSV file
    data.to_csv("S&P500_Preprocessed.csv")

    print("Preprocessed data saved as 'S&P500_Preprocessed.csv'.")
    print(data.head())

    # Save and display S&P 500 historical data plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label="S&P 500 Close Price")
    plt.title("S&P 500 Historical Data")
    plt.legend()
    plt.savefig("sp500_historical_data.png")
    plt.show()

    # Save crash distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Crash'], bins=2, kde=False)
    plt.title('Distribution of Crashes')
    plt.savefig("crash_distribution.png")
    plt.show()

    # Save correlation matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig("correlation_matrix.png")
    plt.show()

    print("Training the logistic regression model...")

    # Define features and target
    X = data[['Returns', 'Volatility', 'MA_5', 'MA_10']]  # Feature columns
    y = data['Crash']  # Target column

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reset the index for test set
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Extract returns for the test set from the original data
    returns = data['Returns'].iloc[-len(y_test):].reset_index(drop=True)  # Match the size of X_test

    print("Training data size:", X_train.shape)
    print("Test data size:", X_test.shape)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train_scaled, y_train)

    print("Model training completed.")
    print("Evaluating the model...")

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Display the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Save the model and scaler
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Define backtesting strategy
    def backtest_strategy(predictions, actuals, returns):
        investment = 100000  # Initial investment
        history = [investment]  # Track investment value over time
        for pred, actual, ret in zip(predictions, actuals, returns):
            if pred == 1:  # Predicted crash
                investment *= 1  # Move to cash, no change
            else:
                investment *= (1 + ret)  # Stay invested
            history.append(investment)
        return investment, history

    # Backtest and log results
    final_investment, investment_history = backtest_strategy(y_pred, y_test.values, returns)
    print(f"Final portfolio value after backtesting: ${final_investment:.2f}")

    # Save investment history plot
    plt.figure(figsize=(10, 6))
    plt.plot(investment_history, label="Portfolio Value Over Time")
    plt.title("Backtesting Results")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.savefig("backtesting_results.png")
    plt.show()

    # Log performance metrics to a file
    with open("model_performance.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write(f"\nFinal portfolio value: ${final_investment:.2f}")

    print("Model evaluation completed.")
    print(f"Final portfolio value after backtesting: ${final_investment:.2f}")

    # Final Checklist
    print("Final Checklist:")
    print("1. Preprocessed Data: Ensure that S&P500_Preprocessed.csv is created and contains clean, feature-rich data.")
    print("2. Plots: Confirm that saved plots (e.g., sp500_historical_data.png, crash_distribution.png) are clear and informative.")
    print("3. Model Performance: Verify the classification report and backtesting results.")
    print("4. Logs: Ensure the script logs key progress and results for clarity.")

    # Optional: Quick test run with a subset of data
    TEST_RUN = False  # Set to True for quick tests

    if TEST_RUN:
        data = data.sample(frac=0.1, random_state=42)  # Use 10% of data for faster runs

except KeyboardInterrupt:
    print("\nScript interrupted by user. Exiting gracefully.")
    exit(0)

except Exception as e:
    print(f"An error occurred: {e}")
