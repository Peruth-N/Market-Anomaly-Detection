# Market Anomaly Detection System

## **Overview**
This project aims to develop an anomaly detection system that serves as an early warning mechanism for identifying potential financial market crashes before they occur. By leveraging machine learning models, the system classifies market conditions and provides actionable investment strategies to minimize losses and optimize returns. The project has been implemented in stages and includes detailed visualizations and a binary classification model to predict market crashes.


## **Milestones Achieved**

### **Milestone 1: Develop an Anomaly Detection Model**
1. **Data Substitution:**
   - The dataset provided initially was problematic and returned an empty DataFrame after preprocessing.
   - Replaced the dataset with real-time financial data fetched using the `yfinance` library, specifically focusing on the S&P 500 Index (`^GSPC`).

2. **Data Preprocessing and Feature Engineering:**
   - Computed key features such as:
     - **Daily Returns:** Percentage change in the closing price.
     - **Volatility:** 10-day rolling standard deviation of returns.
     - **Moving Averages:** 5-day and 10-day moving averages of the closing price.
   - Defined a binary target variable (`Crash`): `1` for returns less than -2%, otherwise `0`.
   - Visualized data with:
     - Crash distribution.
     - Correlation matrix.

3. **Model Training:**
   - Trained a logistic regression model to classify market conditions as "Crash" or "No Crash."
   - Achieved high accuracy with the following evaluation metrics:
     - **Precision, Recall, F1-Score:** Highlighted the model's ability to predict crashes accurately.
     - **Confusion Matrix:** Provided insight into true and false predictions.

### **Milestone 2: Investment Strategy Proposal**
1. **Backtesting Investment Strategy:**
   - Simulated an investment strategy based on model predictions:
     - **Crash Prediction:** Move to cash to minimize losses.
     - **No Crash Prediction:** Stay invested to maximize returns.
   - Calculated the final portfolio value and visualized the investment's performance over time.

2. **Outputs:**
   - **Plots:**
     - Historical S&P 500 close price.
     - Crash distribution.
     - Backtesting results showing portfolio performance.
   - Model and scaler files were saved for future use in predictions.

### **Milestone 3: AI-Driven Bot Integration**
- **Status:**
   - Began integrating an interactive bot using the **Rasa** framework to explain the investment strategy to end users.
   - Encountered challenges related to implementing dynamic user inputs and prediction integration.
   - Continued working on improving the bot for real-time interaction.


## **Challenges Overcome**
1. **Dataset Issues:**
   - The original dataset provided by Headstarter returned an empty DataFrame after preprocessing.
   - Substituted the problematic dataset with reliable financial data fetched via `yfinance`.

2. **Limited Financial Background:**
   - Overcame a lack of familiarity with financial concepts by researching and applying domain-specific knowledge to feature engineering and model evaluation.

3. **Technical Debugging:**
   - Addressed and resolved `IndexError` and data alignment issues during backtesting.
   - Improved visualization handling by saving plots and removing redundant blocking mechanisms.


## **Skills Utilized**
- **Programming:** Python (data manipulation, visualization, and machine learning).
- **Libraries and Frameworks:**
  - `yfinance`: Fetching real-time financial data.
  - `pandas`: Data preprocessing and feature engineering.
  - `matplotlib`, `seaborn`: Data visualization.
  - `scikit-learn`: Model training, evaluation, and backtesting.
  - `joblib`: Model persistence for future use.
  - **Rasa (in progress):** Bot development for explaining strategies interactively.
    

## **Key Insights and "Aha" Moments**
- **Real-Time Data Utilization:** Transitioning to `yfinance` not only resolved data issues but also introduced flexibility for real-time predictions.
- **Feature Engineering:** Learning the importance of financial metrics such as returns, volatility, and moving averages in crash prediction.
- **Backtesting:** Simulating investment strategies provided a practical understanding of how machine learning can aid decision-making in financial markets.
- **Visualization:** Creating clear and informative plots significantly improved understanding and presentation of results.


## **Future Iterations**
1. **Enhancing the Interactive Bot:**
   - Fully integrate the Rasa bot with the trained model for real-time strategy recommendations.
   - Allow user-specific inputs (e.g., custom returns, volatility) to tailor predictions.

2. **Expanding Models:**
   - Experiment with advanced models like Random Forest, Gradient Boosting, or Neural Networks to improve crash prediction accuracy.

3. **Deployment:**
   - Deploy the bot and model as a web or mobile application for broader accessibility.

4. **Improved Backtesting:**
   - Incorporate transaction costs and other real-world factors for more accurate simulations.



Thank you for reviewing my project. I am excited to continue learning and improving this system to make it more dynamic and accessible!

