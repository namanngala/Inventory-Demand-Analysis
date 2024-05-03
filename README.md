![image](https://github.com/Kartikkeyy/Inventory_Demand_Analysis_ML/assets/107746395/839f2950-da1c-47d2-ad38-48a078948c87)
# Inventory_Demand_Analysis_ML

Our Inventory Demand Prediction Model is a machine learning solution designed to forecast the inventory requirements for a retail store. The model utilizes historical sales data to predict the demand for store items, aiding in inventory management and optimization.

## Introduction
Forecasting is a critical tool for businesses, enabling them to analyze data, estimate outcomes, and make informed decisions. Time series analysis helps identify patterns and trends, but the growing volume of data presents challenges. Machine learning techniques enhance forecasting by analyzing vast datasets and uncovering hidden insights. However, accuracy depends on data quality and model efficacy.

## Background
**Implementing ARIMA models** with ML techniques in retail inventory demand forecasting offers several benefits. It allows retailers to:

**Predict future demand with greater accuracy:** By incorporating ML algorithms, retailers can improve the accuracy of their demand forecasts, leading to better inventory management decisions.<br><br>
**Adapt to changing market conditions:** ML techniques enable retailers to quickly adjust their forecasting models in response to changes in consumer behavior, market trends, and external factors.

**Optimize inventory levels:** Accurate demand forecasts help retailers maintain optimal inventory levels, minimizing stockouts and excess inventory costs.

**Improve customer satisfaction:** By ensuring product availability and timely replenishment, retailers can enhance the shopping experience for their customers, leading to increased loyalty and retention.


## Proposed Idea
In our project to predict stock values from historical data (2013-2017), we're leveraging a combination of traditional time series models like ARIMA/SARIMA and machine learning (ML) algorithms. **ARIMA and SARIMA are adept at capturing trends and seasonal patterns in time series data**, while ML models like linear regression, ensemble methods, and KNN regressors offer diverse approaches to predict stock values based on historical trends and market factors.

Here's a brief overview of how each model contributes:

1. **Linear Regression:** Provides a straightforward analysis of the relationship between historical stock prices and future values, serving as a baseline model.

2. **Ridge and Lasso Regression:** Useful for handling high-dimensional data and multicollinearity, preventing overfitting, and improving model performance.

3. **K-Nearest Neighbors (KNN) Regressor:** A non-parametric model that captures complex relationships, ideal for identifying local patterns or outliers.

4. **Ensemble Methods (Gradient Boosting, Extra Trees, Random Forest, XGBoost):** Excel at capturing complex interactions and nonlinear relationships, offering improved predictive performance for potentially large datasets.

5. **Decision Tree Regressor:** Easy to interpret and captures nonlinear relationships, aiding in feature selection and identifying important predictors.

Each model serves a unique purpose, from providing a simple baseline to capturing intricate data interactions. Through experimentation and evaluation, we aim to identify the best approach for accurately predicting stock values while considering factors like interpretability and computational efficiency.

## Implementation
![image](https://github.com/Kartikkeyy/Inventory_Demand_Analysis_ML/assets/107746395/4f93bcaa-af56-4a95-8141-f1dce82b3022)
<br>
Link of flowchart: 
https://boardmix.com/app/share/CAE.CLPGHSABKhBL50COWaAli703L1PBFUeQMAZAAQ/GnZ7qX，

<br>
**Project Overview:**

Our project focuses on predicting stock values using historical data from January to March spanning 2013 to 2017. Here's a concise breakdown of our approach:

1. **Data Exploration:** We begin by exploring the dataset's structure and patterns using methods like df.info(), df.describe(), and visualization tools.<br>
Dataset: https://www.kaggle.com/c/demand-forecasting-kernels-only/data
2. **Data Preprocessing:** We enrich the dataset by incorporating information about weekends and weekdays, adding binary features to distinguish between them.

3. **Feature Engineering:** We extract insights from datetime variables, create lag features, and encode holiday information to enrich the dataset and improve model performance.

4. **Model Selection and Training:** We carefully select regression algorithms like Linear Regression, Ridge Regression, and Decision Tree Regression, among others. These models are trained and evaluated using appropriate metrics like MAPE and R-squared score.

5. **Model Evaluation:** We assess each model's performance using regression metrics to identify the best-performing one for further analysis.

6. **Hyperparameter Tuning:** We fine-tune the selected model's hyperparameters using techniques like GridSearchCV to optimize its performance.

7. **Prediction and Visualization:** Finally, we use the trained model to predict stock values for the test data from 2017. These predictions are visualized alongside the actual values to assess the model's accuracy and reliability.<br>
![image](https://github.com/Kartikkeyy/Inventory_Demand_Analysis_ML/assets/107746395/314be4c6-eb5e-4ed7-bec8-6828a530d9ef)


Our approach integrates data exploration, preprocessing, feature engineering, model selection, and evaluation to deliver accurate stock value predictions. By combining traditional time series models with machine learning algorithms, we aim to uncover valuable insights and improve forecasting accuracy.
<br>

## Result
After analyzing data from 2013 to 2016 and testing on 2017, we used various regression algorithms to forecast inventory demand for the initial three months of 2018. Here are the outcomes:

1. **KNeighbors Regressor:** Average error rate: 0.21%. Captured localized patterns leveraging neighboring data points.

2. **Extra Trees Regressor:** Average error rate: 0.24%. Effectively captured complex relationships by training multiple decision trees.

3. **Random Forest Regressor:** Average error rate: 0.22%. Provided robust predictions by aggregating predictions from multiple trees.

4. **Decision Tree Regressor:** Average error rate: 0.28%. Limited by overfitting to training data.

5. **Gradient Boosting Regressor:** Average error rate: 0.19%. Sequentially trained weak learners and corrected errors for improved accuracy.

6. **Linear Regression:** Average error rate: 0.20%. Provided interpretable results but may be limited by assuming linear relationships.

7. **XGBRegressor:** Average error rate: 0.23%. Employed a gradient boosting framework for accurate predictions.

8. **Lasso Regression:** Average error rate: 0.22%. Identified relevant features for improved accuracy.

9. **Ridge Regression:** Average error rate: 0.20%. Introduced regularization to mitigate overfitting.

The Gradient Boosting regressor yielded the best results with the lowest average error rate, making it our final model for inventory demand forecasting. Its sequential training of weak learners and error correction mechanism demonstrated superior performance in capturing complex relationships and providing accurate predictions.<br>

**Confusion Matrix**
<br>
![image](https://github.com/Kartikkeyy/Inventory_Demand_Analysis_ML/assets/107746395/ef9b4196-9de4-4618-ac31-846ef137b278)
![image](https://github.com/Kartikkeyy/Inventory_Demand_Analysis_ML/assets/107746395/06a1eec2-9dc0-4f1e-9e51-90a0935add81)

**FINAL PREDICTION OF FIRST MONTHS OF 2018** <br>
![image](https://github.com/Kartikkeyy/Inventory_Demand_Analysis_ML/assets/107746395/83fbb48e-649c-4c88-8215-0106f5200e82)


## References 
https://www.investopedia.com/terms/a/autoregressive-integrated-moving-average-arima.asp

https://journals.sagepub.com/doi/full/10.1177/1847979018808673

https://www.zoho.com/inventory/guides/inventory-forecasting.html#:~:text=Inventory%20forecasting%20is%20a%20method,revenue%20and%20decrease%20unnecessary%20costs.

https://www.kaggle.com/datasets/talhanazir168/store-inventory-demand-forecasting-dataset/data?select=train.csv


## Steps to execute
1. git clone
2. import all the libraries
3. attach the dataset to collab file
4. execute

*Thankyou*
