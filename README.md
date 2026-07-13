#  End-to-End Sales Forecasting & Demand Intelligence System

##  Project Overview

This project is an end-to-end sales forecasting and demand intelligence system developed using Python and Machine Learning. The objective is to help retail businesses forecast future sales, detect unusual sales patterns, segment products based on demand, and provide business insights through an interactive Streamlit dashboard.

The project was developed as part of a Data Science Internship.

---

#  Problem Statement

Retail companies need accurate demand forecasting to avoid overstocking and stock shortages. This project predicts future sales using multiple forecasting models, identifies anomalies in sales patterns, clusters products by demand characteristics, and presents the results through an interactive dashboard for business decision-making.

---

#  Features

- Data Cleaning & Preprocessing
- Time Series Analysis
- Sales Forecasting
- Time Series Decomposition
- Stationarity Testing (ADF Test)
- SARIMA Forecasting
- Facebook Prophet Forecasting
- XGBoost Forecasting
- Model Comparison (MAE, RMSE, MAPE)
- Isolation Forest Anomaly Detection
- Z-Score Based Anomaly Detection
- Product Demand Segmentation using K-Means Clustering
- PCA Visualization
- Interactive Streamlit Dashboard

---

#  Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost
- Streamlit

---

#  Project Structure

```
SalesForecasting_Uzma/
│
├── analysis.ipynb
├── app.py
├── train.csv
├── requirements.txt
├── summary.pdf
├── README.md
│
└── charts/
      ├── monthly_sales_trend.png
      ├── decomposition.png
      ├── sarima_forecast.png
      ├── prophet_forecast.png
      ├── xgboost_forecast.png
      ├── anomaly_detection.png
      ├── product_clusters.png
      └── ...
```

---

#  Machine Learning Models

The following forecasting models were implemented and compared:

- SARIMA
- Facebook Prophet
- XGBoost Regressor

The best-performing model was selected based on evaluation metrics such as MAE, RMSE, and MAPE.

---

#  Streamlit Dashboard

The dashboard contains four pages:

###  Sales Overview
- Total Sales by Year
- Monthly Sales Trend
- Region Filter
- Category Filter

###  Forecast Explorer
- Forecast by Category
- Forecast by Region
- Forecast Horizon Selection
- Forecast Chart
- Model Performance

###  Anomaly Report
- Isolation Forest Visualization
- Detected Anomalies
- Sales Anomaly Table

###  Product Demand Segments
- K-Means Clustering
- PCA Cluster Visualization
- Demand Cluster Table
- Business Recommendations

---

#  How to Run

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Streamlit Dashboard

```bash
streamlit run app.py
```

---

#  Dataset

- Superstore Sales Dataset (train.csv)
- Video Game Sales Dataset (vgsales.csv)

---

##  Live Demo

**Streamlit Dashboard:**  
https://uzmafarzeen-end-to-end-sales-forecasting-demand-inte-app-gag7fr.streamlit.app

**GitHub Repository:**  
https://github.com/UZMAFARZEEN/End-to-End-Sales-Forecasting-Demand-Intelligence-System

---

#  Key Outcomes

- Forecasted future sales using three different forecasting models.
- Identified unusual sales spikes and drops using anomaly detection.
- Segmented products into demand groups using clustering.
- Developed an interactive dashboard for business users.
- Generated actionable business recommendations for inventory planning.

---

#  Author

**Syed Uzma Farzeen**

B.Tech – Information Technology

Data Science & Machine Learning Enthusiast

---

#  Acknowledgement

This project was completed as part of a Data Science Internship to demonstrate practical skills in Time Series Forecasting, Machine Learning, Data Visualization, and Business Analytics.
