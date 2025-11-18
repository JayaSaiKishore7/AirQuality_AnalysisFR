# 🌫️ Air Quality Analysis — Machine Learning & EDA Project

This project analyzes air quality measurements, performs exploratory data analysis (EDA), implements a data-cleaning pipeline, and prepares datasets for machine learning. It uses real-world datasets collected from French air-quality monitoring stations.

---

## Project Features
- Complete Exploratory Data Analysis (EDA) workflow  
- Full data cleaning pipeline with missing/zero-value handling  
- Cleaned dataset stored in `data/processed`  
- Professional, reproducible project structure  
- Beginner-friendly notebook and scripts  
- Ready for machine learning modeling

---
## Setup

### 1. Create Conda environment:
```bash
conda env create -f environment.yml
```

### 2 Activate the environment
```bash
conda activate airquality-ml
```
## 📂 Project Structure

```
AirQuality-Project/
│
├── data/
│   ├── raw/          # Original unmodified datasets (NOT uploaded to GitHub)
│   ├── sample/       # Small extracted samples used for EDA
│   └── processed/    # Cleaned datasets ready for ML
│
├── notebooks/
│   └── EDA.ipynb     # Main notebook: exploration, cleaning, visualization
│
├── src/
│   ├── cleaning.py   # Data cleaning logic (optional)
│   ├── features.py   # Feature engineering utilities (optional)
│   └── model.py      # ML model training/evaluation (optional)
│
├── .gitignore        # Ensures datasets are excluded from GitHub
├── environment.yml   # Environment dependencies
└── README.md         # Project documentation (this file)
```

## 📊 Exploratory Data Analysis (EDA)

The notebook `notebooks/EDA.ipynb` includes:

- **Dataset overview:** shape, columns, data types  
- **Cleaning and normalizing** column names  
- **Missing value** analysis and handling  
- **Zero-value checks** and rationale  
- **Summary statistics** for numeric and categorical features  
- **Unique value counts** and cardinality checks  
- **Time-based features:** hour, day, month  
- **Station-level** analysis  
- **Pollutant-level** analysis (NO2, PM10, O3, etc.)  
- **Visualizations:** histograms, boxplots, line trends, time-series plots


## 🧹 Data Cleaning Pipeline

This project follows a structured cleaning workflow:

1. Drop unnecessary or redundant columns  
2. Remove rows with missing `valeur`  
3. Remove zero-value pollution readings  
4. Convert date columns into proper datetime format  
5. Remove rows with invalid date ranges  
6. Remove duplicates  
7. Remove negative pollution values  
8. Save the cleaned dataset into `data/processed/`  

This ensures the dataset is **consistent, reliable, and ready for ML**.



## 🤖 Machine Learning (Planned)

Future tasks include:

- Regression models to predict pollution (`valeur`)  
- Time-series forecasting (LSTM, ARIMA, Prophet)  
- Pollution classification models  
- Feature engineering (rolling averages, lag features, AQI)  
- Model evaluation (RMSE, MAE, R²)  
- Hyperparameter tuning  



## 📥 Data Handling Philosophy

- Raw data (`data/raw`) is **never modified**  
- Sample data (`data/sample`) is used only for quick EDA  
- Processed data (`data/processed`) is created via code  
- Data folders are excluded from GitHub using `.gitignore`  
- All transformations are performed in reproducible code  



## ▶️ How to Run

1. Place the raw CSV file in:
2. Open the notebook:
3. Run the notebook to:
- Perform EDA  
- Generate sample dataset  
- Produce cleaned dataset inside:



## ⭐ Future Improvements

- Build an interactive dashboard (Streamlit / Dash)  
- Integrate real-time AQI APIs  
- Add MLOps pipeline (DVC, MLflow)  
- Automatic reporting and data validation  
- Add multi-city comparison and mapping  






