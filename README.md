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




