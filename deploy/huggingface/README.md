---
title: PACA Air Quality Forecast
emoji: 🌍
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
pinned: false
short_description: Daily 24h air pollution forecasts for the PACA region
tags:
  - air-quality
  - forecasting
  - xgboost
  - mlops
---

Live 24-hour pollutant forecasts for ATMO SUD (PACA, France) monitoring stations.

Every day a GitHub Actions pipeline pulls the latest Geod'Air measurements, runs an XGBoost
forecasting model, checks for data drift with Evidently, and publishes this page.

Source code, tests and pipeline: https://github.com/JayaSaiKishore7/AirQuality_AnalysisFR
