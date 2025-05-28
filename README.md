# Honeypot-Based Vehicular Attack Detection

![Tests](https://github.com/AnasHaroun/attackRate_for_attack_prediction/actions/workflows/python-tests.yml/badge.svg)

This repository provides a modular machine learning pipeline for detecting attacks in vehicular networks using behavioral data and honeypot analysis. It includes data preprocessing, feature extraction, trust analysis, model training, and test automation.

---

## 📂 Files Overview

| File                                | Description                                                             |
|-------------------------------------|-------------------------------------------------------------------------|
| `attackrate_ml_honeypot_refactored.py` | Main ML script: preprocessing, feature extraction, modeling, analysis  |
| `test_attackrate_ml_honeypot.py`   | Unit tests for core functionality                                       |
| `.github/workflows/python-tests.yml` | GitHub Actions workflow for automated test execution                   |

---

## 🚀 Features

- Parses simulation logs (JSON/CSV) into features
- Normalizes and labels known attack types
- Extracts trust and behavior metrics
- Classifies vehicular attacks using Random Forest and Gradient Boosting
- Identifies benign senders and candidates for honeypots
- Fully tested and CI/CD-ready

---

## 🧠 Requirements

Install the dependencies:
```bash
pip install pandas scikit-learn
