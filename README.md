# 🧠 Advanced Data Science — Store Sales Prediction

This project aims to implement machine learning models to **predict store sales** using historical data and key features such as store type, promotions, etc.  
It’s designed as a collaborative data science workflow with full environment reproducibility via Conda.

---

## 🎓 Course Information

**Course:** CAP 3764 — *Advanced Data Science*  
**Institution:** [Florida International University]  
**Team Members:**  
- Luis D. Jimenez  
- Carlos Hernandez  

---

## 📋 Project Overview

The goal of this project is to:
- Explore and clean raw store sales data  
- Engineer predictive features  
- Train and evaluate machine learning models (e.g., Linear Regression, Random Forests)  
- Visualize results and performance metrics  

---

## 🧰 Tech Stack

- **Language:** Python 3.11  
- **Environment:** Conda  
- **Libraries:**
  - [pandas](https://pandas.pydata.org/)
  - [scikit-learn](https://scikit-learn.org/)
  - [matplotlib](https://matplotlib.org/)

---

## ⚙️ Environment Setup (Using `environment.yml`)

This project uses a **Conda environment** defined in a `.yml` file to ensure all team members can recreate the same setup seamlessly.

### 1️⃣ Create the environment from the `.yml` file
Make sure the `environment.yml` file is located in the project’s root directory, then run:
```bash
conda env create -f environment.yml