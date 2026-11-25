# High School & Beyond (HSB) Analytics Workflow

An interactive, end-to-end data science and machine learning web application built with **Python** and **Streamlit**. 

This dashboard takes the classic UCLA High School & Beyond dataset (`n=200`) and provides a complete analytical workflow: from data quality checks and exploratory data analysis (EDA) to statistical inference, supervised machine learning (regression & classification), and unsupervised learning (clustering & dimensionality reduction).

## 🚀 Features
* **Data Quality:** Missingness checks, data dictionary, and descriptive summaries.
* **EDA:** 9+ interactive Plotly visualizations (distributions, boxplots, correlations, scatterplots).
* **Statistical Analysis:** t-tests, ANOVA, Kruskal-Wallis, Chi-square, Pearson/Spearman correlations, and OLS regression diagnostics (VIF, Breusch-Pagan).
* **Supervised ML (Regression):** 19 models including XGBoost, LightGBM, CatBoost, Random Forest, SVR, and more. Includes cross-validation, model leaderboards, residuals, and SHAP interpretability.
* **Supervised ML (Classification):** 17 models predicting median-split binary outcomes. Includes ROC curves and confusion matrices.
* **Unsupervised ML:** 
  * **Clustering:** K-Means, DBSCAN, GMM, Spectral, etc.
  * **Dim. Reduction:** PCA, t-SNE, UMAP, ICA, Factor Analysis.
* **Association Rules:** Apriori and FP-Growth algorithms.
* **Responsible AI:** Built-in guidance on interpretation and limitations (documenting why deep learning/RL are excluded for this tabular dataset).

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Data & Viz:** Pandas, NumPy, Plotly
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, CatBoost
* **Statistics:** SciPy, Statsmodels, Pingouin
* **Interpretability:** SHAP

## 💻 Run Locally
1. Clone the repository:
   `git clone https://github.com/ojwangmaxwell3-ux/hsb-analytics-app.git`
2. Navigate into the directory:
   `cd hsb-analytics-app`
3. Create a virtual environment and activate it:
   `python -m venv .venv`
   `source .venv/Scripts/activate` (Windows)
4. Install dependencies:
   `pip install -r requirements.txt`
5. Run the app:
   `streamlit run app.py`

## 📊 Dataset Source
The dataset (`hsb2.csv`) is a public teaching dataset from the UCLA Statistical Consulting Group.
