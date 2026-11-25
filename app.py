import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                   BayesianRidge, HuberRegressor, QuantileRegressor,
                                   LogisticRegression, SGDClassifier, Perceptron)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier,
                               AdaBoostRegressor, AdaBoostClassifier,
                               ExtraTreesClassifier)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             precision_recall_curve, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN, MeanShift,
                             SpectralClustering, OPTICS, AffinityPropagation)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, TruncatedSVD
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
try:
    import pingouin as pg
    HAS_PG = True
except ImportError:
    HAS_PG = False
try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMED = True
except ImportError:
    HAS_KMED = False


# =========================================================
# SETUP
# =========================================================
st.set_page_config(page_title="HSB Analytics Workflow", layout="wide")
st.title("High School & Beyond | Analytical Workflow")
st.caption("Data quality · EDA · Statistical inference · Supervised & unsupervised ML · Interpretation")

@st.cache_data
def load_default():
    return pd.read_csv("hsb2.csv")

uploaded = st.sidebar.file_uploader("Upload a replacement CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded uploaded file ({len(df)} rows)")
else:
    df = load_default()
    st.sidebar.info(f"Using default dataset ({len(df)} rows)")

REQUIRED = ["id","female","race","ses","schtyp","prog","read","write","math","science","socst"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

SCORE_COLS = ["read","write","math","science","socst"]
GROUP_COLS = ["female","prog","ses","schtyp","race"]

VARIABLE_DICT = pd.DataFrame({
    "Column": REQUIRED,
    "Type": ["int","binary","category","category","category","category","score","score","score","score","score"],
    "Description": [
        "Student ID","Female (1) / Male (0)","Race (1-4)","SES (1=low, 2=middle, 3=high)",
        "School type (1=public, 2=private)","Programme (1=general, 2=academic, 3=vocational)",
        "Reading score","Writing score","Math score","Science score","Social studies score",
    ],
})

tabs = st.tabs([
    "Data Quality", "EDA", "Statistical Analysis",
    "Regression (Supervised)", "Classification (Supervised)",
    "Clustering (Unsupervised)", "Dim. Reduction",
    "Association Rules", "Interpretation & Guidance"
])
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = tabs


# =========================================================
# TAB 1 — DATA QUALITY
# =========================================================
with tab1:
    st.subheader("Raw-data quality and missingness")
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    c1, c2 = st.columns(2)
    with c1:
        miss = df.isna().sum()
        miss = miss[miss > 0]
        if len(miss) == 0:
            st.success("No missing values detected.")
        else:
            st.plotly_chart(px.bar(x=miss.index, y=miss.values,
                labels={"x":"Column","y":"Missing"}, title="Missingness by column"),
                width='stretch')
    with c2:
        st.dataframe(pd.DataFrame({"Column": df.columns,
                     "dtype": df.dtypes.astype(str).values}), width='stretch')

    st.subheader("Variable dictionary")
    st.dataframe(VARIABLE_DICT, width='stretch')

    st.subheader("Descriptive summary (numeric)")
    st.dataframe(df.describe().T.round(2), width='stretch')

    st.subheader("Cleaned data preview")
    st.dataframe(df.head(20), width='stretch')
    st.download_button("Download cleaned CSV", df.to_csv(index=False),
                       "hsb2_clean.csv", "text/csv")


# =========================================================
# TAB 2 — EDA
# =========================================================
with tab2:
    st.subheader("Explore student attainment")
    st.caption("Descriptive patterns — not causal.")

    st.markdown("#### Score distribution")
    subj = st.selectbox("Subject for histogram", SCORE_COLS)
    st.plotly_chart(px.histogram(df, x=subj, nbins=20,
                    title=f"Distribution of {subj}"), width='stretch')

    st.markdown("#### Score by student group")
    c1, c2 = st.columns(2)
    gv = c1.selectbox("Group variable", GROUP_COLS)
    sv = c2.selectbox("Score variable", SCORE_COLS, index=2)
    st.plotly_chart(px.box(df, x=gv, y=sv, color=gv, points="all",
                    title=f"{sv} by {gv}"), width='stretch')

    st.markdown("#### Mean scores by subject")
    m = df[SCORE_COLS].mean().reset_index()
    m.columns = ["Subject","Mean"]
    st.plotly_chart(px.bar(m, x="Subject", y="Mean", text_auto=".1f"), width='stretch')

    st.markdown("#### Subject-score correlations")
    corr = df[SCORE_COLS].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Pearson correlation matrix"), width='stretch')

    st.markdown("#### Score density by group")
    c1, c2 = st.columns(2)
    dg = c1.selectbox("Group for density", GROUP_COLS, index=1)
    ds = c2.selectbox("Score for density", SCORE_COLS, index=3)
    st.plotly_chart(px.violin(df, x=dg, y=ds, color=dg, box=True, points="all",
                    title=f"Density of {ds} by {dg}"), width='stretch')

    st.markdown("#### Subject-to-subject scatterplot")
    c1, c2 = st.columns(2)
    sx = c1.selectbox("X subject", SCORE_COLS, index=0)
    sy = c2.selectbox("Y subject", SCORE_COLS, index=2)
    st.plotly_chart(px.scatter(df, x=sx, y=sy, color=gv, trendline="ols",
                    title=f"{sx} vs {sy} (colored by {gv})"), width='stretch')

    st.markdown("#### Score variability by subject")
    v = df[SCORE_COLS].std().reset_index()
    v.columns = ["Subject","StdDev"]
    st.plotly_chart(px.bar(v, x="Subject", y="StdDev", text_auto=".1f"), width='stretch')

    st.markdown("#### Subject means by programme")
    pm = df.groupby("prog")[SCORE_COLS].mean().reset_index().melt(id_vars="prog",
                    var_name="Subject", value_name="Mean")
    st.plotly_chart(px.bar(pm, x="Subject", y="Mean", color="prog", barmode="group"),
                    width='stretch')

    st.markdown("#### Descriptive statistics")
    st.dataframe(df[SCORE_COLS].describe().T.round(2), width='stretch')


# =========================================================
# TAB 3 — STATISTICAL ANALYSIS (expanded)
# =========================================================
with tab3:
    st.subheader("Statistical inference & diagnostics")
    st.caption("P-values quantify evidence under a model; they do not establish causality.")

    st.markdown("### Group comparison")
    test_type = st.radio("Test family",
        ["t-test / Mann-Whitney", "ANOVA / Kruskal-Wallis", "Correlation", "Contingency (chi-square)"],
        horizontal=True)

    if test_type in ("t-test / Mann-Whitney", "ANOVA / Kruskal-Wallis"):
        c1, c2 = st.columns(2)
        outcome = c1.selectbox("Outcome (score)", SCORE_COLS, key="inf_out")
        group = c2.selectbox("Grouping", GROUP_COLS, key="inf_grp")
        grouped = df.groupby(group)[outcome]
        groups = [g.dropna().values for _, g in grouped]
        labels = [str(k) for k, _ in grouped]

        st.markdown("**Group sample sizes**")
        st.dataframe(pd.DataFrame({"Group": labels, "n": [len(g) for g in groups]}),
                     width='stretch')

        if len(groups) < 2:
            st.warning("Need ≥2 groups.")
        else:
            if test_type == "t-test / Mann-Whitney" and len(groups) == 2:
                t, p = stats.ttest_ind(groups[0], groups[1], equal_var=True)
                tw, pw = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                u, pu = stats.mannwhitneyu(groups[0], groups[1])
                d = (groups[0].mean() - groups[1].mean()) / np.sqrt((groups[0].var(ddof=1)+groups[1].var(ddof=1))/2)
                st.write(f"**Student t:** t = {t:.3f}, p = {p:.4g}")
                st.write(f"**Welch t:** t = {tw:.3f}, p = {pw:.4g}")
                st.write(f"**Mann-Whitney U:** U = {u:.1f}, p = {pu:.4g}")
                st.write(f"**Cohen's d:** {d:.3f}")
            elif test_type == "ANOVA / Kruskal-Wallis" and len(groups) >= 2:
                f, p = stats.f_oneway(*groups)
                h, ph = stats.kruskal(*groups)
                st.write(f"**One-way ANOVA:** F = {f:.3f}, p = {p:.4g}")
                st.write(f"**Kruskal-Wallis:** H = {h:.3f}, p = {ph:.4g}")
                # eta-squared
                grand = np.concatenate(groups)
                ss_between = sum(len(g)*(g.mean()-grand.mean())**2 for g in groups)
                ss_total = ((grand-grand.mean())**2).sum()
                eta2 = ss_between/ss_total if ss_total>0 else np.nan
                st.write(f"**Eta-squared:** {eta2:.3f}")
                if p < 0.05:
                    try:
                        tukey = pairwise_tukeyhsd(grand, np.concatenate([[l]*len(g) for l,g in zip(labels,groups)]))
                        st.text(str(tukey))
                    except Exception as e:
                        st.info(f"Tukey HSD unavailable: {e}")

            # Assumption panel
            st.markdown("**Assumptions**")
            lw, lp = stats.levene(*groups)
            st.write(f"Levene equal-variance: W = {lw:.3f}, p = {lp:.4g} — "
                     f"{'variances differ' if lp<0.05 else 'similar'}")
            for lbl, g in zip(labels, groups):
                if len(g) >= 3:
                    sw, sp = stats.shapiro(g)
                    st.write(f"Shapiro-Wilk **{lbl}**: W = {sw:.3f}, p = {sp:.4g} — "
                             f"{'not normal' if sp<0.05 else 'consistent with normal'}")

            st.plotly_chart(px.box(df, x=group, y=outcome, color=group, points="all"),
                            width='stretch')

    elif test_type == "Correlation":
        st.markdown("**Pairwise correlation with p-values**")
        rows = []
        for i, a in enumerate(SCORE_COLS):
            for b in SCORE_COLS[i+1:]:
                r, p = stats.pearsonr(df[a], df[b])
                rho, prho = stats.spearmanr(df[a], df[b])
                tau, ptau = stats.kendalltau(df[a], df[b])
                rows.append({"Pair": f"{a}–{b}", "Pearson r": r, "p (P)": p,
                             "Spearman ρ": rho, "p (S)": prho,
                             "Kendall τ": tau, "p (K)": ptau})
        st.dataframe(pd.DataFrame(rows).round(3), width='stretch')

    elif test_type == "Contingency (chi-square)":
        c1, c2 = st.columns(2)
        a = c1.selectbox("Variable A", ["female","prog","ses","schtyp","race"], key="ca")
        b = c2.selectbox("Variable B", ["female","prog","ses","schtyp","race"], key="cb", index=1)
        ct = pd.crosstab(df[a], df[b])
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        n = ct.values.sum()
        cramers_v = np.sqrt(chi2/(n*(min(ct.shape)-1))) if min(ct.shape)>1 else np.nan
        st.write(f"Chi-square = {chi2:.3f}, dof = {dof}, p = {p:.4g}")
        st.write(f"Cramér's V = {cramers_v:.3f}")
        st.dataframe(ct, width='stretch')

    st.divider()
    st.markdown("### OLS regression diagnostics")
    c1, c2 = st.columns(2)
    y_ols = c1.selectbox("Dependent variable", SCORE_COLS, key="ols_y", index=2)
    x_ols = c2.multiselect("Independent variables",
                           [c for c in df.columns if c != y_ols and c != "id"],
                           default=["read","write","female","prog"], key="ols_x")
    if x_ols and st.button("Run OLS"):
        sub = df[[y_ols]+x_ols].dropna()
        X = sm.add_constant(sub[x_ols])
        model = sm.OLS(sub[y_ols], X).fit()
        st.text(model.summary().as_text())
        # VIF
        try:
            vif = pd.DataFrame({
                "Variable": X.columns,
                "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            })
            st.markdown("**VIF (multicollinearity)**")
            st.dataframe(vif.round(2), width='stretch')
        except Exception:
            pass
        # Breusch-Pagan
        try:
            bp = het_breuschpagan(model.resid, model.model.exog)
            st.write(f"Breusch-Pagan: LM = {bp[0]:.3f}, p = {bp[1]:.4g}")
        except Exception:
            pass


# =========================================================
# TAB 4 — REGRESSION (supervised)
# =========================================================
def build_regressors():
    m = {
        "Linear": LinearRegression(),
        "Polynomial (deg 2)": make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
        "Huber": HuberRegressor(max_iter=1000),
        "Bayesian Ridge": BayesianRidge(),
        "Quantile": QuantileRegressor(quantile=0.5, alpha=0.01, solver="highs"),
        "Gaussian Process": GaussianProcessRegressor(random_state=42),
    }
    if HAS_XGB: m["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0)
    if HAS_LGBM: m["LightGBM"] = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
    if HAS_CAT: m["CatBoost"] = CatBoostRegressor(n_estimators=200, depth=4, learning_rate=0.05, random_state=42, verbose=0)
    return m

with tab4:
    st.subheader("Regression models")
    c1, c2 = st.columns(2)
    target = c1.selectbox("Target", SCORE_COLS, index=2, key="reg_t")
    test_size = c2.slider("Test size (%)", 10, 40, 20, 5, key="reg_ts")/100
    feat_pool = GROUP_COLS + SCORE_COLS
    default_feats = [c for c in feat_pool if c != target]
    features = st.multiselect("Features", feat_pool, default=default_feats, key="reg_f")

    if features and st.button("Train regressors"):
        X = df[features].copy()
        y = df[target].copy()
        for c in X.columns:
            if X[c].dtype.kind in "if":
                X[c] = X[c].fillna(X[c].median())
            else:
                X[c] = X[c].fillna(X[c].mode().iloc[0])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

        models = build_regressors()
        rows, fitted, preds = [], {}, {}
        progress = st.progress(0.0, text="Training...")
        for i, (name, m) in enumerate(models.items()):
            try:
                m.fit(X_tr, y_tr)
                pred_tr = m.predict(X_tr); pred_te = m.predict(X_te)
                cv = cross_val_score(m, X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=42),
                                     scoring="neg_root_mean_squared_error")
                rows.append({"Model": name,
                             "CV RMSE": -cv.mean(),
                             "Train RMSE": np.sqrt(mean_squared_error(y_tr, pred_tr)),
                             "Test RMSE": np.sqrt(mean_squared_error(y_te, pred_te)),
                             "Test MAE": mean_absolute_error(y_te, pred_te),
                             "Test R²": r2_score(y_te, pred_te)})
                fitted[name] = m; preds[name] = pred_te
            except Exception as e:
                rows.append({"Model": name, "CV RMSE": np.nan, "Train RMSE": np.nan,
                             "Test RMSE": np.nan, "Test MAE": np.nan, "Test R²": np.nan})
            progress.progress((i+1)/len(models), text=f"Trained {name}")
        progress.empty()

        lb = pd.DataFrame(rows).sort_values("Test RMSE").reset_index(drop=True)
        st.subheader("Leaderboard")
        st.dataframe(lb.round(3), width='stretch')

        st.plotly_chart(px.bar(lb.melt(id_vars="Model", value_vars=["CV RMSE","Test RMSE","Test MAE"]),
                        x="Model", y="value", color="variable", barmode="group",
                        title="Model comparison (lower is better)"), width='stretch')

        best = lb.iloc[0]["Model"]
        st.success(f"Best by Test RMSE: **{best}**")
        if best in fitted and best in preds:
            pred = preds[best]; resid = y_te - pred
            c1, c2 = st.columns(2)
            with c1:
                f = px.scatter(x=pred, y=resid, labels={"x":"Predicted","y":"Residual"},
                               title=f"{best} residuals")
                f.add_hline(y=0, line_dash="dash")
                st.plotly_chart(f, width='stretch')
            with c2:
                f = px.scatter(x=y_te, y=pred, labels={"x":"Actual","y":"Predicted"},
                               title=f"{best} actual vs predicted")
                f.add_shape(type="line", x0=y_te.min(), y0=y_te.min(),
                            x1=y_te.max(), y1=y_te.max(), line=dict(dash="dash"))
                st.plotly_chart(f, width='stretch')

            if HAS_SHAP and best in ("XGBoost","LightGBM","CatBoost","Random Forest","Gradient Boosting"):
                st.markdown("**SHAP summary (best model)**")
                try:
                    expl = shap.Explainer(fitted[best], X_te)
                    sv = expl(X_te)
                    st.pyplot(shap.plots.beeswarm(sv, max_display=8, show=False).figure)
                except Exception as e:
                    st.info(f"SHAP unavailable: {e}")
            else:
                perm = permutation_importance(fitted[best], X_te, y_te, n_repeats=5, random_state=42)
                imp = pd.DataFrame({"Feature": X_te.columns, "Importance": perm.importances_mean}).sort_values("Importance")
                st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                                title="Permutation importance"), width='stretch')


# =========================================================
# TAB 5 — CLASSIFICATION (supervised)
# =========================================================
def build_classifiers():
    m = {
        "Logistic": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "SVC": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Gaussian NB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
        "SGD": SGDClassifier(max_iter=2000, random_state=42),
        "Perceptron": Perceptron(random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42),
    }
    if HAS_XGB: m["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
    if HAS_LGBM: m["LightGBM"] = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
    if HAS_CAT: m["CatBoost"] = CatBoostClassifier(n_estimators=200, depth=4, learning_rate=0.05, random_state=42, verbose=0)
    return m

with tab5:
    st.subheader("Binary classification (median split)")
    c1, c2 = st.columns(2)
    target_cls = c1.selectbox("Score to dichotomise", SCORE_COLS, index=2, key="cls_t")
    test_size_cls = c2.slider("Test size (%)", 10, 40, 20, 5, key="cls_ts")/100
    y_bin = (df[target_cls] > df[target_cls].median()).astype(int)
    st.caption(f"Target: 1 if `{target_cls}` > median ({df[target_cls].median():.0f}), else 0")

    feat_pool_cls = GROUP_COLS + [c for c in SCORE_COLS if c != target_cls]
    features_cls = st.multiselect("Features", feat_pool_cls, default=feat_pool_cls, key="cls_f")

    if features_cls and st.button("Train classifiers"):
        X = df[features_cls].copy()
        for c in X.columns:
            if X[c].dtype.kind in "if": X[c] = X[c].fillna(X[c].median())
            else: X[c] = X[c].fillna(X[c].mode().iloc[0])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_bin, test_size=test_size_cls, random_state=42, stratify=y_bin)

        models = build_classifiers()
        rows, fitted = [], {}
        progress = st.progress(0.0, text="Training...")
        for i, (name, m) in enumerate(models.items()):
            try:
                m.fit(X_tr, y_tr)
                pred = m.predict(X_te)
                proba = m.predict_proba(X_te)[:,1] if hasattr(m, "predict_proba") else None
                rows.append({"Model": name,
                             "Accuracy": accuracy_score(y_te, pred),
                             "Precision": precision_score(y_te, pred, zero_division=0),
                             "Recall": recall_score(y_te, pred, zero_division=0),
                             "F1": f1_score(y_te, pred, zero_division=0),
                             "ROC-AUC": roc_auc_score(y_te, proba) if proba is not None else np.nan})
                fitted[name] = (m, proba)
            except Exception:
                rows.append({"Model": name, "Accuracy": np.nan, "Precision": np.nan,
                             "Recall": np.nan, "F1": np.nan, "ROC-AUC": np.nan})
            progress.progress((i+1)/len(models), text=f"Trained {name}")
        progress.empty()

        lb = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
        st.subheader("Leaderboard")
        st.dataframe(lb.round(3), width='stretch')

        best = lb.iloc[0]["Model"]
        if best in fitted:
            m, proba = fitted[best]
            pred = m.predict(X_te)
            st.success(f"Best by ROC-AUC: **{best}**")
            c1, c2 = st.columns(2)
            with c1:
                cm = confusion_matrix(y_te, pred)
                st.plotly_chart(px.imshow(cm, text_auto=True, title="Confusion matrix"),
                                width='stretch')
            with c2:
                if proba is not None:
                    fpr, tpr, _ = roc_curve(y_te, proba)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={roc_auc_score(y_te, proba):.3f}"))
                    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
                    fig.update_layout(title="ROC curve", xaxis_title="FPR", yaxis_title="TPR")
                    st.plotly_chart(fig, width='stretch')


# =========================================================
# TAB 6 — CLUSTERING
# =========================================================
def build_clusterers(k):
    m = {
        "K-means": KMeans(n_clusters=k, random_state=42, n_init=10),
        "Agglomerative": AgglomerativeClustering(n_clusters=k),
        "GMM": GaussianMixture(n_components=k, random_state=42),
        "Spectral": SpectralClustering(n_clusters=k, random_state=42, assign_labels="discretize"),
        "Mean Shift": MeanShift(),
        "DBSCAN": DBSCAN(eps=0.9, min_samples=5),
        "OPTICS": OPTICS(min_samples=5),
        "Affinity Prop": AffinityPropagation(random_state=42),
    }
    if HAS_KMED: m["K-Medoids"] = KMedoids(n_clusters=k, random_state=42)
    return m

with tab6:
    st.subheader("Clustering comparison")
    k = st.slider("k (for partition methods)", 2, 6, 3, key="km_k")
    if st.button("Run clustering"):
        Xc = df[SCORE_COLS].copy().fillna(df[SCORE_COLS].median())
        Xs = StandardScaler().fit_transform(Xc)
        pcs = PCA(n_components=2).fit_transform(Xs)

        rows, labels_all = [], {}
        for name, m in build_clusterers(k).items():
            try:
                if name == "GMM":
                    lab = m.fit_predict(Xs)
                else:
                    lab = m.fit_predict(Xs)
                uniq = set(lab) - {-1}
                if len(uniq) < 2:
                    sil = db = ch = np.nan
                else:
                    sil = silhouette_score(Xs, lab)
                    db = davies_bouldin_score(Xs, lab)
                    ch = calinski_harabasz_score(Xs, lab)
                rows.append({"Method": name, "Clusters": len(uniq),
                             "Silhouette": sil, "Davies-Bouldin": db,
                             "Calinski-Harabasz": ch})
                labels_all[name] = lab
            except Exception:
                rows.append({"Method": name, "Clusters": np.nan,
                             "Silhouette": np.nan, "Davies-Bouldin": np.nan,
                             "Calinski-Harabasz": np.nan})
        lb = pd.DataFrame(rows).sort_values("Silhouette", ascending=False).reset_index(drop=True)
        st.subheader("Method comparison")
        st.dataframe(lb.round(3), width='stretch')

        method = st.selectbox("Visualise", list(labels_all.keys()), key="km_vis")
        lab = labels_all[method]
        pdf = pd.DataFrame({"PC1": pcs[:,0], "PC2": pcs[:,1], "Cluster": lab.astype(str)})
        st.plotly_chart(px.scatter(pdf, x="PC1", y="PC2", color="Cluster",
                        title=f"PCA projection · {method}"), width='stretch')

        sizes = pd.Series(lab).value_counts().reset_index()
        sizes.columns = ["Cluster","Count"]
        st.plotly_chart(px.bar(sizes, x="Cluster", y="Count",
                        title=f"Cluster sizes · {method}"), width='stretch')

        prof = pd.DataFrame(Xc); prof["Cluster"] = lab
        st.dataframe(prof.groupby("Cluster")[SCORE_COLS].mean().round(2), width='stretch')
        st.download_button("Download cluster assignments",
                           prof.to_csv(index=False), "clusters.csv", "text/csv")


# =========================================================
# TAB 7 — DIMENSIONALITY REDUCTION
# =========================================================
with tab7:
    st.subheader("Dimensionality reduction")
    Xd = StandardScaler().fit_transform(df[SCORE_COLS].fillna(df[SCORE_COLS].median()))

    method = st.selectbox("Method",
        ["PCA","t-SNE","UMAP","ICA","SVD","Factor Analysis"],
        key="dr_m")

    if method == "PCA":
        p = PCA(n_components=2).fit_transform(Xd)
        expl = PCA().fit(Xd).explained_variance_ratio_
        st.plotly_chart(px.bar(x=[f"PC{i+1}" for i in range(len(expl))], y=expl,
                        title="Explained variance ratio"), width='stretch')
        pdf = pd.DataFrame({"C1": p[:,0], "C2": p[:,1]})
        st.plotly_chart(px.scatter(pdf, x="C1", y="C2", title="PCA projection"), width='stretch')
    elif method == "t-SNE":
        from sklearn.manifold import TSNE
        p = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(Xd)
        pdf = pd.DataFrame({"C1": p[:,0], "C2": p[:,1]})
        st.plotly_chart(px.scatter(pdf, x="C1", y="C2", title="t-SNE projection"), width='stretch')
    elif method == "UMAP" and HAS_UMAP:
        p = umap.UMAP(n_components=2, random_state=42).fit_transform(Xd)
        pdf = pd.DataFrame({"C1": p[:,0], "C2": p[:,1]})
        st.plotly_chart(px.scatter(pdf, x="C1", y="C2", title="UMAP projection"), width='stretch')
    elif method == "ICA":
        p = FastICA(n_components=2, random_state=42).fit_transform(Xd)
        pdf = pd.DataFrame({"C1": p[:,0], "C2": p[:,1]})
        st.plotly_chart(px.scatter(pdf, x="C1", y="C2", title="ICA projection"), width='stretch')
    elif method == "SVD":
        p = TruncatedSVD(n_components=2, random_state=42).fit_transform(Xd)
        pdf = pd.DataFrame({"C1": p[:,0], "C2": p[:,1]})
        st.plotly_chart(px.scatter(pdf, x="C1", y="C2", title="SVD projection"), width='stretch')
    elif method == "Factor Analysis":
        p = FactorAnalysis(n_components=2, random_state=42).fit_transform(Xd)
        pdf = pd.DataFrame({"C1": p[:,0], "C2": p[:,1]})
        st.plotly_chart(px.scatter(pdf, x="C1", y="C2", title="Factor Analysis projection"), width='stretch')


# =========================================================
# TAB 8 — ASSOCIATION RULES
# =========================================================
with tab8:
    st.subheader("Association rules (Apriori / FP-Growth)")
    if not HAS_MLX:
        st.warning("Install `mlxtend` to use this section: `pip install mlxtend`")
    else:
        st.caption("Categorical baskets derived from binned columns: sex, prog, ses, high/low scores.")
        basket = pd.DataFrame({
            "female": df["female"].astype(str),
            "prog": df["prog"].astype(str),
            "ses": df["ses"].astype(str),
        })
        for c in SCORE_COLS:
            basket[f"high_{c}"] = (df[c] > df[c].median()).astype(str)
        onehot = pd.get_dummies(basket.astype(str))
        algo = st.radio("Algorithm", ["Apriori", "FP-Growth"], horizontal=True, key="ar_algo")
        min_sup = st.slider("Min support", 0.05, 0.5, 0.15, 0.05, key="ar_sup")
        if st.button("Mine rules"):
            freq = apriori(onehot, min_support=min_sup, use_colnames=True) if algo=="Apriori" \
                   else fpgrowth(onehot, min_support=min_sup, use_colnames=True)
            rules = association_rules(freq, metric="lift", min_threshold=1.0)
            if rules.empty:
                st.info("No rules at this support threshold.")
            else:
                rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
                rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
                st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]]
                             .sort_values("lift", ascending=False).head(20).round(3),
                             width='stretch')


# =========================================================
# TAB 9 — INTERPRETATION & GUIDANCE
# =========================================================
with tab9:
    st.subheader("Responsible interpretation")

    st.markdown("""
### What this dashboard supports
- Transparent data intake, quality checks, score distributions and group comparisons.
- Independent statistical inference with assumptions and visual evidence.
- Reproducible train/test splits, cross-validation and multi-model comparison.
- Exploratory clustering and dimensionality reduction for student score profiles.
- Interpretability via permutation importance and SHAP (when available).

### Analytical limitations
- The bundled HSB extract is a historical educational sample (**n=200**) with limited variables.
- No attendance, outcomes after school, intervention exposure or programme costs.
- Binary outcomes here are derived from score medians — they are **teaching labels**, not validated risk labels.
- A new upload replaces the active dataset; it does not accumulate records or persist models.
- Model selection uses a single train/test split; results describe this split, not future cohorts.

### What is intentionally NOT included (and why)
| Category | Why excluded |
| :--- | :--- |
| CNNs, RNNs, LSTMs, GRUs, Transformers, GNNs, ViTs, U-Net | No images, sequences, or graphs — HSB is 200×11 tabular. |
| GANs, VAEs, Autoencoders | Generative/deep methods are not appropriate for n=200 tabular. |
| Reinforcement Learning (Q-learning, DQN, PPO, SAC…) | No environment, reward signal or sequential decisions. |
| Semi-supervised (self-training, FixMatch, Mean Teacher…) | HSB has complete labels — semi-supervised exists for scarce labels. |
| Anomaly detection (Isolation Forest, LOF…) | No known anomalies to validate against — would produce toy output. |

### Responsible use
Prediction does not imply educational causation. Do not deploy student-level classifications as automated high-stakes decisions.
""")