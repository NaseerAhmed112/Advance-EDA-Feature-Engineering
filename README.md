###               Data Science Internship — Project 1**
## Advanced EDA & Feature Engineering | Industrial Training Kit

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized%20Ops-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-KNN%20Imputation-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-2026-blue?style=for-the-badge)

**Powered by [DecodeLabs](https://www.decodelabs.tech) | Greater Lucknow, India**

</div>

---

## 📌 Short Description

> *"Data preprocessing is not janitorial work — it is the structural engineering of mathematical truth."*

**Project 1** is the essential foundation of the DecodeLabs Data Science Industrial Training Kit. Before building any predictive model, raw data must be transformed into a mathematically clean, high-fidelity feature store. This project covers the full enterprise-grade data engineering pipeline — from handling missing values with statistical imputation and neutralizing outliers using IQR/Z-Score methods, to engineering new predictive features and validating the output with runtime schema contracts using Pandera.

---

## 🎯 Project Goal

**Transform raw, chaotic data into a mathematically clean dataset ready for machine learning algorithms** — following the Input → Process → Output (IPO) architecture used in production enterprise systems.

---

## 🏗️ Pipeline Architecture (IPO Blueprint)

```
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│   MODULE 1      │     │     MODULE 2         │     │     MODULE 3         │
│    INPUT        │────►│     PROCESS          │────►│     OUTPUT           │
│                 │     │                      │     │                      │
│ Securing        │     │ Vectorized Math      │     │ Pandera Contracts    │
│ Fidelity        │     │ Encoding             │     │ Feast Feature Stores │
│                 │     │ Collinearity         │     │                      │
│ Missing Values  │     │ Eradication          │     │ ML-Ready Dataset     │
│ Outlier Bounds  │     │                      │     │                      │
└─────────────────┘     └─────────────────────┘     └──────────────────────┘
```

---

## 📂 Repository Structure

```
data-science-internship-project-1/
│
├── 📓 Project1_EDA_Feature_Engineering.ipynb   ← Main notebook (run this)
├── 📊 Dataset_for_Data_Analytics.xlsx           ← E-commerce orders dataset
├── 📄 README.md                                 ← You are here
│
├── outputs/
│   ├── missing_data_heatmap.png                 ← Missingness visualization
│   ├── outlier_boxplots.png                     ← Before/After IQR treatment
│   ├── correlation_matrix.png                   ← Collinearity heatmap
│   ├── feature_distributions.png               ← Engineered features EDA
│   └── cleaned_dataset.csv                      ← Final ML-ready output
```

---

## 📦 Dataset

| Property | Details |
|----------|---------|
| **File** | `Dataset_for_Data_Analytics.xlsx` |
| **Records** | 1,200 orders |
| **Features** | 14 columns |
| **Date Range** | January 2023 – June 2025 |
| **Products** | Monitor, Phone, Tablet, Chair, Printer, Laptop, Desk |
| **Missing Data** | `CouponCode` — 309 null values (25.75%) → KNN Imputation |
| **Key Columns** | `OrderID`, `Date`, `CustomerID`, `Product`, `Quantity`, `UnitPrice`, `TotalPrice`, `OrderStatus`, `PaymentMethod`, `ReferralSource` |

---

## 🔬 Key Engineering Concepts

### Phase 1 — Securing Input Fidelity

#### 🔹 The Missing Data Decision Matrix

| Missingness % | Strategy | Method | Trade-Off |
|:---:|---|---|---|
| **< 5%** | Row Deletion | `df.dropna()` | Low CPU overhead, zero synthetic bias |
| **5% – 20% (Skewed)** | Statistical Imputation | Global Median | Robust against outliers; deflates std deviation |
| **5% – 20% (Correlated)** | Group-Wise Imputation | Sub-group Conditional Mean | Retains variance patterns across sub-populations |
| **> 20%** | Multi-Dimensional Estimation | KNN Imputation | Captures complex relationships; O(N²) complexity |

```python
# Decision logic — structural, not arbitrary
missing_pct = df.isnull().mean() * 100

for col in df.columns:
    pct = missing_pct[col]
    if   pct < 5:    df.dropna(subset=[col], inplace=True)
    elif pct < 20:   df[col].fillna(df[col].median(), inplace=True)
    else:            # Apply KNN Imputation
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df[[col]] = imputer.fit_transform(df[[col]])
```

---

#### 🔹 Outlier Detection & Neutralization — IQR Method

```
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

**Winsorization over Deletion** — `numpy.clip()` preserves row count and sequential integrity. Deleting outlier rows destroys adjacent feature volume and breaks temporal sequences.

```python
# IQR Winsorization — never drop rows
Q1  = df["UnitPrice"].quantile(0.25)
Q3  = df["UnitPrice"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df["UnitPrice"] = np.clip(df["UnitPrice"], lower, upper)  # cap, don't drop
```

---

### Phase 2 — The Vectorized Computation Engine

#### 🔹 Vectorization vs. Iteration

| Approach | Complexity | Mechanism |
|----------|-----------|-----------|
| Python `for` loops | O(N) bottleneck | Dynamic type-checking CPU overhead |
| Pandas/NumPy vectorized | Block-allocated RAM | Compiled C-level SIMD operations |

**Rule:** Never loop over rows. All transformations use vectorized Pandas/NumPy operations that execute directly in system RAM via C.

---

#### 🔹 Categorical Encoding — One-Hot over Label Encoding

Label Encoding assigns integers (London=1, Paris=2, Tokyo=3) — creating a **false mathematical distance** (Tokyo = 3× London). This introduces synthetic spatial hierarchy that corrupts model optimization.

**One-Hot Encoding** maps C categories into C orthogonal coordinate axes — each equidistant from every other.

```python
# Wrong — introduces false ordinal hierarchy
df["PaymentMethod_encoded"] = LabelEncoder().fit_transform(df["PaymentMethod"])

# Correct — orthogonal coordinate space
df = pd.get_dummies(df, columns=["PaymentMethod", "Product", "ReferralSource"],
                    drop_first=True)   # drop_first removes dummy variable trap
```

---

#### 🔹 Collinearity Eradication Algorithm

Multicollinearity makes the feature matrix X non-invertible — Ordinary Least Squares (OLS) parameters become impossible to calculate, and coefficient estimates become unstable.

**4-Step Algorithm:**
```
Step 1: Build absolute correlation matrix
Step 2: Isolate upper triangle (avoid duplicate pairs)
Step 3: Identify all pairs with correlation > 0.80
Step 4: For each pair, drop the feature with lower correlation to target y
```

```python
# Collinearity eradication
corr_matrix = df.corr().abs()
upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop     = [col for col in upper.columns if any(upper[col] > 0.80)]
df.drop(columns=to_drop, inplace=True)
```

---

### Phase 3 — Structural Contracts & Scaling

#### 🔹 Runtime Validation with Pandera

```python
import pandera as pa

schema = pa.DataFrameSchema({
    "TotalPrice"  : pa.Column(float,  pa.Check.ge(0)),
    "Quantity"    : pa.Column(int,    pa.Check.between(1, 100)),
    "UnitPrice"   : pa.Column(float,  pa.Check.ge(0)),
    "OrderStatus" : pa.Column(str,    pa.Check.isin(["Delivered","Shipped",
                                                      "Pending","Cancelled","Returned"])),
})

validated_df = schema.validate(df, lazy=True)  # lazy=True — collects ALL errors at once
```

> `lazy=True` prevents the pipeline from crashing on the first error — it processes the entire dataframe and collects all structural failures into a single diagnostic report.

---

#### 🔹 Feature Store with Feast

Feast acts as the **single source of truth** for features — eliminating Training-Serving Skew (where feature logic is duplicated across offline scripts and online APIs).

| Store | Backend | Use Case |
|-------|---------|---------|
| **Offline Store** | Parquet / Snowflake | High-throughput batch query for model training |
| **Online Store** | Redis / DynamoDB | Sub-10ms latency lookup for real-time API serving |

---

## ⚙️ Feature Engineering — 3 New Predictive Features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `Revenue_per_Item` | `TotalPrice / Quantity` | Captures average unit value per order |
| `Discount_Flag` | `1 if CouponCode is not NaN else 0` | Binary indicator for promotional behavior |
| `Order_Month` | `pd.to_datetime(Date).dt.month` | Captures seasonality patterns |

```python
df["Revenue_per_Item"] = df["TotalPrice"] / df["Quantity"]
df["Discount_Flag"]    = df["CouponCode"].notna().astype(int)
df["Order_Month"]      = pd.to_datetime(df["Date"]).dt.month
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/NaseerAhmed112/data-science-internship-project-1.git
cd data-science-internship-project-1

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn pandera feast jupyter openpyxl

# 3. Launch Jupyter
jupyter notebook Project1_EDA_Feature_Engineering.ipynb

# 4. Run All → Kernel → Restart & Run All
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `Pandas` | Data ingestion, wrangling, group-wise operations |
| `NumPy` | Vectorized math, IQR clipping, array operations |
| `Scikit-learn` | KNNImputer, StandardScaler, train/test split |
| `Pandera` | Runtime schema validation & structural contracts |
| `Feast` | Feature store — offline/online dual-store serving |
| `Matplotlib` | Distribution plots, boxplots, heatmaps |
| `Seaborn` | Correlation matrix, missing data visualization |
| `OpenPyXL` | Reading `.xlsx` dataset |

---

## 📊 Results Summary

```
═══════════════════════════════════════════════════════════════
  PROJECT 1: ADVANCED EDA & FEATURE ENGINEERING — RESULTS
═══════════════════════════════════════════════════════════════
  Dataset          : E-Commerce Orders (1,200 rows × 14 cols)
  Missing Values   : CouponCode (25.75%) → KNN Imputed
  Outliers         : UnitPrice, TotalPrice → IQR Winsorized
  Encoding         : PaymentMethod, Product → One-Hot (OHE)
  Collinearity     : Correlation pairs > 0.80 → Dropped
  Features Added   : 3 new engineered predictive features
  Schema Validated : Pandera runtime contract assertions
  Output           : ML-ready cleaned dataset
═══════════════════════════════════════════════════════════════
```

---

## 📚 Key Skills Demonstrated

```
✅ Statistical Imputation (Mean / Median / KNN)
✅ Missing Data Decision Matrix (5% / 20% thresholds)
✅ Outlier Detection (Z-Score & IQR methods)
✅ Winsorization with numpy.clip()
✅ Vectorized Pandas/NumPy Operations (no for-loops)
✅ One-Hot Encoding (orthogonal coordinate space)
✅ Multicollinearity Detection & Eradication
✅ Feature Engineering (3+ new predictive columns)
✅ Runtime Schema Validation with Pandera
✅ Feature Store Architecture with Feast
✅ Point-in-Time Correctness (preventing data leakage)
```

---

## 🏢 About

| | |
|--|--|
| **Program** | Data Science Industrial Training Kit |
| **Project** | Project 1 — Essential Foundation |
| **Track** | Advanced EDA & Feature Engineering |
| **Organization** | DecodeLabs |
| **Batch** | 2026 |
| **Location** | Greater Lucknow, India |
| **Contact** | decodelabs.tech@gmail.com |
| **Website** | [www.decodelabs.tech](https://www.decodelabs.tech) |

---

## 📜 License

This project is part of the DecodeLabs Industrial Training Program and is submitted for educational and certification purposes.

---

<div align="center">

**Made with ❤️ | DecodeLabs Batch 2026**

*"Your journey to becoming a professional Data Scientist begins right here, right now, with the very first dataset you clean today."*

</div>
