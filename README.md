# A Machine Learning Approach to Predicting GDP Growth Using World Bank Development Indicators

**Student:** Muhammad Wasal Imtiaz (24077342)
**Module:** 7PAM2002 Data Science Project
**University of Hertfordshire — MSc Data Science**

---

## Introduction & Problem Statement

GDP forecasting is critical for governments setting national budgets, central banks adjusting interest rates, and international organisations like the IMF and World Bank allocating aid. Traditional econometric models are complex and difficult to scale. This project demonstrates that freely available World Bank data combined with machine learning can produce GDP growth forecasts that match published academic benchmarks — making economic forecasting more accessible without proprietary data or complex econometric models.

### Research Questions

1. Which machine learning models best predict GDP growth using socio-economic and development indicators from the World Bank Open Data platform?
2. How do linear and non-linear machine learning models differ in predictive performance and generalisation when forecasting GDP growth across countries and time periods?
3. Which socio-economic indicators contribute most to GDP growth prediction, and how consistent are these feature importance patterns across different models?

---

## Data

Seven World Bank indicators were collected covering the period 1995–2023:

| Indicator | Code | Role |
|---|---|---|
| GDP growth (annual %) | NY.GDP.MKTP.KD.ZG | **Target variable** |
| Inflation, consumer prices (annual %) | FP.CPI.TOTL.ZG | Macroeconomic stability |
| Unemployment (% of labour force) | SL.UEM.TOTL.ZS | Labour market health |
| Internet users (% of population) | IT.NET.USER.ZS | Technology adoption — **dropped** |
| Population growth (annual %) | SP.POP.GROW | Demographic driver |
| Education expenditure (% of GDP) | SE.XPD.TOTL.GD.ZS | Human capital — **dropped** |
| Energy use (kg oil eq. per capita) | EG.USE.PCAP.KG.OE | Economic activity — **dropped** |

Three indicators (Education Expenditure, Energy Use, and Internet Users) were dropped from the final model due to over 60% missing values — after imputation they contributed mostly noise rather than genuine signal.

---

## Methodology

### Data Collection & Merging

All seven indicator datasets were downloaded as CSV files from the World Bank Open Data platform. Each was reshaped from wide to long format and merged into a single panel dataset keyed on Country and Year.

### Data Cleaning & Preprocessing

- Filtered to the modern period (1995–2023) for reasonable data coverage.
- Missing values were filled within each country's timeline using forward-fill then backward-fill interpolation.
- Extreme GDP growth outliers were trimmed at the 1st and 99th percentiles.
- Countries with problematic data were removed: those with over 40% missing values, extreme GDP volatility (standard deviation exceeding 3x the global median), or fewer than 10 observations.

### Feature Engineering

Raw indicators showed weak correlation with GDP growth (maximum r = 0.20). Feature engineering raised this to 0.45 through:

- **Lag features** (t-1) for each remaining indicator (Inflation, Unemployment, Population Growth).
- **Autoregressive GDP lags** (lag1, lag2, lag3) — capturing GDP growth's strong autocorrelation, which emerged as the key insight of this project.
- **Rolling means** (2-year and 3-year) to capture GDP growth momentum.
- **GDP growth volatility** (rolling standard deviation) as an economic stability signal.
- **Misery Index proxy** — an interaction term between lagged Inflation and lagged Unemployment.
- **Country-level mean GDP growth** — a fixed-effect proxy capturing each country's structural growth rate.
- **Squared inflation lag** to capture non-linear effects.

### Train / Test Split

A time-aware split was used: training on data up to 2015 and testing on 2016–2023, excluding COVID shock years (2020–2021). This ensures the model is always trained on the past and evaluated on unseen future data — a random split would be unrealistic. COVID years were excluded because no historical economic indicator could anticipate a global pandemic; including them would unfairly penalise all models.

All features were normalised to [0, 1] using MinMax scaling fitted only on training data to prevent data leakage.

---

## Models

Three models of increasing complexity were trained and tuned via GridSearchCV with 5-fold cross-validation:

1. **Ridge Regression** — a linear baseline with L2 regularisation that prevents any single feature from dominating and handles correlated features well.
2. **Random Forest Regressor** — a parallel ensemble of hundreds of decision trees built on random data subsets, capable of capturing non-linear patterns.
3. **Gradient Boosting Regressor** — a sequential ensemble where each tree corrects the errors of the previous one, offering more power but greater overfitting risk.

A DummyRegressor (mean predictor) served as the naive baseline that all models must beat.

---

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (Mean) | 3.134 | 2.255 | -0.064 |
| **Ridge Regression** | **2.451** | **1.643** | **0.350** |
| Random Forest | 2.541 | 1.709 | 0.301 |
| Gradient Boosting | 2.622 | 1.726 | 0.256 |

**Ridge Regression wins** with R² = 0.350, falling within the published academic benchmark range of 0.25–0.50 for GDP prediction tasks. Its L2 regularisation handles multicollinearity in the engineered features better than the tree-based models, which show mild overfitting (particularly Gradient Boosting with a CV-to-test gap of -0.045).

Residual analysis confirms Ridge is unbiased — residuals scatter randomly around zero with no systematic error pattern, and most prediction errors fall within ±5%.

---

## Feature Importance & Explainability

Feature importance was assessed through three methods: Mean Decrease in Impurity (MDI), permutation importance, and SHAP values. All three consistently identify:

- **GDP_Growth_lag1** (autocorrelation) as the dominant predictor — GDP growth strongly predicts itself.
- **Country_GDP_mean** (structural growth rate) as the second most important feature.
- **Inflation and Unemployment lags** as secondary contributors.

SHAP analysis on both Gradient Boosting and Random Forest provides detailed explanations of how each feature pushes individual predictions higher or lower.

---

## Single-Country Case Study — United Kingdom

To test whether models can learn temporal patterns within a single, well-measured economy (removing cross-country noise), a separate UK-specific analysis was conducted. Models were retrained on UK-only data with the same temporal split. The UK timeline plot shows that models track stable economic years reasonably well but cannot predict black swan events like COVID-19.

---

## Noise Simulation Experiment

A validation experiment was conducted using 2,000 synthetic data points with a known true formula and Gaussian noise added at six levels (σ = 0 to 8). All three scikit-learn models plus a PyTorch neural network (2 hidden layers, ReLU activation, Adam optimiser) were tested:

- At **zero noise**: all models achieve near-perfect R² (~1.0), confirming they can learn the true signal.
- At **high noise** (σ = 8): R² drops to approximately 0 for all models, including the neural network.

This validates that the real-world R² of 0.35 is modest because GDP data is inherently noisy — it reflects a limitation of the data, not a failure of the models. It also demonstrates that neural networks are not a silver bullet for noisy economic data.

---

## Conclusions

1. **Best model:** Ridge Regression (R² = 0.350, RMSE = 2.451) — L2 regularisation handles multicollinearity in engineered features better than tree-based alternatives.
2. **Linear vs non-linear:** Ridge generalises better on this dataset. Tree models mildly overfit, and the temporal split provides a fair forward-looking evaluation.
3. **Most important features:** GDP_Growth_lag1 (autocorrelation) and Country_GDP_mean (structural growth rate), confirmed consistently across MDI, permutation importance, and SHAP values.

---

## Limitations

1. **COVID excluded** — 2020–2021 removed from evaluation as no historical features could predict a global pandemic.
2. **Autoregressive dominance** — GDP_Growth_lag1 is the strongest predictor, making the model largely a momentum predictor rather than a structural economic model.
3. **Three indicators dropped** — Education Expenditure, Energy Use, and Internet Users removed due to over 60% missing values.
4. **Forward-fill imputation** — remaining missing values filled using previous year values, introducing some synthetic bias.
5. **Single global model** — one model across 100+ countries cannot fully capture each country's unique economic structure.
6. **No external shocks** — wars, commodity crashes, trade policy changes, and political instability are absent from the indicators.

---

## Data Sources

All datasets are publicly available from the [World Bank Open Data](https://data.worldbank.org) platform:

- GDP growth (annual %) — `NY.GDP.MKTP.KD.ZG`
- Inflation, consumer prices (annual %) — `FP.CPI.TOTL.ZG`
- Unemployment, total (% of labour force) — `SL.UEM.TOTL.ZS`
- Population growth (annual %) — `SP.POP.GROW`
- Individuals using the Internet (% of population) — `IT.NET.USER.ZS` *(dropped)*
- Government expenditure on education (% of GDP) — `SE.XPD.TOTL.GD.ZS` *(dropped)*
- Energy use (kg of oil equivalent per capita) — `EG.USE.PCAP.KG.OE` *(dropped)*
