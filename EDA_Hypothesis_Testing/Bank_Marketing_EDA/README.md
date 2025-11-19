# 🏦 Bank Marketing Campaign Analysis

## 📘 Project Overview
This project analyzes the **Bank Marketing Dataset** from [Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing), which contains data from marketing campaigns of a Portuguese banking institution. The goal is to **identify key factors influencing client subscription to term deposits** and **develop actionable insights** to improve future campaign effectiveness.

---

## 🎯 Objectives
- Explore and clean the dataset for accurate analysis.  
- Perform **EDA** to understand customer demographics, campaign behavior, and financial characteristics.  
- Identify **key features** influencing term deposit subscription.  
- Apply **statistical hypothesis testing** to validate relationships between features and the target variable (`Target` = yes/no).  
- Deliver clear insights and recommendations for marketing optimization.

---

## 🧠 Key Insights
### 📊 Demographics
- Majority of clients are **married** and have **secondary or tertiary education**.  
- Clients aged **18–25** and **65+** show the **highest subscription rates**.  

### 📞 Campaign Patterns
- **Cellular contact** yielded the **highest success (~15%)**; telephone and unknown channels were less effective.  
- **March and December** showed the **highest conversion rates**, while **May**—despite the highest contacts—had the **lowest success**.

### 💰 Financial & Behavioral Trends
- Clients with **higher balances** and **longer call durations** are significantly more likely to subscribe.  
- Past campaign success (`poutcome = success`) is the **strongest positive predictor** (64.7% subscription rate).  

---

## 🧮 Statistical Findings
| Test | Variables | Significant? | Notes |
|------|------------|---------------|-------|
| **ANOVA / Mann-Whitney / KS** | Age, Balance, Duration, Pdays | ✅ | Log-transformed duration showed the strongest significance. |
| **Chi-Square** | Job, Marital, Education, Contact, Month, Poutcome | ✅ | All categorical variables are strongly associated with subscription. |

---

## 🧰 Tech Stack
**Languages:** Python (Pandas, NumPy)  
**Visualization:** Matplotlib, Seaborn  
**Statistical Testing:** SciPy (ANOVA, Mann-Whitney U, KS, Chi-Square)  
**Tools:** Jupyter Notebook, Kaggle  

---

## 🪄 Methodology
1. **Data Loading & Profiling** – checked missing values, data types, distributions.  
2. **Exploratory Data Analysis (EDA)** – univariate and bivariate visualizations.  
3. **Feature Engineering** – log transformation of `duration`, binning of campaign contacts.  
4. **Statistical Testing** – hypothesis validation using appropriate tests.  
5. **Insights & Recommendations** – derived key factors influencing marketing success.  

---

## 📈 Visual Highlights
- Distribution of **education**, **job**, and **marital status**.  
- **Subscription rates by age group**, **month**, and **contact type**.  
- **Correlation heatmap** highlighting most influential numeric features.  
- **Boxplots and bar charts** illustrating call duration, previous contacts, and success outcomes.

---

## 📂 Dataset
- **Source:** Kaggle — Bank Marketing Dataset  
- **Rows:** 45,211  
- **Features:** 17 (7 numeric, 10 categorical)  
- **Target:** `Target` — whether the client subscribed to a term deposit (`yes`/`no`)

---

## 📊 Results Summary
- Term deposit subscription rate: **~11.7%**
- Strong predictors: `duration`, `poutcome`, `month`, and `contact`
- Recommendation: Prioritize clients with **previous positive outcomes**, **higher balances**, and **recent contacts** via **cellular channels**

---

## 🚀 Future Enhancements
- Build **predictive ML models** (Logistic Regression, Random Forest) for campaign success.  
- Deploy an **interactive dashboard** (Power BI / Streamlit).  
- Perform **A/B testing simulations** for campaign strategies.  

---

## 👩‍💻 Author
**Souhila Acil**  
📍 Data Scientist | Machine Learning | Forecasting | Analytics  
🔗 [LinkedIn](www.linkedin.com/in/sue-acil-00a79bb8) | [Portfolio](https://souhilaacil.my.canva.site/)
