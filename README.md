# 💧 Water Safety Predictor

A machine learning web app that predicts whether water is safe to drink,
built on top of a probability & statistics case study on microbial detection.

🔗 **Live app**: https://mvreo9fqxkgpxzmjnfkpdf.streamlit.app/

---

## 🧠 Project story
This project started as a statistics assignment on microbial counts in
environmental water samples (Poisson, Binomial, Bernoulli distributions).
It was then extended into a full ML pipeline with a deployed web app.

## 📊 What it does
- Takes 9 water parameters as input (pH, Sulfate, Turbidity, etc.)
- Predicts whether the water is safe to drink (binary classification)
- Shows ML model confidence + a Poisson-based microbial risk estimate
- Bridges statistical modeling with machine learning

## 🗂️ Dataset
- **Source**: [Water Potability — Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
- 3,276 water samples, 9 features, binary label (safe/not safe)

## 🔬 Methodology
| Stage | What was done |
|-------|--------------|
| 1 | Probability distributions (Poisson, Binomial, Bernoulli) on microbial data |
| 2 | EDA — missing value imputation, class balance, correlation analysis |
| 3 | ML models — Logistic Regression, Random Forest, XGBoost, Gradient Boosting |
| 4 | Streamlit web app deployed on Streamlit Cloud |

## 📈 Model performance
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 55% | 0.54 |
| Random Forest | 80% | 0.79 |
| XGBoost | 78% | 0.78 |
| Gradient Boosting | 80% | 0.79 |
| Random Forest (tuned) ✓ | 80% | 0.79 |

> Note: Low feature-target correlation in this dataset naturally limits accuracy.
> 80% F1 is considered strong for this benchmark.

## 🛠️ Tech stack
`Python` `scikit-learn` `XGBoost` `pandas` `matplotlib` `seaborn` `Streamlit`

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👩‍💻 Built by
BMSCE AI/ML Department · Probability & Statistics for ML 
