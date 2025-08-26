# Credit Card Fraud Detection using State-of-the-Art Machine Learning and Deep Learning

This project implements a robust fraud detection system using advanced Machine Learning (ML) and Deep Learning (DL) algorithms to identify fraudulent credit card transactions. It addresses major challenges such as class imbalance, evolving fraud patterns, and high false positive rates using optimized models, feature selection, and model evaluation strategies.

---

## 🧠 Technologies Used

- **Machine Learning Models**: Logistic Regression, Random Forest, XGBoost, Support Vector Machine (SVM), Decision Tree, K-Nearest Neighbors (KNN), Naive Bayes
- **Deep Learning Models**: Convolutional Neural Networks (CNN) with 14, 17, and 20-layer architectures
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `imblearn`, `matplotlib`, `seaborn`, `tensorflow`, `keras`, `xgboost`
- **Data Handling Techniques**: SMOTE, Class Weight Balancing, StandardScaler, PCA

---

## 📊 Dataset

- The dataset used is a publicly available credit card fraud dataset from European cardholders.
- Total transactions: **284,807**
- Fraudulent transactions: **492** (~0.172%)

> Features include anonymized PCA components (`V1-V28`), `Time`, `Amount`, and `Class` (target variable).

---

## 🏗️ Project Structure

- **Module 1**: Machine Learning Approach
  - Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, KNN, Naive Bayes
- **Module 2**: Deep Learning Approach
  - CNN architectures with 14, 17, and 20 layers tested over 14 to 100 epochs
- **Module 3**: Performance Evaluation and Comparison
  - Evaluated using Accuracy, Precision, Recall, F1-Score, AUC, PRC

---

## 📈 Key Results

| Model Type | Accuracy | F1-Score | AUC | Remarks |
|------------|----------|----------|-----|---------|
| CNN (14-layer) | **99.94%** | 85.71% | 98% | Best DL model |
| XGBoost | 99.5% | 82% | 96% | Best ML model |
| Random Forest | 99.3% | 80% | 95% | Good baseline |
| Decision Tree | 98.7% | 76% | 91% | Simple, interpretable |

---

## ⚙️ Implementation Highlights

- **Class Imbalance Handling**: Applied SMOTE oversampling and under-sampling techniques.
- **Feature Engineering**: PCA-based transformation, feature scaling using StandardScaler.
- **Evaluation Metrics**: Precision, Recall, F1-Score, AUC, PRC, Confusion Matrix, Training & Validation Curves.
- **Model Tuning**: Dropout, BatchNormalization, Optimizers, Custom Callbacks, Layer Depth Optimization.

---

## 📌 Conclusion

Deep Learning models, especially CNNs with deeper architectures, outperform traditional ML models when trained with proper data balancing and hyperparameter tuning. The project demonstrates that using tailored architectures and training strategies can achieve **over 99.9% accuracy** in detecting credit card fraud.

---

## 🔮 Future Work

- Integrating hybrid models combining ML + DL
- Real-time fraud detection with streaming data
- Explainable AI (XAI) for model interpretability
- Edge or mobile deployment for transaction monitoring

---

## 👨‍💻 Authors

- **Rohan Singam** – Reg. No: 126003250 – B.Tech CSE 



---

## 📜 References

- [IEEE Access Base Paper (2022)](https://ieeexplore.ieee.org/document/9755930)
- Kaggle Credit Card Fraud Dataset
- Scikit-learn, TensorFlow, XGBoost Documentation

