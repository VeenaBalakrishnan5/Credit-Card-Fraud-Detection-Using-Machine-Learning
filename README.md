Credit Card Fraud Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

##  Project Overview
Identifying fraudulent transactions is a critical challenge for financial institutions. This project develops a machine learning pipeline to detect fraud in highly imbalanced datasets. By leveraging **SMOTE** for data balancing and **Random Forest** for classification, the model achieves high sensitivity to fraud while maintaining a professional standard of precision.

##  Dataset Specifications
The model utilizes the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), containing transactions made by European cardholders.
* **Total Samples:** 284,807
* **Legitimate (Class 0):** 284,315 (99.83%)
* **Fraudulent (Class 1):** 492 (0.17%)
* **Features:** 28 PCA-transformed variables (V1–V28), `Time`, and `Amount`.

##  Technical Workflow
1.  **Preprocessing:** Scaling of the `Amount` and `Time` features using `StandardScaler`.
2.  **Addressing Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set to ensure the model learns fraud patterns effectively.
3.  **Modeling:** Implemented a **Random Forest Classifier** to handle complex non-linear relationships.
4.  **Evaluation:** Focused on Precision-Recall curves and F1-scores rather than raw accuracy.

##  Performance Results
The model was evaluated on a test set of **56,962 transactions**.

### Classification Report
| Metric | Non-Fraud (0) | Fraud (1) | **Accuracy** | **Macro Average** | **Weighted Average** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Precision** | 1.00 | **0.87** | | **0.93** | **1.00** |
| **Recall** | 1.00 | **0.80** | |  **0.90** | **1.00** |
| **F1-Score** | 1.00 | **0.83** | **1.00** | **0.91** | **1.00** |
| **Support** | 56,864 | 98 |

## Model Performance Analysis
Despite the difficulty of the dataset, the model performs exceptionally well:

**Precision** (0.87): This means that when the model flags a transaction as fraud, it is correct 87% of the time. This is a strong result that minimizes "False Positives," ensuring that legitimate customers are not frequently bothered by false fraud alerts.

**Recall** (0.80): The model successfully identifies 80% of all fraudulent transactions. In financial security, this is the most important metric because it represents the percentage of actual theft the system is catching.

**F1-Score** (0.83): This score is the harmonic mean of precision and recall. Achieving an 0.83 on such a highly imbalanced dataset proves that the combination of SMOTE and Random Forest is a highly effective strategy for this problem.

**Overall Accuracy** (1.00): While the accuracy looks perfect, the precision and recall scores provide a more honest and successful picture of the model's ability to catch the rare "needle in the haystack."

##  Installation & Usage
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
    ```
3.  **Run the Notebook:**
    Open `credit_card_fraud_dectection.ipynb` in Jupyter or Google Colab and execute the cells.

##  Conclusion
The integration of **SMOTE** and **Random Forest** proved highly effective. The model successfully overcomes the "Accuracy Paradox" associated with imbalanced data, providing a specialized solution that prioritizes the detection of the rare but critical fraud class.
