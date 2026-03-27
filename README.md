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
| Metric | Non-Fraud (0) | Fraud (1) |
| :--- | :--- | :--- |
| **Precision** | 1.00 | **0.91** |
| **Recall** | 1.00 | **0.81** |
| **F1-Score** | 1.00 | **0.86** |
| **Support** | 56,864 | 98 |

### Model Effectiveness Analysis
* **Fraud Detection (Recall):** The model successfully identified **81%** of all actual fraudulent cases. In fraud detection, high recall is vital to minimize financial loss.
* **Reliability (Precision):** When the model flags a transaction, it is correct **91%** of the time. This ensures that customers experience minimal "false alarms" or unnecessary card blocks.
* **Overall Balance:** An **F1-Score of 0.86** for the minority class confirms that the model is robust and has not simply memorized the majority class.

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
