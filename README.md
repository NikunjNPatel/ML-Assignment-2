# Lithium-Ion Cell Diagnostic Classification

## Streamlit App: https://ml-assignment-2-2025ac05157.streamlit.app/ 


## 1. Problem Statement

Lithium-ion cells degrade over repeated charge and discharge cycles, affecting capacity, voltage behavior, temperature, and internal resistance.

The goal of the project is to classify the current health condition of a lithium-ion cell using supervised machine learning algorithms and measurable operating features.

The following models are covered:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors
4. Random Forest
5. Gaussian Naive Bayes

The health classes are defined using State of Health (SOH):

| Class | Health State | SOH |
|-------|--------------|-----|
| 0 | Healthy | >= 90% |
| 1 | Advance Degradation | 80 - 90% |
| 2 | Critical | <= 80 |

State of Health is calculated as:

$$SOH = \frac{C_k}{C_{initial}} \times 100$$

where:

- **$C_k$** = discharge capacity at the current cycle
- **$C_{initial}$** = Initial or Reference discharge capacity

The main objective is to determine which classifier can best identify degradation states using voltage, current, temperature, and discharge time related features.

## 2. Dataset Description

The project uses the **NASA Lithium-Ion Battery Aging Dataset**, mainly cells from  folder **BatteryAgingARC-FY08Q4**:

- B0005.mat
- B0006.mat
- B0007.mat
- B0018.mat

**Dataset Source:** https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset/versions/1 

Each discharge cycle is converted into one ML observation.

Typical features include:

- Start Voltage/End Voltage/Standard Deviation Voltage
- Maximum/Minimum/Average/Standard Deviation Temperature
- Maximum/Minimum/Average/Standard Deviation Current
- Discharge Duration till 0%,10%,25%,40% and 50% SOC

Discharge capacity is used to calculate SOH and generate the target class, but capacity is excluded from the dataset in order to avoid data leakage.

## 3. Github Repository Link

**Repository:** https://github.com/NikunjNPatel/ML-Assignment-2 

## 4. Evaluation Metrics

| Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|-----------|--------|----|-----|
| Random Forest | 0.9682539682539683 |	0.9976700668613477 | 0.968470903920862 | 0.961681329423265 | 0.9646535424313202 | 0.9511468800244182 |
| Naive Bayes |	0.8571428571428571 | 0.9631249873506856 | 0.8419501133786849 | 0.8512463343108504 |	0.844519301836375 |	0.7844014833224758 |
| Logistic Regression |	0.9523809523809523 | 0.9933869237071381 | 0.9447634237107921 | 0.9474340175953079 | 0.9455953768453768 | 0.9270194892047279 |
| KNN |	0.9444444444444444 | 0.9873164308539565 | 0.9351787101787101 | 0.9366813294232649 |	0.9358053104888548 | 0.9144496835852509 |
| Decision Tree | 0.9523809523809523 | 0.9584911470640085 |	0.9466688693980535 | 0.9403225806451613 | 0.9430875319764209 | 0.9265709532566386 |

The **F1-score and Recall-score** are used to compare the model because correctly detecting degraded cells is more important than accuracy alone.

## 5. Observation About Model Performance


| Model Name | Observation about model performance |
|------------|-------------------------------------|
| Logistic Regression | Logistic Regression achieved 94.56% F1-score, an AUC of 99.34%, and recall-score of 94.74%. The strong performance indicate that majority part of the classification problem can be explained using simple relationship boundary as it have linear relationship between feature and target. |
| Decision Tree | Logistic Regression achieved almost similar performance to Logistic Regression with 94.31% F1-score, an AUC of 95.85%, and recall-score of 94.03%. As decision tree depends on single tree, it is less robust compared to Random Forest. Moreover, root node is always fix based on the impurity value therefore decision tree always start with fix node. Also, it completely neglect some of the feature even though it might have low importance compared to other features. Because of this drawback, it performance is less robust compared to Logistic Regression and Random Forest. |
| kNN | kNN achieved 93.58% F1-score, an AUC of 98.73%, and recall-score of 93.67%. This indicates that samples with similar battery characteristics generally belong to similar health states. However, kNN is sensitive of feature scaling, number of neighbors selected as well as distance metric used. It performed slightly worse than Random Forest, Logistic Regression and Decision Tree. |
| Naive Bayes | Naive Bayes showed the lowest performance among all the models. It achieved F1-score 84.45%, an AUC of 96.31% and recall-score of 85.12%. Although its AUC score is relatively high, it lower classification metrics suggest that it has difficulty correctly assigning samples to the appropriate health class. One possible reason is that Naive Bayes assume that input features are conditionally independent, whereas battery parameters are often strongly related. |
| Random Forest | Random Forest achieved the best overall result for battery diagnostic classification problem. It achieved F1-score of 96.47%, an AUC of 99.77% and recall-score of 96.17%.This indicate that algorithm provided good balance between detecting correct health class and minimizing false classification. It strong performance is due to number of different decision tree constructed, which help to capture non-linear relationships and interaction between different features. Moreover, it doesn't neglect any feature even though importance is less, as Decision Tree did.  |
| Overall Winner | Random Forest |