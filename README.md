🏦 Bank Fraud Detector: A machine learning app that analyzes credit card transactions and flags potentially fraudulent ones.

How It Works:
Upload a CSV file containing transaction data (columns V1–V28 + Amount), and the model will predict whether each transaction is legitimate or fraudulent, along with a confidence percentage.

Algorithm: Random Forest Classifier (100 trees)
Dataset: Credit Card Fraud Detection 2023 - 550,000+ real European cardholder transactions
Task: Binary classification (Fraud / Legitimate)
Handling imbalance: Undersampling + class_weight='balanced'

Required CSV Columns
V1, V2, V3, ..., V28, Amount
(V1–V28 are PCA-anonymized behavioral features; Amount is the transaction amount in euros)
