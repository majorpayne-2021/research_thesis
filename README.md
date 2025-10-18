# Detection, Ranking and Visualisation of Money Laundering Networks on the Bitcoin Blockchain

This repository contains the full implementation of the research thesis:  
**“Detection, Ranking and Visualisation of Money Laundering Networks on the Bitcoin Blockchain”**  
by **Jennifer Payne (RMIT University)**.  

The project develops a multi-stage analytical framework for detecting, ranking, and visualising money laundering activity on the Bitcoin blockchain using machine learning and network analysis. The workflow is built around the Elliptic++ dataset and is designed for transparency, reproducibility, and investigative interpretability.  

---

## Project Overview  

The framework consists of five main modelling stages and several supporting notebooks. Each stage builds upon the outputs of the previous step, forming an end-to-end Anti-Money Laundering (AML) detection pipeline:  

| **Model Step** | **Notebook Name** | **Purpose** |
|----------------|-------------------|--------------|
| **1** | *Model Step 1 – Elliptic++ Data Preprocessing* | Cleans and prepares the Elliptic++ transaction and wallet datasets, merges class labels, engineers new features, and exports clean tables to BigQuery. |
| **2** | *Model Step 2 – Txn Classification (RF w HP Tuning)* | Trains and tunes a Random Forest model to classify transactions as licit or illicit and predicts classes for unlabelled transactions. |
| **3** | *Model Step 3 – Subnetwork Development* | Builds illicit-only subnetworks from classified transactions using breadth-first search (BFS) expansion and exports structured subnetwork data. |
| **4** | *Model Step 4 – Ranking Nodes in Subnetwork* | Calculates composite node rankings based on PageRank, transaction value, and degree metrics, and compares them with other ranking algorithms. |
| **5** | *Model Step 5 – Subnetwork Visualisation* | Generates transaction-to-transaction and address-to-address visualisations and summary tables for each illicit subnetwork. |

---

## Supplementary Notebooks  

| **Notebook** | **Purpose** |
|---------------|-------------|
| *Validation of Bitcoin Addresses* | Spot-checks sample addresses and transactions using public blockchain explorers (e.g., BTC Scan) to verify data reliability. |
| *Txn Classification – LR (w/o HP Tuning)* | Implements a baseline logistic regression classifier without hyperparameter tuning. |
| *Txn Classification – LR (w HP Tuning)* | Extends logistic regression with cross-validated hyperparameter tuning to improve model recall for the illicit class. |
| *Txn Classification – RF (w/o HP Tuning)* | Establishes an untuned Random Forest baseline for comparison with tuned models. |
| *Illicit Subnetwork Analysis* | Examines subnetwork size, depth, and connectivity metrics to identify structural patterns of illicit activity. |

---

## Python Modules  

| **Module** | **Purpose** |
|-------------|-------------|
| `txn_classification_hyper.py` | Handles preprocessing, training, hyperparameter tuning, and model evaluation for binary transaction classification. |
| `txn_subnetworks.py` | Constructs illicit-only subnetworks from classified transactions and provides reporting and visualisation tools. |
| `txn_rank.py` | Implements node ranking methods (PageRank, degree, composite rank) and correlation comparisons across ranking systems. |

---

## Dataset  

This research uses the **Elliptic++ dataset**, which extends the original Elliptic dataset with real Bitcoin wallet addresses and transaction-level detail, enabling address-level validation and enhanced network analysis.  

- **Elliptic++ Paper and Dataset:** [https://github.com/git-disl/EllipticPlusPlus](https://github.com/git-disl/EllipticPlusPlus)  
- **Elliptic (Original) Dataset:** [https://www.kaggle.com/datasets/ellipticco/elliptic-data-set/data](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set/data)  

---

## Citation  

If using this repository or code in related academic work, please cite:  

> Payne, J. (2025). *Detection, Ranking and Visualisation of Money Laundering Networks on the Bitcoin Blockchain.* Master’s Thesis, RMIT University.  

---

## Repository Structure  

```text
rmit_master_thesis/
│
├── model_modules/
│   ├── txn_classification_hyper.py
│   ├── txn_subnetworks.py
│   └── txn_rank.py
│
├── model_notebooks/
│   ├── Model Step 1 – Elliptic++ Data Preprocessing.ipynb
│   ├── Model Step 2 – Txn Classification (RF w HP Tuning).ipynb
│   ├── Model Step 3 – Subnetwork Development.ipynb
│   ├── Model Step 4 – Ranking Nodes in Subnetwork.ipynb
│   └── Model Step 5 – Subnetwork Visualisation.ipynb
│
├── other_notebooks/
│   ├── Validation of Bitcoin Addresses.ipynb
│   ├── Txn Classification – LR (w/o HP Tuning).ipynb
│   ├── Txn Classification – LR (w HP Tuning).ipynb
│   ├── Txn Classification – RF (w/o HP Tuning).ipynb
│   └── Illicit Subnetwork Analysis.ipynb
│
└── ML_Networks_on_Bitcoin_Blockchain_Payne.pdf

---

## Author  

**Jennifer Payne**  
Master of Data Science, RMIT University  
GitHub: [majorpayne-2021](https://github.com/majorpayne-2021)
