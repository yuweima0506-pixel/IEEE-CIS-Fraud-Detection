# Readme 

Kaggle competition: detect fraudulent transactions using the IEEE-CIS dataset. This project explores a two-stage pipeline — a strong LightGBM tabular baseline followed by a Graph Convolutional Network (GCN) embedding augmentation
**Final result:** 
| Metric         | Score      |
|----------------|------------|
| Validation AUC | **0.9410** |
| Public LB      | 0.9406     |
| Private LB     | **0.9183** |

⠀
# Dataset
| **Split** | **Rows** | **Columns** |
|:-:|:-:|:-:|
| Train | 590,540 | 434 |
| Test | 506,691 | 433 |
1. **Fraud rate:** 3.50% — imbalanced but acceptable (normally less than 1%)
2. Two source tables merged on TransactionID: train_transaction + train_identity
3. Features span transaction metadata, card attributes, identity fields, and Vesta-proprietary V/C/D features

Transaction dataset 
* TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
* TransactionAMT: transaction payment amount in USD
* ProductCD: product code, the product for each transaction
* card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
* addr: address
* dist: distance
* P_ and (R__) emaildomain: purchaser and recipient email domain
* C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
* D1-D15: timedelta, such as days between previous transaction, etc.
* M1-M9: match, such as names on card and address, etc.
* Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

Identify dataset:
Categorical Features: DeviceType DeviceInfo id_12 - id_38, network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions. 

column explaination comes from [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/101203)
⠀
# Project Structure
### 
## Stage 1 — Tabular Baseline
https://github.com/yuweima0506-pixel/IEEE-CIS-Fraud-Detection/blob/main/README.md#:~:text=Feature-,Engineering,-.pdf
This document covers the feature engineering pipeline applied to the competition. We combined domain knowledge with dataset-specific characteristics to construct effective predictive features. The table below summarises the final performance achieved at this stage
| Metric | Score |
|--------|-------|
| Val AUC | **0.9390** |
| Public LB | 0.9358 |
| Private LB | **0.9110** |

## Stage 2 — Graph Embeddings
