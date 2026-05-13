# Feature Engineering

---

## Table of Contents

- [Baseline Model](#1-baseline-model)
- [Improved Model — Encoding Experiments](#2-improved-model--encoding-experiments)
- [User ID Construction](#3-user-id-construction)
- [Categorical Aggregation Features](#4-categorical-aggregation-features)

---

## 1. Baseline Model

**Notebook:** `Baseline_model.ipynb`  
**Features:** all 432 raw columns

### Categorical Handling

- All 50 categorical features encoded via `LabelEncoder` → integer
- Unknown values in val/test mapped to `'unknown'` before encoding
- No `categorical_feature` parameter passed to LightGBM — all features treated as numeric

> ⚠️ Label encoding only was used to establish a quick baseline. It is not appropriate for production use on high-cardinality categoricals.

### Train / Validation Split

Time-based split on `TransactionDT`:
![](Feature%20Engineering/image.png)
| Split | Rows | Fraud rate |
|-------|------|-----------|
| Train | 472,432 | 3.51% |
| Val | 118,108 | 3.44% |

### Distribution Shift Check

Key features with unknown rates in test vs val:

| Feature | Val unknown % | Test unknown % |
|---------|--------------|----------------|
| id_31 | 6.83% | 22.06% |
| id_13 | 3.10% | 13.74% |
| id_30 | 0.12% | 4.39% |
| DeviceInfo | 0.32% | 3.39% |
| card1 | 1.09% | 2.22% |

Unknown rates are within acceptable range — no features dropped.

### Model Config

```
n_estimators   = 500
learning_rate  = 0.05
num_leaves     = 256
early_stopping = 50 rounds
best_iteration = 207
```

### Performance

| Metric | Score |
|--------|-------|
| Val AUC | 0.9208 |
| Public LB | 0.9229 |
| Private LB | 0.8974 |

> Private LB degrades relative to public LB, suggesting the model does not generalise well to the far-future test period.

### Top 10 Features

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | card1 | 3508 |
| 2 | TransactionDT | 3222 |
| 3 | TransactionAmt | 2737 |
| 4 | card2 | 2487 |
| 5 | addr1 | 2299 |
| 6 | D15 | 1340 |
| 7 | dist1 | 1278 |
| 8 | P_emaildomain | 1166 |
| 9 | C13 | 1025 |
| 10 | card5 | 946 |


### Zero Importance Features (removed)

21 features with zero importance removed and added to `delete_cols`:

`id_29`, `V191`, `V196`, `V28`, `V241`, `V107`, `V117`, `V119`, `V120`, `V113`, `V305`, `V88`, `V89`, `V27`, `V41`, `V65`, `V325`, `V327`, `V68`, `id_16`, `id_27`

---


## 2. Improved Model — Encoding Experiments

**Notebook:** `model_v4.ipynb`

### New Engineered Features

* #### TransactionDT Decomposition
![](Feature%20Engineering/image%202.png)<!-- {"width":654} -->
| Feature | Description |
|---------|-------------|
| `hour` | hour of day (0–23) |
| `dayofweek` | day of week (0–6) |
| `day` | absolute day index |
| `hour_sin`, `hour_cos` | cyclic encoding, period = 24 |
| `dayofweek_sin`, `dayofweek_cos` | cyclic encoding, period = 7 |

Sin/cos encoding is used to represent cyclical time — **the model can learn that hour 23 and hour 0 are adjacent.**

* #### TransactionAmt Features
![](Feature%20Engineering/image%204.png)
![](Feature%20Engineering/image%205.png)
![](Feature%20Engineering/image%206.png)
| Feature | Description |
|---------|-------------|
| `log_amt` | log1p of raw amount |
| `amt_is_round` | 1 if amount has no decimal component |
| `amt_decimal` | decimal part rounded to 2 places |
| `amt_last_digit` | last digit for round amounts; −1 for non-round |

`amt_decimal` became the single strongest feature across all model versions.

* #### Log Transform for High-Skew Numerics

1. Skewness computed on train only
2. All numeric features with `|skew| > 10` receive an additional `col_log = log1p(clip(col, 0))` column
3. Original column retained; 195 log columns added in total

* #### D15 Binning
![](Feature%20Engineering/image%207.png)
`D15` replaced with `D15_bin`: 10 quantile bins computed on train, applied to val/test as a categorical feature.

### Categorical Encoding Experiments

#### Experiment 1 — High-Cardinality Encoding Strategy

strategies tested for features with > 255 unique values:

| Aspect             | OriAscat_addFreq` | removeOri_addFreq` | OriasLabelencoder_addFre` | label_encode |
|--------------------|-------------------|--------------------|---------------------------|--------------|
| **count freq**     | ✅                 | ✅                  | ✅                         | ❌            |
| **label encode**   | ❌                 | ❌                  | ✅                         | ✅            |
| **original value** | ✅                 | ❌ removed          | ❌ removed                 | ❌            |
| **Val AUC**        | 0.9061            | 0.9206             | 0.9257                    | 0.9208       |
| **Public LB**      | 0.9021            | 0.9249             | 0.9274                    | 0.9229       |
| **Private LB**     | 0.8799            | **==🔴0.9014==**    | 0.9010                    | 0.8974       |

**Key findings:**
1. Passing high-cardinality features as native LightGBM categoricals triggers the internal 255-bin limit, severely degrading performance (−0.0147 AUC ).
2. Frequency encoding alone (`removeOri_addFreq`) without label encoding achieve higher private score.
3. Combining label encoding and frequency encoding (`OriasLabelencoder_addFreq`) gives higer score on Public, but fail at Private. 

#### Experiment 2 — Sliding Window vs Global Aggregation

Aggregation features computed on `TransactionAmt` grouped by user/card keys:

| Method | Aggregations | Val AUC | Public LB | Private LB |
|--------|-------------|---------|-----------|------------|
| Sliding window | cnt, avg, min, max | 0.9156 | 0.9165 | 0.8953 |
| **Global** | cnt, avg, min, max, p25, p50, p75 | **0.9310** | **0.9310** | **0.9062** |

Global aggregation outperforms sliding window despite being less realistic for production:
- Both train and test benefit from future information in the full dataset
- Significant card overlap exists between train and test, making global statistics informative

### Final Performance 

| Metric | Score |
|--------|-------|
| Val AUC | **0.9310** |
| Public LB | 0.9310 |
| Private LB | 0.9062 |

---





## 3. User ID Construction

**Notebook:** `model_v5.ipynb`

Aggregation features are only meaningful when grouped by the correct user/cardID identity. Without a true identity, aggregating conflates multiple users, diluting fraud signals with legitimate transaction history.

> Aggregation features measure deviation from a user's historical behaviour. The signal is lost if transactions from different users are mixed in the same group.

### Candidate Formulas

Based on Chris Deotte's approach, user identity is approximated by combining card and address features:

| ID | Formula | Distinct count | Mean tx/user | p50 | p75 |
|----|---------|---------------|-------------|-----|-----|
| user_id1 | card1 + addr1 + D1_norm | 217,850 | 2.71 | 1 | 3 |
| user_id2 | card1 + card2 | 14,525 | 40 | 4 | 13 |
| user_id3 | card1 + addr1 | 39,974 | 14 | 2 | 7 |
| user_id4 | card1 + addr1 + P_emaildomain | 90,375 | 6 | 2 | 4 |

`user_id1` is the most granular (fine-grained) but too sparse — most users have only 1 transaction per half-year.

### Experiment 1 — Individual ID Selection

| user_id | LogLoss | Val AUC |
|---------|---------|---------|
| baseline | 0.08139 | 0.9310 |
| user_id1 | **0.07850** | **0.9362** |
| user_id2 | 0.08215 | 0.9293 |
| user_id3 | 0.08163 | 0.9325 |
| user_id4 | 0.08206 | 0.9324 |

`user_id2` eliminated. Remaining IDs tested in combination:

        1, user_id1 + user_id3+ user_id4==> AUC 0.9389
	==🔴**2 ,user_id1 + user_id3 ==> AUC 0.9390**.== 
	3.  user_id1 + user_id4 ==> AUC 0.9364
### Final Performance 

| Metric | Score |
|--------|-------|
| Val AUC | **0.9390** |
| Public LB | 0.9358 |
| Private LB | **0.9110** |

---

## 4. Categorical Aggregation Features

**Notebook:** `model_v6.ipynb`

Beyond numeric aggregation, categorical behaviour features were constructed to capture consistency patterns within user groups:

| Feature type | Description |
|---|---|
| `n_distinct_{col}` | number of distinct values seen for this user |
| `mode_cnt_{col}` | frequency of the most common value for this user |
| `is_mode_{col}` | 1 if current transaction value matches the user's mode |

Features constructed for: `card3`, `card4`, `card5`, `card6`, `addr1`, `addr2`, `R_emaildomain`

### Feature Selection

Each of the 12*3 candidate features was added individually to the model and validated. Only features with positive AUC contribution were retained.
| Feature | AUC | Delta |
|---------|-----|-------|
| n_distinct_R_emaildomain | 0.939193 | +0.000193 |
| n_distinct_card6 | 0.939068 | +0.000068 |
| mode_cnt_R_emaildomain | 0.939066 | +0.000066 |
| n_distinct_addr1 | 0.939044 | +0.000044 |
| is_mode_card6 | 0.939044 | +0.000044 |
| n_distinct_card3 | 0.939044 | +0.000044 |
| is_mode_card3 | 0.939044 | +0.000044 |
| n_distinct_card4 | 0.939044 | +0.000044 |
| is_mode_card4 | 0.939044 | +0.000044 |
| is_mode_addr2 | 0.939044 | +0.000044 |
| is_mode_addr1 | 0.939044 | +0.000044 |
| is_mode_card5 | 0.939044 | +0.000044 |
| n_distinct_P_emaildomain | 0.938965 | -0.000035 |
| mode_cnt_card2 | 0.938521 | -0.000479 |
| is_mode_DeviceType | 0.938472 | -0.000528 |
| is_mode_card2 | 0.938358 | -0.000642 |
| mode_cnt_DeviceInfo | 0.938181 | -0.000819 |
| mode_cnt_DeviceType | 0.938181 | -0.000819 |
| is_mode_P_emaildomain | 0.938082 | -0.000918 |
| n_distinct_card5 | 0.938042 | -0.000958 |
| mode_cnt_card5 | 0.937954 | -0.001046 |
| mode_cnt_addr2 | 0.937941 | -0.001059 |
| is_mode_R_emaildomain | 0.937922 | -0.001078 |
| n_distinct_DeviceInfo | 0.937712 | -0.001288 |
| mode_cnt_card4 | 0.937465 | -0.001535 |
| n_distinct_DeviceType | 0.937395 | -0.001605 |
| n_distinct_addr2 | 0.937252 | -0.001748 |
| mode_cnt_card6 | 0.937230 | -0.001770 |
| mode_cnt_P_emaildomain | 0.937156 | -0.001844 |
| n_distinct_card2 | 0.936951 | -0.002049 |
| mode_cnt_addr1 | 0.936860 | -0.002140 |
| mode_cnt_card3 | 0.936683 | -0.002317 |
| is_mode_DeviceInfo | 0.935907 | -0.003093 |

### Final Performance (model_v6)

| Metric | Score |
|--------|-------|
| Val AUC | **0.9397** |
| Public LB | 0.9361 |
| Private LB | **0.9099** |
