# Supervised Graph Embedding

---

## Graph Structure

### Nodes
Each transaction is a node. Features are all 631 tabular features after preprocessing in engineering.

**Why transactions as nodes (not users/cards)?**
- No complete `user_id` or `card_id` is provided in the dataset and no trustworthy entity cane be directly derived. 
- Payer and receiver information is asymmetric — cannot model as a bipartite graph

### Edges
Two transactions are connected if they share the same value on **any** of the following columns:

```python
### initial version ###
edge_cols = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'user_id1', 'user_id3']
```

**Static graph:** 
edge weight = number of shared attribute columns.

**Temporal graph:** 
edge weight = exponential time decay

```
w = exp(−Δt / (86400 × λ)),   λ = 14–30 days
```

Only edges within a `max_time_gap = n days` are created, a prolonged connection carries less discriminative infor than a recent one. 
![](Supervised%20Graph%20Embedding_after/image%204.png)


### Edge Scale Controls

| Parameter          | Value | Purpose                                                      |
|--------------------|-------|--------------------------------------------------------------|
| `min_shared_attrs` | 2     | avoid accidental connections from a single shared attribute  |
| `max_value_count`  | 500   | skip overly common values (high-cardinality noise, eg: public WIFI) |
| `max_group`        | 200   | cap group size per attribute value, newest connection left   |

---

## Training Pipeline

### Stage 1 — OOF Embeddings (Train)

Train data is sorted by `TransactionDT` and split into 10 time-ordered folds (~59k transactions each).

A sliding window of 6 folds generates OOF embeddings for folds T6–T10:

| Round | Graph nodes | Supervised | Predict (OOF) |
|-------|------------|------------|---------------|
| 1 | T1–T6 | T1–T5 | T6 |
| 2 | T2–T7 | T2–T6 | T7 |
| 3 | T3–T8 | T3–T7 | T8 |
| 4 | T4–T9 | T4–T8 | T9 |
| 5 | T5–T10 | T5–T9 | T10 |

The predict fold's labels are **masked** during GCN training at each round, preserving OOF purity.

**Chain initialisation:** model weights from round N are passed as initial weights to round N+1. This promotes representation consistency across rounds without retraining from scratch.

> Note
> T1–T5 have no OOF embeddings at this stage and those dara are abandoned at stage2. In a production system with continuous data, earlier folds would be covered by preceding windows.

### Stage 2 — Classifier Training

```
Features:  631 tabular + 32 GCN embeddings = 663 total
Train set: T6–T9 (OOF embeddings available)
Val set:   T10
```

Theoretical minimum data: ~70k samples (≈100× number of parameters for 663 dimensions). 
### Stage 3 — Test Embedding Generation (most important part)

Test transactions all occur after the training period. Several strategies were tested for generating test embeddings:
|                    | Training Weight                                              | Using R5 weights                                             |
|--------------------|--------------------------------------------------------------|--------------------------------------------------------------|
| **Fixed window**   | combine T6–T10 + test chunk,<br>re-train from R5 weights<br><br><br>code : <br>Chain Logic with TransactionID Alignment Slice | combine T6–T10 + test chunk, <br>infer forward using Round 5 weights<br><br>code:<br>Fixed T6~T10 + Frozen R5 Weights (Forward Pass Only) |
| **Sliding window** | 6-size sliding window <br>re-train from R5 weights<br><br><br>code:<br>Sliding train + accumulating test | 6-size sliding window <br>forward only using Round 5 weights<br><br>code:<br>bseline  |
| Single window      | T6–T10 + all test<br>re-train from R5 weights;<br><br>code : Chain Logic with TransactionID Alignment Slice. N_chunks == 1 | -                                                            |

> **Note**：
> None of this pipeline is suitable for production.
> In a real-time fraud detection system, risk score must be returned within seconds after transaction arrives. It means that graph construction and inference are computational infeasible.  This approach only apply to competition environment.
> 

---

## Architecture 

Three GCNConv layers, each followed by BN + ReLU + Dropout. The third layer outputs 32-dimensional embeddings. A linear classification head (`Linear(32, 1)`) is used during training for supervised fraud signal.

**How GCN worked?**
GCN aggregates neighbour features weighted by node degree (fixed normalisation). The weight matrix W learns which feature nodes matter most via linear transformation (ReLU adds non-linearity) . Across 3 layers, GCN captures up to 3-hop neighbourhood patterns.

> **Note**：
> **Graph construction** matters far more than architecture choice. Replacing the static graph with a temporal cross-timestep graph improves GNN F1 from 0.290 to 0.549 (+89% relative). Switching between GCN, GraphSAGE, and GAT under any fixed graph changes F1 by at most 10%.*[When Graph Structure Becomes a Liability](https://arxiv.org/html/2604.19514v1)*, 2026

**How is graph embedding applied in production?**
1. Subgraph Inference:  Extracting small local subgraph around the incoming transaction instead of preprocessing whole graph. . [GNN-based real-time fraud detection without external graph storage](https://aws.amazon.com/blogs/machine-learning/build-a-gnn-based-real-time-fraud-detection-solution-using-the-deep-graph-library-without-using-external-graph-storage/)
2. Precomputed Embedding: Train model offline periodically. Search for exact embedding for seen entity, For unseen one, mean embedding among similar nodes.
3. Sub second real-time model: AWS GraphStorm v0.5 claims sub-second node classification on graphs with billions of nodes. [Modernize fraud prevention](https://aws.amazon.com/blogs/machine-learning/modernize-fraud-prevention-graphstorm-v0-5-for-real-time-inference/)
In my working scenario, we use the average of a user/card's historical transaction embeddings. Stage 2 training should also use average historical embeddings (not OOF embeddings) to maintain consistency between training and serving.
   

---

## Results

### Embedding-only vs Combined

| Model                    | Val (T10)  | Public LB | Private LB |
|--------------------------|------------|-----------|------------|
| Tabular baseline (T6–T9) | 0.9410     | 0.9405    | 0.9182     |
| GCN embeddings only      | 0.8536     | —         | —          |
| Tabular + GCN embeddings | **0.9427** | —         | —          |


### Test Embedding Strategy Comparison

|                    | Retraining Weight                  | Using R5 weights               |
|--------------------|------------------------------------|--------------------------------|
| **Fixed window**   | Public：0.9284; Priveta: 0.9036     | Public：0.9188; Priveta: 0.8888 |
| **Sliding window** | **Public：0.9292; Priveta: 0.9042** | Public：0.9095; Priveta: 0.8809 |
| Single window      | Public：0.9268; Priveta: 0.9037     | -                              |

**Key observations:**

- Re-training consistently outperforms fixed R5 weights — the model adapts better to test graph context
- Embeddings improve validation AUC (+0.0017) but all strategies degrade public + private LB compared to the tabular-only baseline (0.9405, 0.9182)

---

## Why Embeddings Hurt

6 graph features go into Top 50 importance, we examine 5 of them below: 
| Embedding | imp_norm | shap_norm | corr      | Top correlated feature        |
|-----------|----------|-----------|-----------|-------------------------------|
| emb_14    | 0.14     | 0.10      | **0.526** | TransactionDT / day           |
| emb_15    | 0.10     | 0.23      | 0.464     | V258, V52, V257               |
| emb_16    | 0.06     | 0.11      | 0.469     | V258, V52, V257               |
| emb_18    | 0.10     | 0.14      | 0.322     | V50, id_19_TransactionAmt_avg |
| emb_5     | 0.11     | **0.06**  | 0.279     | V258, V257, V246              |
```
imp_norm: split importance after normalization 
shap_norm : SHAP magnitude after normalization 
cor: P- correlation
```

### 1. Redundancy with existing tabular features

Embeddings have high correlation with tablular features;
- emb_14 have 0.526 p-correlation with Time-relavant features — already fully captured by time features
- emb_15 and emb_16 correlate with the same V features at similar magnitudes — they learned duplicate representations of each other

### 2. Dense continuous features inflate LightGBM importance

Embeddings are complete (no missing values) while tabular features contain many NaN values, eg V-group. 
* LightGBM skips NaN rows when calculating split gain, diluting split count of sparse features. 
* Embeddings win more splits by default — not because they carry more signal, but because they are always present. 
This inflates importance scores without delivering proportional prediction improvement (SHAP magnitude remains low). eg：emb_15, emb_16, emb_5 show high split importance but add no information beyond existing tabular features. 
Such features only provide noise than effective infor which are supposed provided by tabular feature but these features are replaced because of low split gain. 

### 3. Embedding distribution shifts across rounds and train/test

| Embedding | R1 mean | R5 mean | Test mean | Behaviour |
|-----------|---------|---------|-----------|-----------|
| emb_14 | 0.000 | 1.232 | 1.208 | Jumps at R2, then stabilises — encodes training progression |
| emb_15 | 0.321 | 0.146 | 0.146 | Monotonically declining — representation drifts each round |
| emb_16 | 0.324 | 0.153 | 0.152 | Mirrors emb_15 exactly |
| emb_18 | 0.933 | 1.408 | 0.841 | Unstable; test mean diverges significantly from R5 |
| emb_2  | 0.759 | 0.936 | 0.423 | Largest train/test gap — test distribution completely differs |

Emb suffers from **round-to-round instability** which cause large **train/test distribution gap**.


### 4. GCN averaging dilutes fraud signals in mixed neighbourhoods

Fraudsters frequently connect to legitimate users via shared address/email attributes. GCN aggregates neighbour features — fraud node representations get diluted by the majority of legitimate neighbours, reducing the discriminative signal.

---

## Conclusion

A marginal improvement is observable when adding graph embeddings to the validation set (+0.0017 AUC), and 6 graph-related features appear in the top 50 by importance. 
However, public and private LB performance declines significantly, driven by test embedding distribution shift and redundancy with existing tabular features.

The core finding is that the graph representation in this dataset adds noise rather than signal when tabular features are already strong.

