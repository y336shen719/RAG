---
title: Technical Interview Answers 2026
category: interview
subcategory: technical
tags: [statistics, ml, sql, data-cleaning, modeling, big-data]
priority: medium
last_updated: 2026-02
---

# Technical Interview Bank

---

## 1. Data Engineering & Big Data

### How do you handle data that is too large for memory?

There are multiple strategies depending on scale:

1. Reduce precision if acceptable (e.g., float64 → float32)
2. Chunk processing: process data in batches and aggregate results
3. Push down computation into SQL/data warehouse
4. Use distributed frameworks like Spark
5. Sampling for exploratory analysis

Key principle: move computation to where the data lives.

---

### What is schema?

A schema is the blueprint of a database.  
It defines:

- Tables  
- Columns and data types  
- Primary and foreign keys  
- Constraints  

It ensures structure and consistency.

---

### What is data lineage?

Data lineage tracks where data originates from and how it is transformed across systems.

It improves transparency, debugging, and governance.

---

### What is metadata?

Metadata is data about data — it describes structure, meaning, and usage of data.

---

### What is primary key and foreign key?

- Primary key uniquely identifies each row in a table.
- Foreign key references a primary key in another table.

They enforce relational integrity.

---

### COUNT(*) vs COUNT(column)

- COUNT(*) counts all rows.
- COUNT(column) counts non-null values.
- COUNT(*) - COUNT(column) gives number of nulls.

---

## 2. Data Cleaning & Preprocessing

### How do you conduct data cleaning?

My process:

1. Understand the schema and data dictionary  
2. Profile missing rate, duplicates, outliers  
3. Handle issues:
   - Missing → drop, impute, indicator  
   - Duplicates → define and remove  
   - Outliers → detect and decide error vs valid extreme  
4. Validate results  
5. Make cleaning reproducible via pipeline  

---

### How do you handle missing values?

Steps:

1. Profile missing pattern  
2. Identify mechanism (MCAR, MAR, MNAR)  
3. Drop if missing rate too high  
4. Impute:
   - Numerical → mean/median/rolling mean  
   - Categorical → mode/unknown  
5. Add missing indicator when informative  

---

## 3. Statistics & Inference

### What is p-value?

Assuming the null hypothesis is true, the p-value is the probability of observing results as extreme or more extreme than what we observed.

If p < 0.05, we reject the null hypothesis.

---

### What is Central Limit Theorem?

For i.i.d. samples with finite variance, as sample size increases, the sampling distribution of the mean approaches normal distribution — even if original data is not normal.

---

### What is a confidence interval?

If we repeatedly sample and compute 95% CI each time, about 95% of intervals would contain the true parameter.

---

### Type I vs Type II Error

- Type I: False positive (reject true null)
- Type II: False negative (fail to reject false null)

---

### Correlation vs Causation

- Correlation: variables move together
- Causation: changing one variable directly changes another

Causation requires controlled experiment.

---

## 4. Modeling & ML Concepts

### Walk me through a modeling project

I follow PPDAC:

1. Problem → define business goal & metrics  
2. Plan → experimental design  
3. Data → cleaning, validation  
4. Analysis → EDA + modeling  
5. Conclusion → interpretation + actionable plan  

---

### What is bias-variance tradeoff?

- High bias → underfitting
- High variance → overfitting

Goal: balance both to generalize well.

---

### What is underfitting vs overfitting?

- Underfitting: model too simple
- Overfitting: model too complex and sensitive to training data

---

### What is collinearity?

Highly correlated predictors in regression cause unstable coefficients.

Solution:
- Drop one feature
- Use VIF
- Regularization

---

### What is decision tree vs random forest?

- Decision tree: single tree model
- Random forest: ensemble of trees using bagging

Random forest reduces variance.

---

## 5. Product Analytics & Business Analytics

### What is A/B testing?

Randomly split users into control and treatment groups to measure causal impact.

---

### What is funnel analysis?

Funnel analysis tracks user progression through steps toward a goal and identifies drop-off points.

---

### How do you validate analysis accuracy?

- Sanity check  
- Sensitivity analysis  
- Peer review  
- Cross-method validation  

---

## 6. MLOps & Advanced Topics

### What is data shift vs concept drift?

- Data shift: input distribution changes  
- Concept drift: relationship between input and output changes  

Solutions:
- Monitoring
- Retraining
- Adaptive models

---

### What is incremental learning?

Continuously updating model with new data without full retraining.

---

### What is catastrophic forgetting?

When a model forgets previously learned information after learning new data.

---

