---
title: technical_2026
category: interview_bank
tags: [big data, p-value, teamwork, data cleaning, central limit theorem]
priority: medium
last_updated: 2026-02
---

**Technical Problems**

1.  [How to handle big data which is too large to your
    memory]{.underline}

-   Compression

Reduce the accuracy of your numerical value, e.g. double float to single
float

-   Chunking

> Divide into chunks and do preprocessing and aggregation such as take
> average, min, max for each chunk. Then merge them together

-   Pushdown computation

Do heavy filtering, cross-table joining in database/data warehouse using
SQL before

-   Distributed data processing framework:

distributed data processing framework like spark can run big data across
multiple machines parallelly

2.  [How you conduct the data cleaning?]{.underline}

-   Understand the data

Check the schema

walk through the data dictionary

understand the data definition, unit, domain

-   Quick analysis/profiling to find major issues

Look into the data distribution, missing rate, duplicate rate, locate
outliers

-   Handle issues

```{=html}
<!-- -->
```
-   Missing value: drop, impute with some value, create a missing value
    indicator

-   Duplicate: figure out what duplicate means, remove duplicate

-   Outliers: identify if it is error or extreme value,

```{=html}
<!-- -->
```
-   Error: such as wrong decimal point we can correct it

-   Extreme value: clip data, keep it but use more robust analysis
    method

```{=html}
<!-- -->
```
-   Data consistency: standardize data type & format, string data with
    lower/uppercase

```{=html}
<!-- -->
```
-   Data quality validation: spot check, especially look into
    distribution after processing to check shift

-   Fit the data cleaning workflow into a pipeline, to make it
    reproducible

3.  [How would you handle missing values in a dataset]{.underline}

-   Quick Missing Value Profiling

Missing rate and pattern by column, row; over time, group

clarify the missing mechanism (why data is missing): MCAR (missing is
random), MAR (missing is related to observed variables), MNAR (missing
is related to unobserved value)

-   Drop

```{=html}
<!-- -->
```
-   Columns: a column with too many missing values, drop it

-   Rows: a row with too many missing values and sample size is large
    enough, drop it

```{=html}
<!-- -->
```
-   Imputation based on data type

```{=html}
<!-- -->
```
-   Numerical: mean (smooth), median (robust), rolling mean/median for
    time series data

-   Categorical: mode, unknown

```{=html}
<!-- -->
```
-   Treat missing value as a usable signal

Add a missing value indicator, common in linear model

4.  What is the difference between a null field and a blank field in
    SQL?

-   Null indicates there is no value within a database field for a given
    record. It does not mean zero because zero is a value.

```{=html}
<!-- -->
```
-   Blank is an empty string, or a white-space only string

5.  Explain the difference between correlation and causation

-   Correlation: Two variables move together or not. It describes a
    pattern, not a "why."

-   Causation: impose a treatment to the user/product to see the effect.
    Changes in variable A directly cause changes in variable B with
    other factors held controlled.

6.  How do you determine which visualization is most proper for your
    data?

-   Determine what question the visual is used to answer

-   Determine the plot type based on data type and structure

Categorial VS. Numerical: bar plot

Distribution: density/frequency histogram, box plot

Time series: line graph

Proportion: pie chart, stacked bar chart

-   Determine who is the viewer

```{=html}
<!-- -->
```
-   Non-tech: choose plot type that are familiar with, like line chart,
    able to understand at the first glance

-   Tech: with statistical background, box plot / QQ plot, deeper
    insights

7.  Walk me through how you would approach a new data analysis project?

-   Use the PPDAC framework

```{=html}
<!-- -->
```
-   [Problem]{.underline}: clarify business problem we try to solve

> Set primary, secondary, guardrail metrics, and then success criteria

-   [Plan]{.underline}: experimental design

> Build data requirement list, determine what data we need

-   [Data]{.underline}: pull data

> Data cleaning, data validation, sanity check
>
> Build data dictionary, ER diagram

-   [Analysis]{.underline}: EDA, A/B testing

-   [Conclusion]{.underline}: data visualization

> Limitation
>
> Actionable plan

8.  What statistical tests would you use to compare two groups?

-   Parametric tests such as t-test

-   Non-parametric tests such as u-test

9.  How do you validate the accuracy of your analysis?

-   Sensitivity check

-   Sanity check

-   Peer review

-   Logic check

10. How to you handle the unexpected results in your analysis?

-   Cross-check: use different methods to verify the same problem

> Check if it is error or just an extreme value

-   Error: diagnostic mindset, isolate problem and locate real root
    cause

> Change methodology

-   Extreme value, try to explain it

11. P-value

Assuming the null hypothesis is true, the probability of observing as
extreme as/ or extremer than the current observed data

P-value \< significance level \~ 0.05, we reject the hull hypothesis

12. Central limit theorem

for i.i.d. samples with finite variance, as sample size n becomes large,
the sampling distribution of the sample mean becomes approximately
normal, even if the data are not normal.

Take sample of 5 people to compute their average incomes, the
distribution of results will be very likely skewed, but take sample of
1000 people to compute their average incomes, and repeat many times, the
distribution will be approximately normal

13. a/b testing

14. Linear Regression

15. What is bias and variance trade off

Too simple assumption will cause systemic bias, which is underfitting

Too complex model will cause high variance, which is the model is very
sensitive to different training dataset, overfitting (predication
changes a lot across different training datasets)

We try to find a balanced point to make model has good performance on
both training and testing data.

16. confidence interval

For a parameter (we want to estimate), repeat the same sampling process
and compute a 95% CI each time, about 95% of those intervals would
contain the true parameter.

17. type I and type II errors

-   type I: no effect, but we think it has effect (Null hypothesis is
    true, but we reject it) -- 假阳性 fake positive

-   type II: has effect, but we think it no effect (Null hypothesis is
    false, but we failed to reject it) -- 假阴性 fake negative

18. unbalanced data

-   stratified sampling

19. correlation analysis

-   define target variate (DAU: daily active user) and candidate
    explanatory variates

-   conduct EDA,

```{=html}
<!-- -->
```
-   time series map for analysis on seasonality

-   scatterplot for two continuous numerical variates

```{=html}
<!-- -->
```
-   Spearman correlation analysis (non-linear, robust to outliers and
    skew, for heavy-tailed product data)

Pearson correlation analysis (usually as a linear reference/supplement)

e.g. DAU increases during tax season; video length negatively correlated
with completion rate

20. What is funnel analysis?

Funnel analysis breaks down how customers reach a goal into a sequence
of steps. For each step, we measure how many users enter it, how many
progress to the next step, how many drop off, and how long it takes to
move forward. It's especially useful for identifying friction points in
the user journey.

21. What is the Agile methodology?

Agile methodology is a method for delivering projects

-   Iterative delivery: delivery MVP first and refine

-   Fast feedback

-   Adapting to change

-   Define epic (high level goal we want to reach, e.g. we want to
    improve new user experience), user stories (break epic into many
    smaller achievable tasks, e.g. new user can sign up with Google
    account, new user can sign up with their mobile phone number)

Use backblog to manage the user stories

Deliver small progress every 1-2 sprint

22. Modeling problem

-   Experiment design:

Understand to make what business decision

Determine the experiment unit/unit of predication

Define our label

-   Data

What data we need

Time window of data, e.g. 10-year/20-year

-   EDA/profiling

Missing value, data distribution & outliers

Slicer by time/people segmentation to check seasonality, group features

-   Data preprocessing

EDA and data preprocessing don't happen in a strict sequence

-   Feature engineering

Create new feature, lagging/rolling

Reduce dimension, PCA

-   Split dataset

Training, testing, sometime maybe validation

-   Model fitting and tuning

Baseline model, explainable, linear regression/logistic regression

Advanced model, tree

-   Evaluation

Multiple metrices, log loss + AUC + SHARP ratio

Evaluate from different perspectives: accuracy, stability

23. What is underfitting? What is overfitting?

Underfitting: change a more complex model, add/create new features,
weaker regularization, boosting

24. What is collinearity?

In regression model, two features are highly correlated to each other,
this will make model confused about which feature is making real effect

Will cause training not stable (change a different batch of training
data, the coefficient will change a lot)

Less explainable

Drop one of the correlated features, VIF could be the solution to reduce

25. What is decision tree? What is random forest?

26. What is schema?

-   Database blueprint, define how data is structured

-   Tables, columns & data types, primary key & foreign key, constraints

27. What is constraint?

-   Rules enforced by database to ensure the data integrity

-   NOT NULL, UNIQUE, CHECK(自定义条件必须满足), DEFAULT(自动填充)

28. What is metadata?

-   Data about data

documentation and descriptors that make data understandable and
governable

29. What is data lineage?

-   Where the data came from and how it was transformed

30. What is difference between NA and NULL in SQL?

NULL is the missing value in SQL. NA usually is a string placeholder,
empty string, or a white-space only string

NULL is part of the 3-valued logic (true/false/null), any comparison
with NULL is NULL (NULL=NULL's result is NULL)

31. What is primary key and foreign key?

-   Uniquely identifies each row in a table

-   A column (or columns) that references a primary key in another table

32. What is data integrity?

-   Correctness, uniqueness, consistency

-   Correctness: values reflect the reality

-   Uniqueness: things should be unique are actually unique

-   Consistency: facts are consistent across tables

33. What is SCD (Slow changing dimension) Type 1 and Type 2?

-   Type 1: every change, we overwrite and only keep the latest
    information,

-   Type 2: when change, we insert a new row instead of overwriting,
    maybe we add timestamp, support as-of analysis

34. COUNT(\*) VS. COUNT(col)

-   COUNT(col) will count all non-null values in that col

-   COUNT(DISTINCT col)

-   COUNT(\*) will count rows, including NULL

-   We use COUNT(\*)-COUNT(col) to get number of NULL

35. How to ensure the data security and database security

-   Least privilege: give the user the minimum access they need

-   Role-based access control: different roles are assigned with
    different R/W access

Different access has different view, some sensitive info are masked to a
specific group of users

36. How to build a database

-   Understand what the database will be used for

What demand/requirements meet: check by region? Check by time?

-   Setting about database

On-premises/on cloud; real-time/daily refresh/weekly refresh

Do we need separate into developing, testing and production
environments?

-   Decide what tables we should have in the database

Decide each tables' grain: what a single row represents? A user/an
account (a user can have multiple accounts)

Decide columns and data types

Dimension tables e.g. assign a unique ID, Users(userID, name, age,
location)

Event tables e.g. Payments, Transactions, account_balance_snapshot

Relationship, primary key & foreign key

Define constraints to ensure data integrity, value of balance should be
non-negative

-   Enforce data security and access control

Principle of least privilege: give the users the minimum access they
need

Different role has different read/write access

Different role has different view of data, some info can be masked to a
specific group of users

37. What is data shift? What is concept drift? How to solve them?

38. What is incremental learning in MLOps?

39. What is the difference of inductive bias between CNN and
    Transformer?

40. Why GAN has larger variance than diffusion?

41. What is Catastrophic Forgetting?

42. What is normalization of schema?
