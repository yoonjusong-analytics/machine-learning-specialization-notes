# Week 1: Introduction to Machine Learning

## DeepLearning AI - Machine Learning Specialization

## Topic Covered

- Overview of Machine Learning 
- Supervised vs. Unsupervised Learning
- Linear Regression
- Cost Function 
- Cost Function Intuition
- Visualizing the Cost Function 

--- 

# Learning Objectives

After completing this week, I can:

- Explain what Machime Learning is. 
- Distinguish supervised lerarning from unsupervised learning.
- Understand how Linear Regression predicts continous values.
- Explain the purpose of the Cost Function.
- Interpret how the Cost Function evaluates model performance.

# 1. Overview of Machine Learning 

## What is Machine Learning?

Machine Learning is a field of Artificial Intelligence that enables computers to learn patterns from data instead of relying on explicitly programmed rules.

Rather than writing fixed instructions, a model learns relationships from examples and improves its prediction based on data.

### Typical Applications

| Industry | Example | 
|----------|----------|
| Finance | Fraud Detection | 
| Retail | Product Recommendation | 
| Marketing | Disease Prediction | 

----

# 2. Supervised vs. Unsupervised Learning 

## Supervised Learning

The model learn from labeled data.

- Input (X)
- Correct Answer (Y)

Goal: 

Predict the correct output for new data.

Examples

- House Price Prediction
- Email Spam Detection
- Credit Risk Prediction 

---

## Unsupervised Learning 

The model learns patterns from unlabeled data. 

Goal: 

Discover hidden structures within data. 

Examples

- Customer Segmentation
- Topic Modeling
- Market Basket Analysis 

## Comparison 

| Supervised Learning | Unsupervised Learning |
|---------------------|-----------------------|
| Uses labeled data | Uses unlabeled data | 
| Predicts outputs | Finds hidden patterns | 
| Regression | Clustering | 
| Classification | Dimensionality Reduction | 

---

# 3. Linear Regression 

## Purpose 

Linear Regression predicts continous numerical values.

Examples

- House Pirce
- Sales Revenue
- Temperature
- Stock Price (basic prediction)

---

## Model Representation 

Prediction Function

ŷ = wx + b

Where

| Symbol | Meaning | 
|--------|----------|
| x | Input Feature | 
| w | Weigh (Slope) | 
| b | Bias (Intercept) | 
| ŷ | Predicted Value | 

--- 

## Business Example 

Input 

- House Size

↓

Linear Regression Model

↓

Output

- Estimated House Price 

--- 

# 4. Cost Function

## Purpose 

The Cost Function measures how well the model fits the training data.

Lower Cost means better predictions.

Higher Cost means poorer predictions. 

---

## Formula 

\[
J(w,b)= \frac{1}{2m}\sum_{i=1}^{m}(f(x^{(i)})-y^{(i)})^2
\]

Where 

| Symbol | Meaning | 
|---------|---------|
| m | Number of training examples | 
| f(x) | Predicted value | 
| y | Actual value | 
| J | Cost | 

---

## Cost Function Intuition

Prediction

↓

Prediction Error 

↓ 

Squared Error 

↓

Average Error 

↓

Cost

Goal 

Minimize the Cost Function to find the best model parameters.

---

## Visualizing the Cost Function

The Cost Function forms a bowl-shaped curve. 

The lowest point represents the Optimal values of **w** and **b**.

This minimum point corresponds to the model with the smallest prediction error. 

---

# Optional Labs 

## Objectives 

- Understand model representation
- Visualize Linear Regression
- Observe the effect of changing parameters
- Understand how the Cost Function changes

---

## Key Python Libraries

- NumPy
- Mataplotlib

--- 

## Important Functions

```python
import numpy as np
import matplotlib.pyplot as plt

plt.scatter()

plt.plot()
```

