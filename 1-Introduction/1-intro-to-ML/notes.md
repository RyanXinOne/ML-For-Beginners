# Concepts

+ ***Artificial intelligence (AI)***: About consuming the machine learning, build applications by applying machine learning models to solve practical problems.  
It studies algorithms and automated systems developed for problem solving, focuses on how to solve problems instead of the problems themselves.

+ ***Machine learning (ML)***: Algorithms that use data to find inner patterns and learn something useful  
subset of A.I algorithms that are divided into two stages, learning and execution. On learning the algorithm generates a data model that tries to abstract a characteristic of the data On execution the data model is used to reproduce the learnt characteristic on new data samples.

+ ***Deep learning***: subset of M.L. algorithms that uses connectionist approaches (neural networks), which are heavily based on linear algebra transformations and gradient descent, they often minimize by small steps a function that evaluates how well the "data model" fits the target characteristic, for each data sample. It's called Deep because they often perform more than one transformation on the input data, more transformations imply a "deeper" model.

# Video: MIT's John Guttag introduces machine learning
## Feature Engineering
Features never fully describe the situation
> "All models are wrong, but some are useful." - George Box

## Distance Measurement
**Minkowski Metric**
$$ dist(X1, X2, p) = (\sum_{k=1}^{len} abs({X1}_k - {X2}_k)^p)^{1/p}\\p = 1: Manhattan\ Distance\\p = 2: Euclidean\ Distance$$

## Statistical Measures
Define:
+ TP = true positive
+ TN = true negative
+ FP = false positive
+ FN = false negative
$$ accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$
$$ positive predictive value = \frac{TP}{TP + FP} $$
---
*Percentage correctly found*
$$ sensitivity = \frac{TP}{TP + FN} $$
*Percentage correctly rejected*
$$ specificity = \frac{TN}{TN + FP} $$
