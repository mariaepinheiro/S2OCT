# Mixed-Integer Linear Optimization for Semi-Supervised Optimal Classification Trees

This repository contains all the information needed to reproduce the results for the paper [Mixed-Integer Linear Optimization for Semi-Supervised Optimal Classification Trees](https://arxiv.org/abs/2401.09848) authored by Jan Pablo Burgard, Maria Eduarda Pinheiro, Martin Schmidt.

# Abstract
Decision trees are one of the most famous methods for solving
classification problems, mainly because of their good interpretability proper-
ties. Moreover, due to advances in recent years in mixed-integer optimization,
several models have been proposed to formulate the problem of computing
optimal classification trees. The goal is, given a set of labeled points, to split
the feature space with hyperplanes and assign a class to each partition. In
certain scenarios, however, labels are exclusively accessible for a subset of the
given points. Additionally, this subset may be non-representative, such as in
the case of self-selection in a survey. Semi-supervised decision trees tackle the
setting of labeled and unlabeled data and often contribute to enhancing the
reliability of the results. Furthermore, undisclosed sources may provide extra
information about the size of the classes. We propose a mixed-integer linear
optimization model for computing semi-supervised optimal classification trees
that cover the setting of labeled and unlabeled data points as well as the overall
number of points in each class for a binary classification. Our numerical results
show that our approach leads to a better accuracy and a better Matthews cor-
relation coefficient for biased samples compared to other optimal classification
trees, even if only few labeled points are available
