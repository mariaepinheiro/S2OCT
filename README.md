# Mixed-Integer Linear Optimization for Semi-Supervised Optimal Classification Trees

This repository contains all the information needed to reproduce the results for the paper [Mixed-Integer Linear Optimization for Semi-Supervised Optimal Classification Trees](https://arxiv.org/abs/2401.09848) authored by Jan Pablo Burgard, Maria Eduarda Pinheiro, Martin Schmidt.

# Abstract
Decision trees are one of the most famous methods for solving classification problems, mainly because of their good interpretability properties. Moreover, due to advances in recent years in mixed-integer optimization, several models have been proposed to formulate the problem of computing optimal classification trees. The goal is, given a set of labeled points, to split the feature space  with hyperplanes and assign a class to each partition. In certain scenarios, however, labels are exclusively accessible for a subset of the given points. Additionally, this subset may be non-representative, such as in the case of self-selection in a survey. Semi-supervised decision trees tackle the setting of labeled and unlabeled data and often contribute to enhancing the reliability of the results. Furthermore, undisclosed sources may provide extra information about the size of the classes. We propose a mixed-integer linear optimization model for computing semi-supervised optimal classification trees that cover the setting of labeled and unlabeled data points as well as the overall number of points in each class for a binary classification. Our numerical results show that our approach leads to a better accuracy and a better Matthews correlation coefficient for biased samples compared to other optimal classification trees, even if only few labeled points are available

# Preliminary needed tools

- Gurobi: https://juliapackages.com/p/gurobi


# How to use the Semi-Supervised Optimal Classification Trees
S2OCT return the hyperplanes, the objective function and the classification of the unlabeled data
Semi-Supervised Optimal Classification Trees need the following entrance value:
S2OCT return the hyperplanes, the objective function and the classification of the unlabeled data

## arguments:
- Xl: Labeled points such that the first ma points belong to class \mathcal{A},
- Xu: Unlabeled points. all points belong to \mathbb[R}^p
- ma: number of labeled points that belong to class  \mathcal{A},
- τ: how many unlabeled points belong to class  \mathcal{A},
- D: deep of the tree: integer number between 2 and 5
- C: penalty parameter:  number between 0.5 and 2.
- M: Big M value: η*s*\sqrt{p}+1 where η is the maximum distance between two points in [Xl Xu]
- maxtime: time limit,
 - s: bound of ω,
- solver: By default we use solver=1, which means we are using Gurobi. For that, it is necessary a Gurobi license. If choose any different value, SCIP is used.
 
# How to use the Optimal Classification Trees
 [Optimal Classification Trees](https://link.springer.com/article/10.1007/s10994-017-5633-9) is proposed by Bertsimas and Dunn and only considers labeled data.
