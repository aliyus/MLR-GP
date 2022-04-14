# Multiple Linear Regression Hybridisation with Genetic Programming (MLR-GP)

The code implements a hybridisation of GP with Multiple Linear Regression (MLR-GP). 

Isolating the fitness-contribution of substructures is typically a difficult task in Genetic Programming (GP). Hence, useful substructures are lost when the overall structure (model) performs poorly. Furthermore, while crossover is heavily used in GP, it typically produces offspring models with significantly lower fitness than that of the parents. In symbolic regression, this degradation also occurs because the coefficients of an evolving model lose utility after crossover. 

The MLR-GP method keeps the features untangled and allows GP to designate different parts of the tree structures as distinct and transferable features. A focus of this paper is to investigate how this transferability improves the normally disruptive nature of crossover. However, note that subtree crossover can swap both complete features or subtrees within the features. 

The study this code implements isolates the fitness-contribution of various substructures and reducing the negative impact of crossover by evolving a set of features instead of monolithic models. The method then leverages multiple linear regression (MLR) to optimise the coefficients of these features. Since adding new features cannot degrade the accuracy of an MLR produced model, MLR-aided GP models can bloat. To penalise such additions, we use {\em Adjusted} $R^2$ as the fitness function.  Experimental results show that the proposed method matches the accuracy of the competing methods within only 1/10th of the number of generations. Also, the method significantly decreases the rate of post-crossover fitness degradation.




# Derived Pulications:

Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah. *“Feature Engineering for Enhanced Performance of Genetic Programming Models",*
In: GECCO '20 Companion, Genetic and Evolutionary Computation Conference Companion, July 2020. URL: https://doi.org/10.1145/3377929.3390078.

 
Aliyu Sani Sambo, R. Muhammad Atif Azad, Yevgeniya Kovalchuk, Vivek P. Indramohan, Hanifa Shah. *“Improving the Generalisation of Genetic Programming Models with Evaluation Time and Asynchronous Parallel Computing"*, In: GECCO '21 Companion, Genetic and Evolutionary Computation Conference Companion, July 2021. URL: https://doi.org/10.1145/3449726.3459583

