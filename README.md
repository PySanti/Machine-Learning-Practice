
# Machine learning practice


The idea of this repository is to **practice my incipient knowledge in the area of machine learning**.

I will take a real dataset from [UCI](https://archive.ics.uci.edu) and the goal is to build a **meta-model** able to perform a super high prediction rate, based on the **stacking** strategy in the context of **ensemble learning**. The following algorithms will be implemented (it will be a classification problem):

 * Logistic Regression
 * SVC (with its linear, rbf and polynomial kernel variants)
 * Random Forest.

In addition, several preprocessing strategies will be implemented such as: 

* Use of **Imputers** to handle missing values
* **OneHotEncoding** to handle categorical variables
* Correlation analysis
* Sampling bias analysis
* Use of Scalers for normalization or standardization of numerical variables (if applicable).


Santiago De Andrade, 12/02/2025



# Day 2 

## Basic preprocessing

In the process of preprocessing the data I faced a rather annoying problem that took me a while to understand: it turns out that when I tried to run the transformers needed to encode and remove Nans respectively, I received an error from the encoding transformer. After a while I realized that this was because the category classes contained Nan values that had to be fixed before the encoding process, so I created a transformer for it.

## Sampling Bias

Once the basic preprocessing process was finished, I made a study of the possible oversampling of the dataset. Indeed, there was a 70-30 distribution of the classes. Through the implementation of the SMOTE (Synthetic Minority Over-sampling Technique) transform, I solved this problem and the models started to generate considerably better results.


## Correlation

In addition to the above, I made a study of the feature-feature and feature-target correlations, which showed a very low influence of the features native-country, occupation, workclass, etc. on the target. In spite of the above, the most convenient is to leave these classes since the difference in training time is minimal and it generates better results to leave them.


# Training

After going through the preprocessing stage, I started to do some tests with classification algorithms, the first ones I tried were **Logistic Regression** and **Random Forest**.

After doing a few tests, I reached an accuracy of **90% f1 score for Random Forest** and **86% for Logistic Regression**.

In this process, I learned the concept of hyperparameter optimization through Grid Search and Randomized Search.  I made a program to search for the most optimal parameters for the logistic regression algorithm, and after 3 hours (!!!) I arrived at the following result.

{'C': 10, 'max_iter': 500, 'penalty': 'l1', 'solver': 'liblinear', 'tol': 0.0001}

For which the algorithm also generates 86% accuracy. Now the plan will be to create a program that will search for the most optimal hyperparameters for SVC and Random Forest and then save the models to disk through the joblib library.

# Day 3

After leaving the PC with the program running for about 20 hours, I realized that it would take about 300 hours more to find the most optimal model for SVC, so I only ended up finding the most optimal model for Random Forest and Logistic Regression. These are the final results.

```
Logistic Regression Study 

~~~~~~~ Train Set
Positive accuracy : 0.8724723060520087
Negative accuracy : 0.8742672866891158
~~~~~~~ Test Set
Positive accuracy : 0.8751088850174216
Negative accuracy : 0.8779007877368533


Random Forest Study 

~~~~~~~ Train Set
Positive accuracy : 0.9376570495772272
Negative accuracy : 0.940308729782414
~~~~~~~ Test Set
Positive accuracy : 0.907832825906449
Negative accuracy : 0.912786129589859
```

The conclusion is that the best algorithm for this specific problem is Random Forest, even though it is generating some overfitting.

It should be noted that in this specific problem no preprocessing solutions such as feature extraction or feature selection were applied. 

Blessings.

# Update

I will save the most optimal **Random Forest params** here as i could not upload the models on github.

{'bootstrap': False, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 40, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 10, 'min_weight_fraction_leaf': 0.0, 'monotonic_cst': None, 'n_estimators': 500, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

