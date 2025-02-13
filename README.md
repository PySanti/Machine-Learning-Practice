
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