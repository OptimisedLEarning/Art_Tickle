---
title: "Function Transformer"
datePublished: Wed Apr 03 2024 03:58:35 GMT+0000 (Coordinated Universal Time)
cuid: cluja34uo000b08jz6qinaz6g
slug: function-transformer
tags: data-science, machine-learning, concepts

---

feature encoding technique basically a mathematical transformer (Applied on numerical data )

To improve the model performance as these transformations conform to model assumptions which in turn amplifies the model's predictive power and increases the quality of the model

1. It can even out the variance
    
2. To make the feature more normal
    
3. It can reduce the skew
    
4. It can linearize the relationship between the feature and the target
    
5. Reduce the impact of outliers
    

How to check for Normality

1\. PDF

2\. Skew

3\. QQ Plot

4\. Statistical test

## What are Feature Transformations?

Feature transformation involves applying mathematical operations to modify the original data features in a way that enhances their representation for machine learning models. These transformations can help in improving model accuracy, meeting algorithm assumptions.

1\. Power Transformations

2\. Log Transformations

3\. Quintile Transformations

Problems after transformation

1\. Interpretation

2\. Finding the best transformation is tricky

3\. Additional step in the pipeline

**There are 3 types of Feature transformation techniques:**

1. Log Transformers
    
2. Power Transformers
    
3. Quantile Transformers
    

Let us learn and earn intuition of widely used transformation.

### Log Transform

The log is applied to <mark>every single distribution </mark> of the data (i.e. columns )and the result from the log is considered the final day to feed the machine learning algorithms.

Algorithms that benefit from Log Transform:

*<mark>1. Linear Models</mark>*

*<mark>2. ANOVA</mark>*

*<mark>3. Time series analysis</mark>*

*<mark>4. K-Means</mark>*

*<mark>5. PCA</mark>*

*<mark>6. Gaussian Naïve Bayes</mark>*

*<mark>7. Training of Neural Networks</mark>*

**When to use?**

1\. When you have <mark>right-skewed data</mark> (Log transformer takes the log of the values in a column. For example if an income column ranges from 1800 to 1,20,000,the log values will range from about 7.5 to 11.7.)Through experiments, it is proven that **log transforms** perform so well on the <mark>right-skewed data</mark>.

2\. When your data contains outliers

3\. Reduces <mark>Heteroskedasticity</mark> (in data <mark>isn't the same across all levels</mark> of another variable. It means that <mark>as one thing changes</mark>, the <mark>spread </mark> of data <mark>around a lin</mark>e also <mark>changes. </mark> This can <mark>mess up </mark> calculations in *regression analysis* because it assumes the spread stays the same. ) That is why we have to <mark>make sure </mark> our data turns out to be a little bit <mark>close to normal distribution</mark>

**When not to use?**

1\. With negative values/zero values, values ranging from 0 to 1 (because we don't want our scale to reduce more than that )

2\. With a normal or uniform distribution

3\. Interpretation Inverse Transform

### Square Transform

In this transformation, the data is applied with the **square function**, where the square of every single observation will be considered as the final transformed data.

### Square Root Transform

In this transform, the **square root** of the data is calculated. This transform performs so <mark>well</mark> on the <mark>left-skewed data</mark> and <mark>efficientl</mark>y transforms the left-skewed data into normally distributed data.

### Reciprocal Transform

In this transformation, the reciprocal of every observation is considered. This transform is useful in some of the datasets as the reciprocal of the observations works well to achieve normal distributions.

### Custom Transforms

In every dataset, the log and square root transforms can not be used, as every data can have different patterns and complexity. Based on the domain knowledge of the data, custom transformations can be applied to transform the data into a normal distribution. The custom transforms here can be any function or parameter like sin, cos, tan, cube, etc.

## Power Transformers

Power Transformation techniques are the type of feature transformation technique where the power is applied to the data observations for transforming the data.

There are two types of Power Transformation techniques:

1. **Box-Cox Transform**
    
2. **Yeo-Johnson Transform**
    

### Box-Cox Transform

The Box-Cox transformation is a family of <mark>power transformations</mark> that are applied to data to <mark>stabilize variance, make the data more normally</mark> distributed, or <mark>improve</mark> the <mark>s</mark>*<mark>kewness </mark>* of both the <mark>right and left-skewed data.</mark> It's particularly useful when dealing with non-normal data that violates the assumptions of many statistical tests and models.

This transform technique is mainly used for transforming the data observations by applying power to them. The power of the data observations is denoted by Lambda(λ). There are mainly two conditions associated with the power in this transform, which is lambda equals zero and not equal to zero. The mathematical formulation of this transform is as follows:

![X_{i}^{ambda}= eftegin{matrix} n{X_i} & ;athrm{for }  ambda = 0   rac{X_{i}^{ambda}-1}{ambda} &  ;athrm{for }  ambdaeq 0  nd{matrix}ight.](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-c10fbcdf5d9f8ab2176cbb23a687e550_l3.svg align="center")

Here the lambda is the power applied to every data observation. (so we try to calculate this power) Based upon the iteration technique every single value of the lambda is examined and the <mark>best-fit valu</mark>e of the lambda is then applied to the data to <mark>transform it</mark>. There are two techniques to calculate it

<mark>1. Maximum likelihood </mark> (in logistic regression) <mark>2.Bayesian statistics( part of inferential statistics)</mark>

The transformed value (**Lambda(λ)**)of every data observation will lie between <mark>5 </mark> to <mark>-5. </mark> One major <mark>disadvantage associated</mark> with this tran<mark>sformation tech</mark>nique is that this technique <mark>can only </mark> be <mark>applied</mark> to <mark>positive observation</mark>s. it is <mark>not </mark> applicable for <mark>negative and zero</mark> values of the data observations.

### Yeo Johnson Transform

This is an advanced form of a box Cox transformation technique where it can be applied to <mark> even zero and negative </mark> values of data observations.

The mathematical formulations of this transformation technique are as follows:

![X_{i}=  eftegin{matrix} rac{eft ( y+1 ight )^ambda-1}{ambda} & ;athrm{for}yeq 0athrm{and}ambdaeq 0  ogeft ( y+1 ight ) & ;athrm{for}yeq 0athrm{and}ambda = 0  rac{eft (1-y ight )^{2-ambda}-1}{2-ambda} & ;athrm{for}y<0athrm{and}ambda eq 2   -ogeft ( 1-y ight ) & ;athrm{for}y< 0athrm{and}ambda = 2   nd{matrix}ight. ](https://www.geeksforgeeks.org/wp-content/ql-cache/quicklatex.com-8895dd00812fdf384b4ce2e645008e72_l3.svg align="left")

In this transformation technique, y represents the appropriate value of X<sub>i</sub>. In scikit learn the default parameter is set to Yeo Johnson in the Power Transformer class.

[Reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)

<mark>Note</mark> internally both (Box-Cox and Ye Johnson apply <mark>Standard scaler</mark>)

## Key Takeaways

* The featured transformation techniques are used to transform the data to normal distribution for better performance of the algorithm.
    
* The Log transforms perform so well on the right-skewed data. Whereas the square root transformers perform so well on left-skewed data.
    
* Based on the domain knowledge of the problem statement and the data, the custom data transformation technique can be also applied efficiently.
    
* Box-Cox transformations can be applied to only positive data observations which return the transformed values between -5 to 5.
    
* Yeo Johnson’s transformation technique can be applied to zero and negative values as well.