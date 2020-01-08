# Part 1: ML Methods

Goals:

- Foundational Machine Learning Knowledge
- Practical tips and Pitfalls
- code for own ml models
- how did history of ml lead here

Learn how to

- Identify why deep learning is popular
- optimize and evaluate models using loss functions and performance metrics
- Mitigate common problems that arise in machine learning
- create repeatable training, evaluation, and test datasets 

- Differentiate between the major categories of ML problems
- Place major ML methods in the contest of their historical development
- identify why deep learning is popular

## Supervised Learning

Supervised models => labels (training includes correct answers)
unsupervised models => no labels (correct answers unknown)

Regression and classification are supervised ML model types

**Regression Model** when the label is continuous

**Classification Mode**l when the label has a discrete number of options

=> prediction Problems

**Clustering Model** when data isn't labeled (unsupervised)

=> Description Problems

## History of ML

**Linear regression:** was invented for growth of peas

$ y = w_0x_0 + w_1x_1 + w_2x_2 + ...$

=> $y = X*w$ (X := Matrix)

$ L = ||y - x*w ||^2$ // Loss for linear regression 

$ w = (X^T *X)^{-1} * X^T *y$ = weight vector = Gram Matrix * Moment Matrix (closed form solution, but not practical for big datasets) 

Laufzeit: $O(n^2)$

**=> Gradient Descent** Finding Gradient fo the loss function to find the lowest valley

Learning rate is a hyperparameter that helps determine gradient descent's step size along the hypersurface to hopefully speed up convergance

**Perceptron:** (1940s) Precursor to neural Networks!

It is a *Binary Linear Classifier*. Inputs feed into a single layer of perceptrons are weighted and the sum will be performed. The sum passes through an activation function, which is a mathematical function that is applied to each element.

weighted Sum of Inputs have to reach a certain Value-threshold in the Unit Step Function to activate into an output 

Perceptron is analogous to Dendrites of a real Neuron

Problem: cant learn simple functions like XOR

**Neural Networks:** Nonlinear activation functions can be used to add more layers of Perceptrons. If Linear then the layers can be compressed into one layer => nonlinear is necessary.

Problem: sometimes gets hung up in local minima.

Common Activation Functions:

- Linear
- Sigmoid = $\sigma (x) = \frac{1}{1+e^{-x}}$
- Tanh = $tanh(x) = \frac{2}{1+e^{-2x}} -1$ 
- ReLU = $f(x) = {0\ for\ x <0}\  and\  {x\ for\ x \geq 0}$ (in DNNs creates a hyperplanar decision surface, which can be very powerful)
- ELU = $f(x) = \alpha (e^x-1)\ for\ x<0\ and\ x\ for\ x \geq 0$

**Decision Trees:**  creates a linear decision surface, which is what ReLu layers give you as well.

Process: Look for a feature, then split data into subsets

Combining Decision trees into forests is helpful/powerful

**Kernel Method:** Support Vector Machines (**SVMs**) 

Core: nonlinear activation plus a sigmoid output. 
Logistic regression as main thing. 
*In SVMs* two parallel hyperplanes on either side of the decision boundary hyperplane where they intersect with the closest data point on each side of the hyperplane (**support vectors**).

Maximising the margin is the goal, using a hinge loss function

Linear Kernel, Polynomial kernel, Gaussian radial basis function kernel

**Random Forests:**

Simulate many weaker learners by turning of random Neurons, then combine the weak learners and take a majority vote for classification or get a mean, max median or something like that for a regression problem. Aggregation reduces both bias and variance. Result is usually a similar bias, but a lower variance

Random subspaces are made when we sample from the features, and if we random sample examples too is called random patches. Adaptive boosting or AdaBoost in gradient boosting are both examples of boosting, 

Validation set for early stopping, so we dont overfit the trainingdata. 

Stacking is also possible. meta-learners learn and then can be turned into meta-meta-learners and so on.

**Modern Neural Networks:**

NNs with hundreds of layers, millions of parameters. Combining some of the other ML methods.

# Part 2: Optimization

Learn how to:

- Quantify model performance using loss functions
- Use loss functions as the basis for gradient descent
- Optimize gradient descent to be as efficient as possible
- Use performance metrics to make business decisions

5 topics:

- Defining ML models
- Introducing Loss Functions
- Gradient Descent
- TensorFlow Playground
- Performance Metrics

**ML models:** are mathematical functions with parameters and hyper-parameters

**Parameter:** changed during model training

**Hyper-parameter:** Set before training

*Linear models* : Output = Bias + Input * weight. 
Bias and weight are Model Parameters. Input as Matrix is common

**Decision boundary:** boundary in between two or more classes.



### Natalaty Dataset

Problem statement: Not all babies get the care they need.
possible feature in the model: Mother's Age. 
possible label: Mother's Age. Baby Weight?

When we try to model a non-linear problem with a linear model we call that underfitting

analytical methods for finding the best set of model parameters don't scale

### Loss Functions

- Use Loss Functions

**Error** = 

1. actual(true) - predicted value
2. Compute the squares of the error values
3. Compute the mean of the squared error values
4. Take a square root of the mean

=> $\sqrt{\frac{1}{n} * \sum_{i=1}^n (\hat{Y_i} - Y_i)^2}$ = **RMSE** (Root of the mean squared error) // for regression

RMSE doesn't work well for classification

**Cross Entropy Loss**

$\frac{-1}{N} * \sum_1^N y_i * log(\hat{y_i}) + (1 - y_i) * log(1- \hat{y_i})$

First Term is positive (when label is 1), Negative term when label is 0 (label = yi) 

### Gradient Descent

- Take a loss function and turn it into a search strategy

"Walking down the surface formed by using our loss function on all the points in parameter space"

Simple Algorithm:

```
while loss > Epsilon:
	derivative = computeDerivative()
	for i in range(self.params):
		self.params[i] = self.params[i] -learning_rate * derivative[i]
	loss = computeLoss()
```

Loss function Slope and Magnitude can tell you the StepSize and Direction!

**Learningrate = Hyperparameter**

"This doesn't mean that our algorithm doesn't work, it simply means we tend not to encounter the sorts of problems where it excels"

To increase training time: batch size down (number of data points to compute the derivative)

Batch Gradient Descent is not Mini Batch Gradient Descent.

Mini Batch is called Batch Size (typically between 10-10 000)

To increase the training time: not checking the loss as frequently 

### Performance Metrics

**Inappropriate Minimum**: doesn't reflect the relationship between features and label, wont generalize well

Loss Functions:

- During training
- Harder to understand
- Indirectly connected to business goals

Performance Metrics:

- After training 
- Easier to understand
- Directly connected to business goals

Example for Performance Metrics:

Confusion Matrix:

​	**Accuracy** = (TP + TN) / total examples given
​	**Precision** = true positives / total classified as positive
​	**Recall** = true positives / all actual positives in our reference

Summary: Models are sets of parameters and hyper-parameters -> FInd the best parameters by optimizing loss functions through gradient descent -> Experiment with Neural Networks in TF Playground

# Part 3: Generalization and sampling

When is the most accurate ML model not the right one to pick?

Learn how to: 

- Asses if your model is overfitting
- Gauge when to stop model training
- Create repeatable training, evaluation and test datasets
- Establish performance benchmarks

Loss metrics for Regression:
MSE or RMSE

Checking the model against unseen data in order to check for overfitting

**Split Dataset into Training and Validation Datasets And Test Dataset!** 

Tuning Hyperparameters for less overfitting. 

Choose the model that has a lower loss on the validation set, NOT the training set

Split multiple times to not "waste" data (**Bootstrapping, Cross-validation**)

If you have *lots of data* use a held-out test dataset

if you have little data, use cross-validation

Carefully choose which field will split your data. Split your data on  a field you can afford to lose.

*Develop your TF code on a small subset of data, then scale it out to the cloud* 



```sql
#standardSQL
SELECT
	date,
	ABS(FARM_FINGERPRINT(date)) AS date_hash,
	MOD(ABS(FARM_FINGERPRINT(date)),70) AS remainder_divide_by_70,
	MOD(ABS(FARM_FINGERPRINT(date)), 700) AS remainder_divide_by_700,
	airline,
	departure_airport,
	departure_schedule,
	arrival_airport,
	arrival_delay
FROM
	`bigquery-samples.airline_ontime_data.flights` 
	
WHERE
	#Pick 1 in 70 rows where hash / 70 leaves no remainder
	MOD(ABS(FARM_FINGERPRINT(date)), 70) = 0
	
	AND
	MOD(ABS(FARM_FINGERPRINT(date)), 700) >= 350
	
	AND
	MOD(ABS(FARM_FINGERPRINT(date)), 700) < 525
```

3 Steps:

**EXPLORE THE DATA, **

**CREATE DATASETS, **

**CREATE A BENCHMARK**

Rule of Thumb: **All Data is Dirty!**

useful thing for dataframes is .describe() to look at some facts about the dataframe

















