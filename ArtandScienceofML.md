# Art and Science of Machine Learning

## Module 1 Art of ML

- Generalize your model
- Tune batch size and learning rate for better model performance
- Optimize your model
- Apply the concepts in TensorFlow Code

### Regularization

**Definition:** Regularization refers to any technique that helps generalize a model. A generalized model performs well not just on training data but also on never before seen test data

Simple models are usually better. **Occam's razor principle**: When presented with competing hypothetical answers to a problem, one should select the one that makes the fewest assumptions.

Minimize: **loss(Data|model) + complexity(Model)** 

=> **L1 & L2 regularization**

represent model complexity as the magnitude of the weight vector (and try to keep it in check)

L2 Norm is euclidian distance: $c = \sqrt{a^2+b^2}$ (pythagoras) 

- results in circle shape in regularization

L1 Norm = |a|+|b|

- results in diamond shape in regularization
- results in a more sparse solution
- it is used for feature selection, causing some weights to be zero -> simplifying the model



**Learning rate** controls the size of the step in weight space. If to small training takes long if too large training will bounce

 **Batch size** controls the number of samples that gradient is calculated on. If too small, training will take bounce around, if too large it will take a very long time. 

=> gotta find dat balance!

### Optimization

- GradientDecent -- traditional
- Momentum -- Reduces learning rate when gradient values are small
- AdaGrad -- Give frequently occuring features low learning rates
- AdaDelta -- Improves AdaGrad by avoiding reducing LR to zero
- Adam -- AdaGrad with a bunch of fixes
- Ftrl -- "Follow the regularized leader" works well on wide models

How to control these??:

1. Control batch size via input_fn
2. Control learning arte via optimizer passed into model
3. Set up regularization in optimizer
4. Adjust number of steps based on batch_size, learning_rate
5. Set number of steps, not number of epochs because distributed training doesnt play nicely with epochs

(Number of steps equal number of epochs multiplied by number of examples divided by batchsize)

## Module 2 Hyperparameter Tuning

- Differentiate between parameters and hyperparameters
- Think beyond simpler grid search algorithms
- Take advantage of Cloud ML Engine for hyperparameter tuning

How to use GCP for hyperparameter tuning:

1. Make parameter a command-line argument (use parser.add_argument)
2. Make sure outputs dont clobber each other (use good output folderstructure)
3. Supply hyperparameters to training job 

```python
#num_steps calculation: 
num_steps = (len(traindf) / args['batch_size']) / args['learning_rate']
```

## Module 3 Pinch of Science

### Regularization for sparsity

L1 regularisation => some low predictive features' parameter weights in the model will have zero values and get canceled out of the equation, makin the model smaller and less complex **this can help against overfitting**! L2 penalizes large weights more

**Logistic Regression:**

transform linear regression by a sigmoid activation function 

## $\hat{y} = \frac{1}{1+e^{-(w^Tx+b)}}$ 

casting binary problems into probabilistic problems

lossfunction typically cross-entropy!
Less emphasis on errors where the output is relatively close to the label.
$LogLoss = \sum -ylog(\hat{y}) - (1-y) log(1-\hat{y})$ 

Use **ROC curve** (Receiver Operating Characteristic) to choose the decision threshold based on decision criteria
The **Area-Under-Curve (AUC)** provides an aggregate measure of performance across al possible classification thresholds

*Logistic Regression predictions should be unbiased!*

=> **important things to do in linear regression!**

- Add regularization!
- Chose a tuned threshold!
- Check for bias! (Average of prediction close to average of observations)

## Module 4 Neural Networks 

Neural nets combine features algorithmically as an alternative to feature crossing. 

Do not use hidden layers of same size after another! It doesnt do anything! Use Non-Linear Transformation Layer in between with an activation function! 

**Favorite non-linearity is Rectified Linear Unit (ReLU)** -> often ten times speed of training with sigmoid, but some layers might die off due to negative side of ReLU

- softplus ($ln(1+e^x)$) is discouraged
- LeakyReLU in negative area the negative area is not 0 but 0.01
- PReLU letting through alpha x, (alpha gets learned too!)
- ReLU6 = min(max(0,x), 6) positive maxed out at 6
- ELU = Smooth in negative cuz of $\alpha (e^x-1)$ in negative

Add **Neurons to increase hidden dimensions**, add **layers to increase function composition**, add **outputs** if i have multiple **labels per example**

How to train:

```python
model = tf.estimator.DNNRegressor(model_dir = outdir, # like before
                                  hidden_units = [64, 32, 16], # new
                                  feature_columns = INPUT_COLS,
                                  optimizer = adam,
                                  dropout = 0.1) # dropout is a form of regularization. Randomly drop 												out neurons


```

Three common failure modes for gradient descent:

1. Gradients can vanish
   - Each additional layer can successively reduce signal vs. noise
   - Using ReLu instead of sigmoid/tanh can help
2. Gradients can explode
   - Learning rates are imortant here
   - Batch normalization (useful knob) can help
3. ReLu layers can die
   - Monitor fraction of zero weights in Tensorboard
   - Lower your learning rates

Small feature values help! 

### Multi class NNs

1vAll or 1vRest approach turns multiple classes in a non binary problem into many binary problems

1vOne approach gives a model for each binary combination -> n Classes -> n*n-1 Models

**Softmax** is an additional constraint, that total of outputs = 1.0 

- ## $p (y = y|x) = \frac{exp(w_j^Tx+b_j)}{\sum_{k\in K} exp(w^T_kx+b_k)}$

Approximate versions of softmax:

- **Candidate Sampling** calculates for all the positive labels, but only for a random sample of negatives : tf.nn.sampled_softmax_loss
- Noise-contrastive approximates the denominator of softmax by modeling the distribution of outputs: tf.nn.nce_loss

If labels are both mutually exclusive and also the probabilities: tf.nn.softmax_cross_entropy_with_logits_v2

If labels are mutually exclusive, but probabilities aren't: tf.nn.sparse_softmax_cross_entropy_with_logits

If labels arent mutually exclusive: tf.nn.sigmoid_cross_entropy_with_logits 

## Module 5: Embeddings

- Manage sparse data
- Reduce dimensionality
- Increase model generalization
- Cluster observations
- Create reusable embeddings

Instead of one hot encoding we put everything through a dense layer that create an embedding

use tf.feature_column.embedding_column( categorical_column , can make checkpoints for jumpstart) Works with any categorical column, not just feature crosses. 

Example: transferlearning might work on seemingly different problems as long as they share the same latent factors

- First layer: feature cross
- Second layer: a mystery box labeled latent factor
- Third layer: the embedding 
- Fourth layer: one side: image of traffic
- Second side: image of people watching TV 

### Recommendations

for example 500k movies to a million users. For every user our task is to recommend five to 10 movies.

Approach 
organize movies by similarity (1D) based  Attributes like age 
=> Input = n dimensions -> reduced to d-dimensional point

fewer inputs is better => embedding layer. The embeddings will change over training itself.

**Sparse Tensors**: mapping each feature to an integer, then store only the ids the user has seen to keep the Tensor sparse. (imagine how many movies you have watched and rated out of the 500 thousand available on the platform)

- If you know the keys beforehand: tf.fc.categorical_column_with_vocabulary_list('feature', voabulary_list = [...] ),
- If your data is already indexed: tf.fc.categorical_column_with_identiry('feature', num_buckets = 5),
- If you dont have a vocabulary of all possible values: tf.fc.categorical_column_with_hash_bucket('employeeId', hash_bucket_size = 500 )

For example if the embedding layer has 3 dimensions you can imagine that every node in the layer before the embedding could be represented by a vector in 3-dimensional space

Logit-Layer forms the output Layer of a neural network

Example: 

- Raw bitmap of the hand drawn digit -> 3 dimensional Embedding; Other features
-  -> Logit Layer
- -> Softmax Loss
- -> Target Class Label

Embeddings cluster similar items in space 

higher dimensions => more accuracy, but also overfitting and slow trading

dimensions = ca. 4th root of possible values

## Module 6: Custom Estimators 

- Go beyond canned estimators
- Write a custom estimator
- Gain control over model functions
- Incorporate Keras models into Estimator

Model Function:  (from a)

1. Look at train_and_evaluate function: tf.estimators.Estimator(**model_fn** = myfunc, model_dir = output_dir) ( )
2. myfunc (features (dict of features) , targets (labels) , mode (train / eval / predict)):  Job of myfunc is to create and return an EstimatorSpec
3. EstimatorSpec:
   1. Mode is pass-through
   2. Any tensors you want to return
   3. Loss metric
   4. Training op (needs to be carried out only when mode is train)
   5. Eval ops (needs to be carried out only when mode is eval)
   6. Export outputs 