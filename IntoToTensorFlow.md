# Module 1 Intro

**Tensorflow is an open-source high-performance library for numerical computation that uses directed graphs**

### DAGs

A DAG is a language independent representation of the code in your model.

DAGs give you language and hardware portability

Tensorflow supports federated learning: (e.g. on mobile) aggregate many users' updates -> forms consensus change -> update weights on cloud..

### Tensorflow API hierarchy

tf.estimator								High-level API for distributed training

tf.layers, tf.losses, tf.metrics 	Components useful when building custom NN models

Core Tensorflow (Python) 		Python API gives you full control

Core Tensorflow (C++), 			C++ API is quite low level

CPU , GPU, TPU, Android , 	TF runs on different hardware



### Lazy Evaluation

tf operations like tf.add dont evaluate right away. They build a DAG.
They do not get evaluated until you run the session. With session.run.

Eager mode will be used in development sometimes, but usually we use Lazy evaluation

=> Build Stage + Run Stage

### Graph and Session

A Tensorflow DAgG consists of Tensors (edges) and operations on those tensors (Nodes)

Very easy to work with since it is visually appealing and simple to modify. It is also fast.

DAG can be remotely assigned and executed (from mobile devices for example)

***Common arithmetics are overloaded*** (like tf.add(a,b) ==  a+b)

For eager execution call 
from tensorflow.contrib.eager.python import tfe 
tfe. enable_eager_execution() exactly once.

#### Writing your session graph into a file

```python
with tf.Session() as sess:
	with tf.summary.FileWriter('summaries', sess.graph) as writer:
		a1, a2 = sess.run([z1, z3]) # z1,z3 = Tensors
```

#### Tensorboard

```python
from google.datalab.ml import Tensorboard
TensorBoard().start('./summaries')
```



You can also write to gs:// and start TensorBoard from CloudShell

### Tensors

**Tensor:** n-Dimensional Array of Data

**Stacking:** 

```python
x1 = tf.constant([2,3,4]) 	# Shape (3,) 			1D
x2 = tf.stack([x1,x1])		# Shape (2,3)			2D
x3 = tf.stack([x2,x2,x2,x2]) # Shape (4,2,3) etc.	 3D
```

**slicing:**

```python
x = tf.constant([[3,5,7],[4,6,8]])

y = x[:, 1] # all rows, Column number 1 (pyton zeroindexing => second column)
y2 = x[1, :]# Row 1 and all columns
y3 = x[1, 0:2]# Row 1 and columns 0 to 2 excluding 2 (so 0 and 1)
```

**reshaping:**

```python
x = tf.constant([[3,5,7], [4,6,8]])
y = tf.reshape(x, [3,2])

=> [[3,6], [7,4], [6,8]]

y2 = tf.reshape(x, [3,2]) [1, :]
=> [7,4]
```



### Variables

```python
def forward_pass(w,x):
    return tf.matmul(w,x)

def train_loop(x, niter = 5):
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        w = tf.get_varuable("weights",
                           shape = (1,2), #1x2 matrix
                           initializer = tf.truncated_normal_initializer(),
                           trainable = True)
    preds = []
    for k in xrange(niter):
        preds.append(forwardpass(w,x))
        w = w + 0.1 # "gradient update"
    return preds


with tf.Sesion() as sess:
    preds = train_loop(tf.constant([[3.2, 5.1, 7.2], [4.3, 6.2, 8.3]])) # [1,2]x[2,3] => [1,3]
    tf.global_variables_initializer().run()
    for i in xrange(len(preds)):
        print "{}:{}".format(i,preds[i].eval() )
```

**Placeholders** allow you to feed in values such as by reading from a text file 

```
a = tf.placeholder("float", None)
b = a * 4
with tf.Session() as session:
	print(session.run(b, feed_dict={a: [1,2,3]}))
```



## Debugging Tensorflow Programs

1. Read Error
2. Isolate method in question
3. Send made-up data into method
4. know how to solve common problems

- The most common problem tends to be tensor **shape**
- Shape problems also happen because of batch size or because you have a scalar when a vector is needed (or vice versa)

**Shape problems** can often be fixed using:

1. tf.reshape()
2. tf.expand_dims(input, pos)  (shape as an additional dimension of size 1 added at position pos)
3. tf.slice()
4. tf.squeeze() (same type as input, but has one or more dimensions of size 1 removed) (inverse of expand_dims)



Another common problem is **data type**

For example int arrays cant be added to float arrays. Solution: be more clear. decide if you want to round floats to ints or cast ints to float.

3 methods to **debug a full-blown program**:

1. tf.print()
2. tfdbg
3. TensorBoard (high-level debugging)

*Change log-level to INFO with tf.logging.set_verbosity(tf.logging.INFO)* (default is WARN which is kinda quiet)

*tf.Print() can be used to log specific tensor values*

*tfdbg is called on terminal with "python xyz.py --debug" and is an interactive debugger with steps and such*

# Module 2 Estimator API

- Create production-ready machine learning models the easy way
- Train on large datasets that do not fit in memory
- Monitor your training metrics in TensorBoard

## Estimator API

Advantages: 

- Quick model
- checkpointing
- out-of-memory datasets
- Train/eval/monitor
- Distributed training
- Hyper-parameter tuning on ML-Engine
- Production: serving predictions from a trained model

tf.estimator.Estimator. 

- LinearRegressor
- DNNRegressor
- DNNLinearCombinedRegressor
- LinearClassifier
- DNNClassifier
- DNNLinearCombinedClassifier

**Pre-made Estimators:**

1. Features need to be chosen

```python
featcols = [
	tf.feature_column.numeric_column("sq_footage"),
	tf.feature_column.categorical_column_with_vocabulary_list("type", ["house", "apt"])
]

model = tf.estimator.LinearRegressor(featcols) 
# "Model": predicts PRICE
```



*Under the hood*: **feature columns** take care of packing inputs to the input vector of the model

```python
def train_input_fn():
    features = {"sq_footage": 	 [1000, 2000 ,3000],
               "type": 			 ["house", "house", "apt"]} # ...
   	labels = 					[500, 1000, 1500] # the examples we learn with, so correct prices
    return features, labels

model.train(train_input_fn, steps=100)

def predict_input_fn():
    features = {"sq_footage": [1500, 1800],
               "type": 		["house", "apt"]}
    return features
predictions = model.predict(predict_input_fn)

print(next(predictions))
	
```



1. Feature columns to make the data shaped something the model can understand
2. use estimator for model
3. then train
4. then predict

**Checkpoints:** 
Allow to 

1. Continue Training
2. Resume on failure
3. Predict from trained model

*Checkpoints are default for estimators:* Just add the folder name when calling the estimator

In memory data from numpy or Pandas can be used directly:

```python
def numpy_train_input_fn(sqft, prop_type, price):
    return tf.estimator.inputs.numpy_input_fn(
        x = {"sq_footage": sqft, "type": prop_type},
        y = price,
        batch_size = 128,
        num_epochs = 10,
        shuffle = True,
        queue_capacity = 1000
    )
# OR
def pandas_train_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
    	x = df,
    	y = df['price'],
    	batch_size = 128,
        num_epochs = 10,
        shuffle = True,
        queue_capacity = 1000
    )
```

### Training on large datasets with Dataset API

Real World ML Models

Problem: Out of memory data, Solution: Use the Dataset API

tf.data.Dataset

- .TextLineDataset
- .TFRecordDataset
- .FixedLengthRecordDataset

```python
def decode_line(txt_line):
    cols = tf.decode_csv(txt_line, record_defaults=[[0], ['house'],[0]])
    features = {'sq_footage':cols[0], 'type': cols[1]}
    label = cols[2] # price
    return features,label

# dataset = tf.data.TextLineDataset("train_1.csv").map(decode_line) # Map to transform each line into data items with tf.decode_csv in each line

dataset = tf.data.Dataset.list_files("train.csv-*")\
				.flat_map(tf.data.TextLineDataset)\
    			.map(decode_line)

dataset = dataset.shuffle(1000).repeat(15).batch(128)

def input_fn():
    features, label = dataset.make_one_shot_iterator().get_next()
    return features, label

model.train(input_fn)
```

*input functions are executed once and create a pair of nodes that get connected to the graph and provide the data to the DAG*

An input function is not called every time the model needs data. They return nodes, not data. (**!!!**)

### Distributed Training

Problem: Distribution, Solution: Use train_and_evaluate

**data parallelism**: replicate your model on multiple workers

you gotta do 4 things:

1. tf.estimator.RunConfig(model_dir=output_dir, ...)
2. tf.estimator.Estimator(featcols, config=run_config)
3. tf.estimator.TrainSpec(input_fn, max_steps)
4. tf.estimator.LatestExporter(serving_input_receiver_fn=serving_input_fn)
5. tf.estimator.EvalSpec(input_fn, exporters)
6. tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

**shuffling is even more important in distributed training!** -> shuffle for each worker even when data comes shuffled on disk!

**Problem**: Need to evaluate during training, **Solution**: Use train_and_evaluate + TensorBoard

**Problem**: Deployments that scale, **Solution**: Use serving input function

Inputs at serving time often different than at training time.

on google cloud we got to read json data.

```
gcloud ml-engine predict --model <model_name> --json-instances data.json

OR

gcloud ml-engine local predict --model-dir outdir\modelname --json-instances test_jsons.txt 
```

```python
def serving_input_fn():
    json = {'jpeg-bytes': tf.placeholder(tf.string, [None])}
    
    def decode(jpeg):
        pixels = tf.image.decode_jpeg(jpeg, channels=3)
        return pixels
    
    pics = tf.map_fn(decode, json['jpeg_bytes'], dtype=tf.uint8)
    
    features = {'pics': pics}
    return tf.estimator.export.ServingInputReceiver(features, json)
```





## Models on GCP

- Train model on GCP
- Monitor model training
- Deploy a trained model as a microservice



scaling out not up is often better

Usually files split in two: task.py and model.py

Scale Tier Options:

- BASIC (single cpu)
- STANDARD (small cluster)
- BASIC_GPU (single gpu)
- BASIC_TPU (single tpu)

```python
gcloud ml-engine jobs describe job_name #details of current state

gcloud ml-engine jobs stream-jobs job_name # latest logs

gcloud ml-engine jobs list --filter='createTime>2017-01-16T10:00'
gcloud ml-engine jobs list --filter='jobID:census*' --limit=3
```







