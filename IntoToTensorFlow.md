# Part 1

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





