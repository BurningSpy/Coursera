# Features Engineering Part 1

### Raw Data to Features

- finding features
- creating features
- transforming features
- synthetic features
- hyperparameter tuning

Feature engineering takes about 50-70% of the time of creating a good ML model (**Data Preprocessing**)

What makes a good feature?

1. Be related to the objective
   - Dont just throw arbitrary data in there. Dont dredge
2. Be known at prediction-time (can be tricky **!**)
   - If the data is not known at prediction time, you cant use it for training, because it will be useless
   - feature definitions should not change over time!
   - ambiguous feature field values can cause issues
   - Some data might come from another system which we may not have knowledge or control over
3. Be numeric with meaningful magnitude
   - Word2Vec could be used to make a non-numeric variable numeric
   - you can give variables meaning using one-hot-encoding
4. have enough examples
   - at least ~5 examples so that the model doesnt learn wrong things based on too few examples 
   - plot histograms to find out if there are enough
5. bring own human insight to problem

*Different Problems in the same domain may need different features*

Options for encoding categorical data:

- if you know the keys beforehand: tf.feature_column.categorical_column_with_vocabulary_list(<name>, <list>)
- If your data is already indexed:  tf.feature_column.categorical_column_with_identiy(<name>, <num_buckets>)
- If you dont have a vocabulary of all possible values: tf.feature_column.categorical_column_with_hash_bucket



**Statistics**: "I've got all the data I'll ever get" -> throw away outliers

**ML**: lots of data, keep outliers and build models for them. learn the outliers and handle them

### Preprocessing and Feature Creation

In Order to rescale you need max and min. Else you need to preprocess the entire dataset 

Things you do in Preprocessing:

- remove examples that you dont want to train on
- Compute vocabularies for categorical columns
- Compute aggregate statistics for numeric columns
- Compute time-windowed statistics (e.g. number of products sold in previous hour) for use as input features (in Beam)
- Scaling, discretization,etc. of numeric features
- Splitting, lower-casing, etc. of textual features
- resizing of input images
- normalizing volume level of input audio

### Cloud Dataflow

written in apache beam api, then sent into dataflow

pipeline gets built first, then executed at the end only when called with the run method (kinda like the lazy evaualtion in tensorflow)

- Pipeline needs a **source** (input)
- Pipeline consists of **steps** which are transforms
- Output of the Pipeline is called **sink**
- **runner** is needed to run the pipeline

Data in the pipeline stored in **PCollections** which are like pointer collections to where the data is actually stored 

**Lab:** 
What files are being read? 
Looks like java files

What is the search term?
searchTerm == 'import'

Where does the output go?
into tmp/output as Text

What does the transform do?
The transform yields all lines that start with the search term import. And supposedly discards the rest.
Thus it searches for all the imports 

What does the second transform do?
It writes the data to another text file

Where does its input come from?
from the java files 

What does it do with this input?
It reads the INput, then transforms it with the grep function

What does it write to its output?
the lines with the searchTerm (all the Imports)

Where does the output go to? 
to the tmp/output folder

What does the third transform do?
it writes the data

Does the output seem logical?
Yeah, it is the expected import lines from all files

### pipeline scaling 

Data is **sharded**, then distributed across multiple nodes through a mapping. **MapReduce -> Result**. 
**ParDo** class (for Parrallel use) useful for filtering, extracting, converting, calculating

**Lab 2:**

What custom arguments are defined?
--output-prefix and --input (input directory)

What is the default output prefix?
'/tmp/output'

How is the variable output_prefix in main() set?
It's set on default

How are the pipeline arguments such as --runner set?
run is wait until finish. 

What are the key steps in the pipeline?

1. Reading the input from text
2. get all the inputs as transform
3. get all the package names
4. check how many packages are used
5. take the top 5 
6. write them out into a file

Which of these steps happen in parallel?
Get Imports and PackageUse happen in parallel 

Which of these steps are aggregations?
totalUse and Top5 

### Cloud Dataprep

Upload from anywhere, then explore data, then flows (like pipeline) then output

Lab:

True or False, the majority of the cab rides for 2015 were less than 2 miles?
True

Examine the **pickup_time** histogram. Which hours had the fewest amount of pickups? The most?
fewest is 5-6, most is 19-20

Explore the **average_fare_amount** histogram. Is there a range of fares that are most common?
12-13 is the most common

### Feature Crosses

Combine Features to create a linear model rule.

Discretize the entire input space, then you got weights for each cell. BUT you gotta memorize which is the opposite of generalization and not the goal BUT feature crosses work well on large datasets

- Feature crosses # massive data is an efficient way for learning highly complex spaces
- Feature crosses allow a linear model to memorize large datasets
- optimizing linear models is a convex problem
- Before Tensorflow, Google used massive scale learners
- Feature crosses as preprocessor make neural networks converge a lot quicker

Too much of it can be bad -> L1 Regularisation canceles out weights of a feature (removes them)

```python
tf.feature_column.crossed_column(
	[dayofweek, hourofday], 
    24*7)
```

Choosing the number of hash buckets is an art, not a science

The number of hash-buckets controls sparsity and collisions

Rule of thumb: "In practice, I tend to choose a number between half square root n and twice n" 

embedding feature crosses 

```
fc.embedding_column
```

3 possible **places to do feature engineering**

1. Do it on the fly as you read in data (in tf)
2. Seperate step before the training (in dataflow)
3. dataflow+tf (tf.transform)

1. in tf call the add_engineered method from all input functions (training, eval, serving), in serving use a serving input receiver
2. in dataflow: addFields  in training and prediction pipeline

Lab feateng.inbpy

**Analyze in Beam, Transform in Tensorflow! -> hypbrid tf.transform**

