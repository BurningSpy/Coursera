## Module 1

Why ML:

1. How does google do ML
2. Strategy of Machine Learning
3. Machine Learning with Tensorflow 	(serverless machine learning)
4. Improving ML Accuracy
5. Machine Learning at Scale
6. Specialized ML models (image 	classification, sequence models, Recommendation systems)

**over 4000 Tensorflow ML models** at Google between 2012-2016

To be **successful**, dont only think about creating models, but also serving ML predictions.


Recommender systems allow you to make **personalized algorithms**

Google processes **batch and streamed data** the same way.

**AI is a Discipline**  

it is about Theory and Methods

**Machine Learning is a Toolset**

like Newtons laws of mechanics


**In Machine Learning, Machines learn. They dont start out Intelligent, they become intelligent.**





 ## Module 2

**Stages of Machine Learning:**

**Stage 1: Train ML with examples (supervised Learning)**

**Stage 2: Inference (putting ML models into production)**

Example consists of Label and Input

-  	Label = Cat, Dog etc.
-  	Input = Pixels of a Picture

A ML model is a mathematical function.

-> make tiny adjustments to the function so that the Output is as close as possible to the label

=> ML training relies on labeled examples

-> After Training model works on unlabeled inputs

Data is the fuel for ML




Deep Nns = Neural Networks with lots of layers

Google ML in almost every product

**Many Models per Product ->** Break down programs into multiple smaller problems, then build models for those to solve the bigger problem  




Google Translate:

-  	Model for identify the sign
-  	Model for OCR the Characters
-  	Model for Identify Language
-  	Model for translating
-  	Model for superimpose
-  	Model for Font and Color



Gmail recommender: seq2seq model.


**Replacing heuristics**

Programming paradigms change -> teach computer and then it does what you want


ML is about Logic, not just about data

RankBrain: deep neural network for rank searching

ML replaces hardcoded heuristic rules

"ML is a way to replace rules" (heuristic programmed rules)


**Its all about data**

Machine Learning is about collecting the appropriate data and then finding the right balance of good learning and trusting the examples

**Framing an ML Problem**

Use case 1:  

Manufacturing

-  	Predictive maintenance or condition monitoring
-  	warranty reserve estimation
-  	propensity to buy
-  	demand forecasting
-  	process optimization  	
-  	telematics

1. what is being predicted?:

How many products need to be made? 

2. What data is needed? 

Customer numbers, purchase times, frequency of purchases, distance from manufacturer to customer or warehouse or retail store. -> What's the demand?, What's the capacity?, How does demand change over time?

3. What is the API for the problem during prediction? 

The store page and manufacturers system. Widget ID and the month

4. Who will use this service? How are they doing it today?

The manufacturers use the system to optimize their processes. Without ML they will have to guess and write heuristic rules that are based on standards or personal experience.

5. What data are we analyzing?

Mostly customer decisions, opinions and tendencies.

6. What data are we predicting?

Customer tendencies and actions.
product errors (warranty reserve estimation).

7. What data are we reacting to?

Customer actions (buying/returning a product)

### Pre trained models

greatly decrease difficulty in time consuming tasks like classifying pictures.

Pre trained models are excellent ways to replace user input by machine learning

speech transcripts for chat interfaces 

We will focus on custom machine learning models

### Training and serving skew

Why go through manual data analysis?: 

1. if you're doing manual data analysis, you probably have the data already
2. if you dont have the data yet: if you cannot analyze your data to ge reasonable inputs towards making decisions, then theres no point in ML (it helps you fail fast, better than fail slow)
3. to build a good model you need to know your data. 
4. ML is a journey towards automation and scale. you are automating manual analysis because you want it to scale

*If you cant do analytics, you cant do ML*

**Training serving-skew**: unless the model sees the exact same data in serving as it was used to seeing during training the results will be off.

**Solution**: data pipelines have to process both batch and stream.

During training: key aspect is scaling to a lot of data
During prediction: key aspect is speed of responce, high QPS

### Transofrm your business

- Infuse your apps with ML (simplify user input, adapt to user)
- fine tune your business (streamline your business processes, create new businesses)
- Delight users (anticipate user needs)



*Supervised learning models are trained on labeled data.* (Two stages of ML video)

*A Better model requires a lot of data*, which the team does not have, a  lot of data can lead to insights in creating the model. Also, deploying a weak model may result in a  poor user experience, leading to loss of users.

## Module 3

### ML Surprise

- Defining KPI's (Key Performance Indicator) (5%)
- Collecting data (35%)
- Building infrastructure (35%)
- Optimizing ml algorithm (5%) (most people spend too much effort on this)
- Integration (20%)

### Secret sauce

Top 10 pitfalls: 

1. ML requires just as much software infrastructure
2. No data collected yet
3. Assume the data is ready for use (usually harder than expected)
4. keep humans in the loop 
5. product launch focused on ml algorithm (ML not always most important thing of the product)
6. ml optimizing for the wrong thing
7. is your ml improving things in the real world 
8. Using a pre-trained ml algorithm vs building your own
9. ML algorithms are trained more than once
10. trying to design your own perception of NLP algorithm

### ML and Business Processes

Improving Processes through feedback loops:

Inputs -> Process -> Outputs -> Insight Generation --- (operational parameters) ---> Tuning --> back to process with updated instructions.

How change happens in phases: (Path to ML)

1. Individual contributor (Process)  (task not parallized or scaled)
2. delegation (Process) (z.B. Store checker -> same actions across multiple individuals)
3. digitization (Process) (z.B. ATMs)
4. Big data and Analytics (Insight Generation) (Analyze every node and optimize)
5. machine learning (Tuning) (automatize these processes)

### The Path to ML

1. **Individual contributor:**

Great for prototypes -> fail quickly & iterate
Dangers of skipping this step: 

- inability to scale
- Product heads make big incorrect assumptions that are hard to change later

Dangers of lingering too long: 

- one Person gets skilled and then leaves
- Fail to scale up the process to meet demand in time (process takes too long)

2. **Delegation:**

Ramp up to include more people
Dangers of skipping:

- Not forced to formalize the process
- great product learning opportunity due to inherent diversity in human responses
- Great ML systems will need humans in the loop (see pitfalls from earlier)

Dangers of lingering:

- paying high marginal cost to serve each user
- more voices will say automation isnt possible
  - -> organizational lock-in

3. **Digitization**

Automate mundane parts of the process

Dangers of skipping: 

- You will need the infrastructure for the ML 
- IT Project and ML success tied and the whole project will fail if either does need humans in the loop

Dangers of lingering:

- your competitors are collecting data and tuning their offers from these new insights -> this will give them an edge and you will fall behind

4. **Big Data and Analytics**

Measure and achieve data-driven success (reassess old views or processes)

Dangers of skipping:

- Unclean data means no ML training
- You cant measure success

Dangers of lingering:

- Limit the complexity of problems you can solve

5. **Machine Learning**



Questions to ask yourself during the 5 phases:

- Who **executes** the Process?
- How were the operational **Parameters** chosen?
- How were the parameters **Feedback** to execution?

## Module 4

How to identify the origins of bias in ML
make models inclusive
Evaluate ML models with biases

Agenda: 

- Evaluating metrics with inclusion for ml system
- equality of opportunity
- how to find errors in dataset using Facets

### Machine Learning and Human Bias

Bias happens through bias in data / bias of the people providing the data

**interaction bias**: if a certain shape or form or pattern in data is dominant, minority patterns might get ignored by the ML system due to lack of data on that pattern

**latent bias:** for example scientists in the past have been mostly men, so if you train a system on pictures of past scientists, the system will think scientists are men

**selection bias:** Are you making sure to include everything and everybody in your selection?

### Evaluating metrics for inclusion

Evaluate model over subgroups

-> Confusion matrix for every Subgroup 

|            |          | Model Predictions                                            |                                                              |
| ---------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|            |          | Positive                                                     | Negative                                                     |
| **Labels** | Positive | True Positives (TP) <br />Label says something exists, <br />model predicts it | False Negative (FN)<br />Label says something exists,<br />Model doesn't predict it<br />Type II Error |
|            | Negative | False Positives (FP)<br />Something doesn't exist, but Model predicts it<br />Type I Error | True Negatives (TN)<br />Model predicts correctly that there's nothing there |

**False Negative Rate** = $$\frac{False Negatives}{FalseNegatives+TruePositives}$$

**False positive rate** is the fraction of the faces that the ML model detects that are not really faces = $\frac{False Positives}{False Positives + True Negatives}$

**Accuracy (ACC):** $\frac{\sum True Positive + \sum True Negative}{\sum Total Population}$

**Precision:** $\frac{\sum True Positive}{\sum Predicted Condition Positive}$

False Positives might be better than False Negatives (blurring out personal data or sexual content) , Sometimes False negatives are better than false positives (marking something as spam)

Simulating decisions with group unaware holds everyone to the same standard, which can be unfair to some groups

**Facets:** gives users a quick understanding of the distribution of values across features of their datasets

## Module 5

Python notebooks in the cloud **Agenda**:

- AI Platform Notebooks
- Compute Engine and Cloud Storage
- Data Analysis with BigQuery
- Machine Learning APIs

Learn how to:

- Carry out data science tasks in notebooks
- rehost notebooks on the cloud
- execute ad-hoc queries at scale
- invoke pre-trained ML models from AI Platform Notebooks

### Cloud Datalab

ML often carried out in self-descriptive, sharable, executable notebooks (Jupyter Notebook Lab)
**AI Platform Notebooks** are a fully hosted version of JupyterLab notebook environment. It has replaced Cloud Datalab as the default notebook environment for Google Cloud Platform. 
**AI Platform ** (perviously known as Cloud ML Engine) makes it easy for ML devs, data scientists , and data engineers to take ML projects from ideation to production and deployment. 

### Compute Engine and Storage

**Compute Engine** allows to rent a VM to run workloads

**Cloud Storage** is durable, persistent and organized in buckets
Name of a bucket is globally unique

**Third Wave of Cloud:** Having Managed Services do the scaling for you. Autoscale your processes

### Pre-trained ML APIs

Vision API

- Vision API provides Label & web detection,

- OCR

- Logo detection

- Landmark detection
- crop hints
- explicit content detection

Video Intelligence API:

- Label detection
- shot change detection
- explicit content detection
- regionalization

Cloud Speech API

- Speech to text transcription
- Speech timestamps
- Profanity filtering
- Batch & streaming transcriptions
- Translate text
- language detection

Translation and NL

- Extract entities
- Detect sentiment
- Analyze syntax
- Classify content
- 