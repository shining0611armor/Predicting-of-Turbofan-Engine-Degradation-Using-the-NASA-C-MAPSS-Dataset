<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Implementing a Method for Predicting the Remaining Life of Turbofan Engine Degradation Using the NASA C-MAPSS Dataset</title>
</head>
<body>

<h1>Implementing a Method for Predicting the Remaining Life of Turbofan Engine Degradation Using the NASA C-MAPSS Dataset</h1>

<p>This repository contains the implementation of a method to predict the remaining useful life (RUL) of turbofan engines based on the NASA C-MAPSS dataset. The approach is based on the reference paper by Shcherbakov et al., 2022, and includes data preprocessing, model training, and evaluation steps.</p>

<h2>Table of Contents</h2>
<ul>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#data-preprocessing">Data Preprocessing</a></li>
    <li><a href="#dataset-description">Dataset Description</a></li>
    <li><a href="#model-implementation">Model Implementation</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#references">References</a></li>
</ul>

<h2 id="introduction">Introduction</h2>

<p>The goal of this project is to develop an intelligent maintenance system for turbofan engines. By predicting the remaining useful life of these engines, we can plan maintenance activities more effectively, reduce downtime, and improve overall safety.</p>

<h2 id="data-preprocessing">Data Preprocessing</h2>

<p>First, we examine and prepare the dataset. We need to load the dataset:</p>

<pre><code class="python">
import pandas as pd

!pip install gdown

# Downloading and unzipping the dataset
!gdown --id 1uNIreOQ1GWxIu_Aw-YWLT-xobcSWV2Kj -O /content/dataset.zip
!unzip -q /content/dataset.zip -d /content/dataset
!rm /content/dataset.zip
</code></pre>

<h3>Dataset Description</h3>

<p>The dataset used is the Turbofan Engine Degradation dataset from NASA's C-MAPSS collection. This dataset includes sensor outputs from a set of simulated turbofan jet engines. The sensor outputs are recorded as time-series data.</p>

<img src="screenshot009.png" alt="Dataset Example" style="width:30%;">

<p>In the training dataset, the engine starts operating normally until a fault occurs, which gradually increases. In the training dataset, this fault escalation continues until failure, after which no data is recorded.</p>

<p>The recorded information includes operational settings of the engines, recorded in 3 columns, and measurements from 21 sensors. This information is recorded for 100 units. However, the recorded data (remaining cycles) before failure in the test set is provided in a separate file because the test data ends before reaching failure.</p>

<p>Overall, the dataset is recorded in 26 columns, which are as follows:</p>

<pre><code>
1) unit number
2) time, in cycles
3) operational setting 1
4) operational setting 2
5) operational setting 3
6) sensor measurement 1
7) sensor measurement 2
...
26) sensor measurement 26

Data Set: FD001
Train trajectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)
</code></pre>

<p>To work with this dataset, we can read the CSV files as shown below, or use the original text files. We have examined both methods.</p>

<pre><code class="python">
import pandas as pd

# we load the training and test sets
train_df = pd.read_csv('/content/dataset/train_FD001.csv', header=None)
test_df = pd.read_csv('/content/dataset/test_FD001.csv', header=None)
rul_df = pd.read_csv('/content/dataset/RUL_FD001.csv', header=None)

columns = ['rows', 'engine_number', 'time_in_cycles'] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
train_df.columns = columns
test_df.columns = columns[:3] + [f'op_setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

train_df = train_df.iloc[1:].reset_index(drop=True)
test_df = test_df.iloc[1:].reset_index(drop=True)
</code></pre>

<p>The goal of working with this dataset is to predict the number of operational cycles remaining before failure, as creating a warning system before failure is crucial. Naturally, if the degradation falls below a certain threshold, it indicates failure.</p>

<h2 id="model-implementation">Model Implementation</h2>

<p>The implementation includes training a machine learning model to predict the remaining useful life of the engines based on the preprocessed dataset. The model's performance is evaluated using standard metrics to ensure its accuracy and reliability.</p>

<h2 id="results">Results</h2>

<p>The results section showcases the performance of the implemented model, including accuracy, precision, recall, and other relevant metrics. Visualization of the predictions versus actual remaining life is also provided.</p>

<h2 id="references">References</h2>

<ul>
    <li>Shcherbakov, et al., 2022. [Reference Paper Title]</li>
</ul>

</body>
</html>

