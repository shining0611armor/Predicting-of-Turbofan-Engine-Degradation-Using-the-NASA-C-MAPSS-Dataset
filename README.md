<!DOCTYPE html>
<html lang="en">

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

<img src="Images/screenshot009.png" alt="Dataset Example" style="width:30%;" class="center">

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



<p>The preprocessing of the dataset includes four steps: Data Selection, Data Normalization, Data Labeling, and Time Windowing.</p>

<h3>Data Selection</h3>

<p>In the Data Selection phase, sensors that do not show significant changes with the increase of the life cycle are removed from the dataset. These sensors do not provide valuable information for determining the Remaining Useful Life (RUL) and only add complexity to the network. According to the reference paper, the following columns are removed: <code>c3</code>, <code>s1</code>, <code>s5</code>, <code>s10</code>, <code>s16</code>, <code>s19</code>. We follow the same approach in this implementation and remove the specified columns, although a few other columns could also be removed.</p>

<p>After applying Data Selection, the training dataset is as shown in Figure 1. The same operations are applied to the test dataset.</p>

<pre><code># we remove the specified columns
columns_to_remove = ['rows', 'op_setting_3', 'sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_19']
train_df.drop(columns=columns_to_remove, inplace=True)
test_df.drop(columns=columns_to_remove, inplace=True)
</code></pre>

<img src="screenshot007" alt="Data Selection Applied to Training Dataset" />
<p><em>Figure 1: Data Selection applied to the training dataset.</em></p>

<p>As shown in Figure 1, the specified columns have been removed from the dataset. This process is repeated for the test dataset. Note that the first two columns are not involved in the training process and will be separated in later stages. The same process is applied to the test dataset, and finally, 18 features are selected for training and testing.</p>

<p>If processing directly from the original text file, the corresponding sensor columns are identified and removed based on their indices. The columns to be removed are:</p>

<pre><code>[0, 1, 4, 5, 9, 14, 20, 23]
</code></pre>

<p>Thus, these columns are removed from the dataset. For example, after removing these columns from the test dataset, the following code is used:</p>

<pre><code># Data Selection implementation
engine_time_df = test_df[['engine_number', 'time_in_cycles']].copy()
columns_to_be_dropped1 = [0, 1, 4, 5, 9, 14, 20, 23]

columns_to_be_dropped = ['engine_number', 'time_in_cycles', 'rows', 'op_setting_3', 'sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_19']

test_data_dropped = test_data.drop(columns=columns_to_be_dropped1)
test_df_dropped = test_df.drop(columns=columns_to_be_dropped)
</code></pre>

<h3>Data Normalization</h3>

<p>Since the data is captured from various sensors, they have different ranges. Some sensors may have large measurements while others may have smaller values. Encountering different scales of values (very large or very small) can make the learning process difficult for the network. This can impose unnecessary heavy computations and cause computational saturation in the network, potentially biasing the network towards features with large values. To prevent this, normalization and standardization techniques are used. According to the approach presented in the reference paper, sensor features are initially mapped to the range of 0 to 1 using the following formula.</p>

<p>After Data Selection:</p>

<img src="capture1" alt="" />

<pre><code>from sklearn.preprocessing import MinMaxScaler

features_to_normalize = train_df.columns[3:]

train_df[features_to_normalize] = train_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')
test_df[features_to_normalize] = test_df[features_to_normalize].apply(pd.to_numeric, errors='coerce')

# we use the min-max-scaler in sklearn lib
scaler = MinMaxScaler()
train_df[features_to_normalize] = scaler.fit_transform(train_df[features_to_normalize])
test_df[features_to_normalize] = scaler.transform(test_df[features_to_normalize])
</code></pre>

<p>After applying Data Normalization, the training dataset is as shown in Figure 2. The same normalization is applied to the test dataset as well.</p>

<img src="screenshot011" alt="Normalized Test Dataset" />
<p><em>Figure 2: Normalized test dataset.</em></p>

<img src="screenshot012" alt="Normalized Training Dataset" />
<p><em>Figure 3: Normalized training dataset.</em></p>


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

