# Motion Capture Hand Postures
Classify  the motion capture from hand postures through supervised learning models

  
## 🤔 What is this?
**Description:**  Provide the best possible classification models Testing exclusively the following models

- Linear models
- Tree Based models
- Naive Bayes
- K-Nearest Neighbours

For **Motion Capture from Hand Postures**.

Notes for processing the data:
- The Variable to Classify is Class.
- Pay attention to the variable User as it is numeric and probably should be considered categorical
- Eliminate the first row of the data set as there are only Zeros
- There are plenty of missing values identified by '?' These should be handled accordingly

Models should examine different hyperparameters and select the best one [Remember: everything else being similar, the simplest models should be prefered] You should do Simple Cross Validation (testing=25% of all data) for evaluating the models, but the same data partition must be used for all models - DO NOT USE KFOLD CV

This involves Data Understanding, research and design of Artificial Intelligence supervised learning algorithms for **classification** problem in time series

A typical data science project has several phases (Cross-Industry Standard Process for Data Mining (CRISP-DM)). Phases adapted to this project
1. Business Understanding: Understand the business problem and define the objectives of the project.

2. Data Understanding: Understand the data and its structure. We look at the data types, missing values, and other characteristics of the data. Discover and visualize the data to gain insights.
- Load Data
- PCA for reducing features (Need for?)
- Exploratory Data Analysis (EDA): Data Analysis Process with estatistical analysis and data visualization for knowing data and obtain findings
a) Data Exploration: Examining summary statistics, visualizing data distributions, and identifying patterns or relationships.
b) Data Visualization: Presenting insights through plots, charts, and graphs to communicate findings effectively

3. Data Preparation: Prepare the data for modeling by cleaning it up and transforming it into a format that can be used by machine learning algorithms.  
- Data Cleaning: Handling missing values and categorical values, removing outliers and duplicates, and ensuring data quality.
- Data Transforming
- Feature Engineering: Transforming variables, creating new features, or selecting relevant variables for analysis.
-- Feature selection: selecting the most useful features to train on among existing features. 
-- Feature extraction: combining existing features to produce a more useful one (e.g. handling missing data, encoding variables, dealing with categorical variables, dimensionality reduction algorithms ...). 
-- Creating new features by gathering new data.
- PCA for reducing features (or in the begining after loading data) (need for?)
- Data Visualization after Data Preparation: Presenting insights through plots, charts, and graphs to communicate findings effectively

4. Modeling / Select and Train Models: We create a model that can be used to make predictions or classify new data. 
- Retrieve Class (y) from the dataset (X)
- Split Data
- Normal Test
- OverSampling (SMOTE) and undersampling (need for?)
- Calculate the weights for each sample based on the feature (transaction amount) and class weight (need for?)
- Intermediate Results (need for?)
- Train model (w/cross-validate) (based on OverSampling (SMOTE) and undersampling) (need for?)

5. Evaluation: We evaluate the model’s performance and determine whether it meets the business objectives.
- Evaluation using cross-validate
- Fine-tune Models  
- Model Evaluation on Test Set

6. Deployment Deploy the model into production and monitor its performance over time.


## 📚 Data


Data with the projects (to train the model and to apply the model) are in data dir.
 
Task Dataset: Motion Capture Hand Postures
https://archive.ics.uci.edu/dataset/405/motion+capture+hand+postures

The files 'Postures.csv' is provided

5 types of hand postures from 12 users were recorded using unlabeled markers on fingers of a glove in a motion capture environment. Due to resolution and occlusion, missing values are common.

Due to the manner in which data was captured, it is likely that for a given record and user there exists a near duplicate record originating from the same user. We recommend therefore to evaluate classification algorithms on a leave-one-user-out basis wherein each user is iteratively left out from training and used as a test set. One then tests the generalization of the algorithm to new users. A 'User' attribute is provided to accomodate this strategy. 

instances: 78095
Dataset Characteristics
Multivariate

Data is provided as a CSV file. A header provides the name of each attribute. An initial dummy record composed entirely of 0s should be ignored. A question mark '?' is used to indicate a missing value. A record corresponds to a single instant or frame as recorded by the camera system.

'Class' - Integer. The class ID of the given record. Ranges from 1 to 5 with 1=Fist(with thumb out), 2=Stop(hand flat), 3=Point1(point with pointer finger), 4=Point2(point with pointer and middle fingers), 5=Grab(fingers curled as if to grab).
'User' - Integer. The ID of the user that contributed the record. No meaning other than as an identifier.
'Xi' - Real. The x-coordinate of the i-th unlabeled marker position. 'i' ranges from 0 to 11.
'Yi' - Real. The y-coordinate of the i-th unlabeled marker position. 'i' ranges from 0 to 11.
'Zi' - Real. The z-coordinate of the i-th unlabeled marker position. 'i' ranges from 0 to 11.

Each record is a set. The i-th marker of a given record does not necessarily correspond to the i-th marker of a different record. One may randomly permute the visible (i.e. not missing) markers of a given record without changing the set that the record represents. For the sake of convenience, all visible markers of a given record are given a lower index than any missing marker. A class is not guaranteed to have even a single record with all markers visible.


##  🚀 Quick Install


jupiter notebook (.ipynb) file in model dir


## 📖 Documentation


Please see the description in .ipynb about this project.


##  🚀 Results 


Executing the list of scalers, imputers,models in a total of **126 models**  we can verify that RandomForest, with standard scaler and simple imputer has the better f1 (0.9901). In terms of models ther macro order is RandomForest, DecisionTree , KNN (some time can surpass DecisionTree depending on the hyperparameters), GaussianNB

