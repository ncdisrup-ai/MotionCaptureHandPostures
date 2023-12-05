# Classify Motions Worms
Classify (time series classification) and Study the motion of Worms (time series forecasting)
  
## 🤔 What is this?
**Description:**  The main goal of this project is to answer to those questions:

**(Objective 1)** Can we classify the type of worm using the information provided by the eigenworm series?

**(Objective 2)** For a specific worm, how can we model its motion, i.e., the eigenworm?

Detail:

**(Objective 1)** Can we classify the type of worm using the information provided by the eigenworm series?

To answer the first research question of this project we will find the best classifier model and/or representation model for our dataset.
To do so we will try different scalers, models and representation methods combinations, with 4 Scalers, 4 Representation methods and 14 models.

Scalers:
* PowerTransformer()
* MinMaxScaler()
* StandardScaler()
* TimeSeriesScalerMeanVariance(mu=0, std=1)

Representation methods:
* PiecewiseAggregateApproximation(n_segments=12)
* PiecewiseAggregateApproximation(n_segments=16)
* SymbolicAggregateApproximation(n_segments=10, alphabet_size_avg=40)
* SymbolicAggregateApproximation(n_segments=32, alphabet_size_avg=40)

Models:
* LogisticRegression(C = 0.01)
* DecisionTreeClassifier(max_depth = 10)
* DecisionTreeClassifier(min_samples_leaf = 5)
* DecisionTreeClassifier(criterion = 'gini')
* DecisionTreeClassifier(criterion = 'entropy')
* GaussianNB()
* KNeighborsClassifier(n_neighbors = 1,weights = 'distance')
* KNeighborsTimeSeriesClassifier(n_neighbors = 1, metric = 'euclidean')
* KNeighborsTimeSeriesClassifier(n_neighbors = 1, metric = 'dtw')
* KNeighborsTimeSeriesClassifier(n_neighbors = 1, metric = 'sax',metric_params = dict)
* RandomForestClassifier(n_estimators=50, random_state=0)
* RandomForestClassifier(n_estimators=10, random_state=0)
* SVC(C=50,gamma='auto')
* SVC(C=10,gamma='auto')

We chose the C = 0.01 in the **LogisticRegression** Classification model because the dataset has a good number of features and low C values have a stronger regularization and helps prevent the overfitting of the model.

In the **DecisionTreeClassifier** model we tested a diverse set of hyperparameters using only one value for the max_depth and min_sample_leaf hyperparameters and 2 other simpler hyperparameters
criterion = gini and criterion = entropy.

Only one Naïve Bayes Model, that assumes that each variable is normally distributed and therefore can be modeled as a Gaussian.

We know from a baseline that the KNN model is intrinsically slower than most supervised learning models so we just chose one model to test.

The **KNeighborsTimeSeriesClassifier** model implements the k-nearest neighbor for time series. With this model we have three possible metrics:
* 1-NN with Euclidean distance
* 1-NN with DTW
* 1-NN with SAX, in this case we need to set two other parameters: `n_segments` and `alphabet_size_avg`. The first parameter means the number of Piecewise Aggregate Approximation pieces to compute and the latter is the number of SAX symbols to use. To fix these parameters, we use the parameter `metric_params` and provide a dictionary with the two parameters required.

We chose the **RandomForestClassifier** model because its one of the best classification models and it is suprisingly robust and does not overfit easily.

And finally the **Support Vector Classifier(SVC)** model, because SVMs are among the best learning algorithms extant and they are also extremely powerful.

The evaluation metric used to evaluate the classification models and representation methods performance was the **F1-Score**. 

To find the best classification models we only combined each model with different scalers. For the representation methods case, we combined each scaler with each representation method with each classifier model.

With that scenario in mind, we created 2 tables:
* The **Results for Classification Models Performance** table with the Precision, Recall, F1 and the Matthews Correlation Coefficient (MCC) scores, the name of the scaler used and the name of the model.
* The **Results for Representation Methods Performance** table with the same information has the first one but with an additional information about the representation method used.

The first ojective involves Data Understanding, research and design of Artificial Intelligence supervised learning algorithms for **classification** problem in time series

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


**(Objective 2)** For a specific worm, how can we model its motion, i.e., the eigenworm? 

Perform time series analysis to model the movement of one single worm. In this case, we will consider the worm in the train set indexed by 5.
we will work through a time series forecasting project from end-to-end, from downloading the dataset and defining the problem to training a final model and making predictions. The steps we will work through are as follows:
1. Problem description
2. Test harness
3. Persistence
4. Data analysis
5. ARIMA models
6. Model validation


## 📚 Data


Data with the projects (to train the model and to apply the model) are in data dir.
 
The data relates to 258 traces of worms converted into four "eigenworm" series. The eigenworm data are lengths from 17984 to 100674 (sampled at 30 Hz, so from 10 minutes to 1 hour) and in four dimensions (eigenworm 1 to 4). There are five classes: N2, goa-1, unc-1, unc-38 and un63. N2 is wildtype (i.e., normal) the other 4 are mutant strains. These datasets are the first dimension only (first eigenworm) averaged down so that all series are lengths 900 (the single hour-long
series is discarded). This smoothing is likely to discard discriminatory information. We address the problem of classifying individual worms as wild-type or mutant based on the time series of the first eigenworm, down-sampled to second-long intervals. We have 258 cases, which we split 70%/30% into a train and test set. Each series has 900 observations, and each worm is classified as either wild-type (the N2 reference strain – 109 cases; class 1) or one of four mutant
types (149 cases; class 2): goa-1 (44 cases); unc-1 (35 cases); unc-38 (45 cases) and unc-63 (25 cases). The data were extracted from the C. elegans behavioral database WormWeb.

The following files are provided:
- 'worms_trainset.csv' – file with the training dataset, where the first column has the class label (181x901 matrix)
- 'worms_testset.csv' – file with the test dataset, where the first column has the class label (77x901 matrix)

##  🚀 Quick Install


jupiter notebook (.ipynb) file in model dir


## 📖 Documentation


Please see the description in .ipynb about this project.


##  🚀 Results 


**(Objective 1)** Can we classify the type of worm using the information provided by the eigenworm series?

The best classification model obtained was the **SupportVectorClassifier(C=10,gamma='auto')** with a **StandarScaler()** scaler with a `Precison Score` of 0.5854, `Recall Score` of 0.7500, `F1-Score` of 0.6575 and a `MCC Score` of 0.3602.
The Support Vector Machine classifier model got the best result (SVC), with the hyperparameter C = 10 and gama  = 'auto. **The gamma parameter controls the width of the Gaussian Kernel** it determines the scale of what it means for points to be close together. The **C parameter** is a regularization parameter similar to the linear models, it limits the importance of each point, so a low **C** can avoid the overfitting of the model.
This makes sense in being the best classification model combined with the Scaler **StandardScaler()**, as we found in the EDA the dataset has outliers and the  StandarScaler manage to handle outliers.

The best representation method obtained was the **PiecewiseAggregateApproximation(n_segments=12)** with a **StandardScaler()** scaler and a **SupportVectorClassifier(C=10,gamma='auto')** model with `Precison Score` of 0.5854, `Recall Score` of 0.7500, `F1-Score` of 0.6575 and a `MCC Score` of 0.3602. The same values and same best model that without  representation method.

The **PiecewiseAggregateApproximation** representation method reduces the dimensionality of the time series dataset by splitting them into 12 equal-sized segments in this case which are computed by averaging the values in these segments. The StandardScaler makes all variables of the dataset directly comparable by subtracting the mean and dividing it by the standard deviation. As we can see here the representation method doesn't improve the performance.

The dataset has only 258 traces of worn (180 traces for training, from which 23 were duplicates so just 157 traces for training, and 78 traces for testing). Each trace had 900 observations. We just needed to classify each trace of worn like class 1:wild-type (has 109 cases: training+test), or class 2 mutant (four types) (has 149 cases: training+test). The training dataset (157 traces for training,) had no nulls, and just a few outliers (we used scaler). The best results that we got with classification models was: precision = 0.5854; recall = 0.7500, f1 = 0.6575; mcc = 0.3602 for two classes..... After analyzing the results, we can conclude that we can classify the type of worm using the information provided by the eigenworm, but the **dataset was small**, with only 157, w/ no duplicates, traces for trainning and 78 traces for testing, that made our classification challenging, so the **results are promising, but there is still room for improvement, mainly if we obtain more data to train**.

**(Objective 2)** For a specific worm, how can we model its motion, i.e., the eigenworm?
   
We can see that the **results for the validation dataset were good**, because:
- the RMSE for validation dataset (RMSE=0.028) was even better than the RMSE for TrainTest dataset (RMSE=0.036) and the persistency model (RMSE=0,029)
- The prediction and the validation set are almost 100% aligned 

The model is validated
