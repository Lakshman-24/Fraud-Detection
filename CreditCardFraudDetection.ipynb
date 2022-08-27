Importing necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import make_scorer,f1_score,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score,roc_curve
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import binarize, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
%matplotlib inline
Creating functions which are being used repeateadly
# define a function to print accuracy metrics
def print_accuracy_metrics(Input,Output):
  print("Recall:", recall_score(Input, Output))
  print("Log Loss:", log_loss(Input, Output))
  print("Precision:", precision_score(Input, Output))
  print("Accurcay:", accuracy_score(Input, Output))
  print("AUC: ", roc_auc_score(Input, Output))
  print("F1 Score:", f1_score(Input, Output))
  confusion_matrix_value = confusion_matrix(Input,Output)
  print('Confusion matrix:\n', confusion_matrix_value)
  class_names=[0,1] # name  of classes
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  # create heatmap
  sns.heatmap(pd.DataFrame( confusion_matrix_value), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  
# defined a function to print cross validation score
scoring = {'recall' : make_scorer(recall_score)}
def cross_validation_metrics(log_reg, X, y):
 log_reg_score = cross_val_score(log_reg, X,y,cv=5,scoring='recall')
 print('Logistic Regression Cross Validation Score(Recall): ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')
# function for plotting feature importance
def feature_importance(model, X):
  importances = model.feature_importances_
  std = np.std([tree.feature_importances_ for tree in model.estimators_],
               axis=0)
  indices = np.argsort(importances)[::-1]

  # Print the feature ranking
  print("Feature ranking:")

  for f in range(X.shape[1]):
      print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

  # Plot the feature importances of the forest
  plt.figure()
  plt.title("Feature importances")
  plt.bar(range(X.shape[1]), importances[indices],
          color="r", yerr=std[indices], align="center")
  plt.xticks(range(X.shape[1]), indices)
  plt.xlim([-1, X.shape[1]])
  plt.show()
# function to draw ROC curve
def plot_auc_curve(model,):
  auc = roc_auc_score(y, y_pred_prob)
  fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
  
  plt.plot(fpr, tpr)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.title('ROC Curve\n AUC={auc}'.format(auc = auc))
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.grid(True)
### Reading data as a pandas dataframe
data = pd.read_csv('creditcard.csv')
Data Exploration
#### Exploring data set
data.head()
id	Time	V1	V2	V3	V4	V5	V6	V7	V8	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
0	21749	58670.0	-0.854092	0.644458	1.805656	1.146369	-0.519127	1.844676	-0.935942	1.056104	...	0.193673	0.789467	0.218834	-0.577043	-0.727521	0.612977	-0.219109	-0.063157	11.50	0
1	105607	164361.0	-0.863534	0.291699	0.594479	-1.190707	0.117851	0.169880	0.065587	0.289947	...	-0.223345	-0.333300	-0.455269	0.185385	0.432974	0.931127	-0.414413	-0.284978	25.42	0
2	187884	38767.0	-1.192107	-0.896044	1.204410	-1.593935	0.432699	-1.101769	-0.299815	0.222793	...	0.470749	0.932440	0.159099	0.215700	-0.169315	-0.320951	0.310243	0.210535	58.75	0
3	238501	571.0	-2.355336	2.316182	0.701735	0.174501	0.677346	1.029705	0.792868	-0.060581	...	0.008872	0.955806	0.047292	-0.650140	-0.282282	-0.286391	0.335493	0.223061	2.89	0
4	252649	51507.0	-1.302336	1.016359	1.007046	-0.127051	0.435740	-0.092143	0.709650	0.590142	...	0.150091	0.059446	-0.262177	-0.354871	0.680078	-0.402172	-0.111834	-0.044427	51.59	0
5 rows × 32 columns

data.describe()
id	Time	V1	V2	V3	V4	V5	V6	V7	V8	...	V21	V22	V23	V24	V25	V26	V27	V28	Amount	Class
count	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	...	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000	244807.000000
mean	142394.376301	94817.409077	0.002267	-0.002347	0.001627	0.000226	0.000116	-0.000695	0.000235	0.000451	...	-0.000663	0.000085	0.000678	0.000009	0.000082	-0.000326	0.000042	0.000369	88.221638	0.001724
std	82214.109334	47544.660749	1.958778	1.655048	1.511011	1.414850	1.378017	1.332263	1.229381	1.174459	...	0.725261	0.725420	0.622169	0.606201	0.520588	0.482409	0.404819	0.330222	250.663749	0.041483
min	1.000000	0.000000	-56.407510	-72.715728	-48.325589	-5.683171	-113.743307	-26.160506	-37.060311	-50.943369	...	-22.757540	-10.933144	-36.666000	-2.836627	-8.696627	-2.604551	-22.565679	-15.430084	0.000000	0.000000
25%	71174.500000	54134.500000	-0.918719	-0.597789	-0.888899	-0.846298	-0.690880	-0.767694	-0.553464	-0.207827	...	-0.228840	-0.543300	-0.161045	-0.354847	-0.316905	-0.327190	-0.070876	-0.052801	5.610000	0.000000
50%	142272.000000	84747.000000	0.019709	0.065045	0.180200	-0.020248	-0.055758	-0.273607	0.039812	0.022329	...	-0.029477	0.006637	-0.010750	0.041093	0.015969	-0.052227	0.001480	0.011233	22.000000	0.000000
75%	213591.500000	139393.000000	1.315616	0.802233	1.027866	0.741737	0.610432	0.397342	0.569658	0.326222	...	0.186693	0.530219	0.147784	0.439776	0.350486	0.240038	0.091125	0.078232	77.050000	0.000000
max	284807.000000	172792.000000	2.454930	19.167239	9.382558	16.875344	34.801666	73.301626	120.589494	20.007208	...	27.202839	8.272233	22.528412	4.584549	7.519589	3.517346	31.612198	33.847808	25691.160000	1.000000
8 rows × 32 columns

Observations
data.shape
(244807, 32)
data.columns
Index(['id', 'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class'],
      dtype='object')
#### Checking for null values in dataset
data.isnull().sum().max()
0
#### There are no null values in dataset 
####  Checking for unique values of ids
data.id.nunique()
244807
Data is pretty clean and there are no duplicate ids are present now let's check distribution of each feature
# Plot the histograms of each 
data.hist(bins=50, figsize=(30,20))
plt.show()

We can observe that all the features in dataset are scaled except amount and time.
So, in next step I am going to scale Amount column in dataset and delete time column.
data['normal_amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount','Time'], axis=1)
X = data.loc[:,data.columns != 'Class']
y = data.loc[:,data.columns == 'Class']
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 30

ax1.hist(data.normal_amount[data.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(data.normal_amount[data.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()

We can observe from the above figure that the fraud transactions amount is very less.
In next step I am going to visualize number of fraud transactions and number of Non-fraud transactions.
# Now lets check the class distributions
sns.countplot("Class",data=data)
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
Text(0, 0.5, 'Frequency')

As you can observe from the plot, we have so many 0 (non-fraud) compared to 1 (fraud).
This kind of imbalance in the target variable is known as class imbalance.
# Showing ratio
print("Percentage of normal transactions: ", len(data[data.Class == 0])/len(data))
print("Percentage of fraud transactions: ", len(data[data.Class == 1])/len(data))
print("Total number of transactions in data: ", len(data))
Percentage of normal transactions:  0.9982761930827141
Percentage of fraud transactions:  0.0017238069172858619
Total number of transactions in data:  244807
Most of the transactions were Non-Fraud (99.83%) of the time,
while Fraud transactions occurs 0.17% of the time in the dataframe.
Splitting original dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)
1] Logistic regression on imbalanced dataset
lr = LogisticRegression()
lr.fit(X_train,y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
# Accuracy metrics for 
y_pred = lr.predict(X_test)
cross_validation_metrics(lr,X_train,y_train)
print_accuracy_metrics(y_test,y_pred)
Logistic Regression Cross Validation Score(Recall):  60.16%
Recall: 0.6481481481481481
Log Loss: 0.021162677654140833
Precision: 0.9090909090909091
Accurcay: 0.9993872799313753
AUC:  0.8240263478860329
F1 Score: 0.7567567567567568
Confusion matrix:
 [[73328     7]
 [   38    70]]

Observatios
By observing the accuracy we can conclude that algorithm is performing extremely well . But it’s not true. As most of the labels 0, even random guess gives you 99% accuracy. So we need a better measure to understand the performance of the model.

Recall
Recall is a measure which measures the ability of model to predict right for a given label. In our case, we want to test the model how accurately it can recall fraud cases as we are interested in that. As you can observe from the results, the recall for 1.0 is only 0.6016 compared to 99% for 0. So our model is not doing a good job of recognising frauds. So this shows that how imbalanced data is effecting accuracy of model.

2] Using Class Weight (Logistic regression)
Scikit-learn logistic regression has a option named class_weight when specified does class imbalance handling implicitly. So trying to predict using this technique

lr_balanced = LogisticRegression(class_weight = 'balanced')
lr_balanced.fit(X_train,y_train)
LogisticRegression(C=1.0, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='warn', n_jobs=None, penalty='l2', random_state=None,
          solver='warn', tol=0.0001, verbose=0, warm_start=False)
y_balanced_pred = lr_balanced.predict(X_test)
cross_validation_metrics(lr_balanced,X_train,y_train)
print_accuracy_metrics(y_test,y_balanced_pred)
Logistic Regression Cross Validation Score(Recall):  89.86%
Recall: 0.9115646258503401
Log Loss: 0.7882701074435376
Precision: 0.06470304200869145
Accurcay: 0.9771777676345634
AUC:  0.9444277359227315
F1 Score: 0.12082957619477007
Confusion matrix:
 [[83359  1937]
 [   13   134]]

y_balanced_pred_prob = lr_balanced.predict_proba(X_test)[:, 1]
print('Prob:', y_balanced_pred_prob[0:20])
Prob: [0.11795312 0.10853555 0.16233188 0.04109433 0.06952336 0.8054823
 0.33638875 0.01525269 0.01663033 0.45291477 0.03846819 0.02975751
 0.00137332 0.00313013 0.00775784 0.05660248 0.01766819 0.00898342
 0.09351198 0.05154421]
print('Prob:', y_balanced_pred[0:20])
Prob: [0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
Undersampling of the dataset
Undersampling is one of the techniques used for handling class imbalance. In this technique, we under sample majority class to match the minority class. So in our example, we take random sample of non-fraud class to match number of fraud samples. This makes sure that the training data has equal amount of fraud and non-fraud samples.

number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = data[data.Class == 0].index
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
under_sample = data.iloc[under_sample_indices,:]
under_sample.shape
(844, 30)
So there are total 844 observations in our undersample dataframe.

Visualising Undersampled Data
# Now lets check the class distributions
sns.countplot("Class",data=under_sample)
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
Text(0, 0.5, 'Frequency')

In the above plot, you can observe that classes are distributed evenly now.
If we try to correlate class and features on imbalanced dataset then it will be of no use because we will not see true correlations of features with result. While now I am going to see the features and their correlations w.r.t class on undersampled dataframe.

# correlation matrix
corrmat =under_sample.corr()
fig,ax= plt.subplots()
fig.set_size_inches(25,15)
sns.heatmap(corrmat,square=True)
<matplotlib.axes._subplots.AxesSubplot at 0x1cea74d0>

rf = RandomForestClassifier(n_estimators=100, 
                            criterion='gini', 
                            max_features='sqrt',
                            n_jobs=-1)
rf.fit(X_under_train, y_under_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
feature_importance(rf,X_under_train)
Feature ranking:
1. V14 (0.163468)
2. V10 (0.142475)
3. V4 (0.135820)
4. V17 (0.077558)
5. V3 (0.066908)
6. V12 (0.063024)
7. V11 (0.059748)
8. V7 (0.035103)
9. V2 (0.029975)
10. V16 (0.025015)
11. V18 (0.020058)
12. V8 (0.014964)
13. V6 (0.014854)
14. V5 (0.011935)
15. V23 (0.011827)
16. V9 (0.011814)
17. V19 (0.011713)
18. V13 (0.011508)
19. V1 (0.010258)
20. V27 (0.010227)
21. V20 (0.010050)
22. normal_amount (0.009788)
23. V21 (0.008971)
24. V22 (0.008304)
25. V15 (0.008091)
26. V28 (0.006979)
27. V26 (0.006722)
28. V25 (0.006679)
29. V24 (0.006168)

From the above heeatmap we can say that many of features are correlated but we are more interested in correlation of features with class. So I am going to list those features whose correlation coefficient w.r.t class is less than -0.6 or greater than 0.6
#negative correlations smaller than -0.5
corr = under_sample.corr()
corr = corr[['Class']]
corr[corr.Class < -0.6]
Class
V10	-0.633606
V12	-0.677653
V14	-0.740321
#positive correlations greater than 0.5
corr[corr.Class > 0.6]
Class
V4	0.698864
V11	0.688469
Class	1.000000
BoxPlots
We will use boxplots to have a better understanding of the distribution of these features in fradulent and non fradulent transactions.

#visualizing the features with high correlation
f, axes = plt.subplots(nrows=2, ncols=3, figsize=(26,16))
f.suptitle('Features With High Correlation', size=35)
sns.boxplot(x="Class", y="V10", data=under_sample, ax=axes[0,0])
sns.boxplot(x="Class", y="V12", data=under_sample, ax=axes[0,1])
sns.boxplot(x="Class", y="V14", data=under_sample, ax=axes[0,2])
sns.boxplot(x="Class", y="V4", data=under_sample, ax=axes[1,0])
sns.boxplot(x="Class", y="V11", data=under_sample, ax=axes[1,1])
f.delaxes(axes[1,2])

Box plots provide us with a good intuition of whether we need to worry about outliers as all transactions outside of 1.5 times the IQR (Inter-Quartile Range) are usually considered to be outliers. However, removing all transactions outside of 1.5 times the IQR would dramatically decrease training data size, which is not very large, to begin with. Thus, I decided to only focus on extreme outliers outside of 2.5 times the IQR.

under_sample.shape
(844, 30)
Removing extreme outliers
v14_fraud = under_sample['V14'].loc[under_sample['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
v14_iqr = q75 - q25
v14_cut_off = v14_iqr * 2.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
under_sample= under_sample.drop(under_sample[(under_sample['V14'] > v14_upper) | (under_sample['V14'] < v14_lower)].index)
v12_fraud = under_sample['V12'].loc[under_sample['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25
v12_cut_off = v12_iqr * 2.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
under_sample= under_sample.drop(under_sample[(under_sample['V12'] > v12_upper) | (under_sample['V12'] < v12_lower)].index)
v10_fraud = under_sample['V10'].loc[under_sample['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25
v10_cut_off = v10_iqr * 2.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
under_sample= under_sample.drop(under_sample[(under_sample['V10'] > v10_upper) | (under_sample['V10'] < v10_lower)].index)
v4_fraud = under_sample['V4'].loc[under_sample['Class'] == 1].values
q25, q75 = np.percentile(v4_fraud, 25), np.percentile(v4_fraud, 75)
v4_iqr = q75 - q25
v4_cut_off = v4_iqr * 2.5
v4_lower, v4_upper = q25 - v4_cut_off, q75 + v4_cut_off
under_sample= under_sample.drop(under_sample[(under_sample['V4'] > v4_upper) | (under_sample['V4'] < v4_lower)].index)
v11_fraud = under_sample['V11'].loc[under_sample['Class'] == 1].values
q25, q75 = np.percentile(v11_fraud, 25), np.percentile(v11_fraud, 75)
v11_iqr = q75 - q25
v11_cut_off = v11_iqr * 2.5
v11_lower, v11_upper = q25 - v11_cut_off, q75 + v11_cut_off
under_sample= under_sample.drop(under_sample[(under_sample['V11'] > v11_upper) | (under_sample['V11'] < v11_lower)].index)
under_sample.shape
(976, 30)
After removing outliers our under sample dataframe is reduced to
I have tried to run this notebook without removing outliers and after removing outliers but I got better result before removing outliers.Before even thinking of removing outliers there should be enough evidence that these observations are actual outliers Before removing outliers. It should be done by in depth statistical analysis to make sure that these observations are actual outliers because different ML methods used for detecting fraud, are based on anomaly detection and they treat such extreme outliers as frauds. So, by deleting it we delete the most important observations, that have higher probability of being frauds

Splitting under sampled dataframe
X_under = under_sample.loc[:,under_sample.columns != 'Class']
y_under = under_sample.loc[:,under_sample.columns == 'Class']
X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(X_under,y_under,test_size = 0.3, random_state = 0)
3] Logistic regression with C=0.01
lr_under_C1 = LogisticRegression(C=0.01,penalty = 'l1')
lr_under_C1.fit(X_under_train,y_under_train)
LogisticRegression(C=0.01, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
# Prediction on original dataframe
y_pred_full_model1 = lr_under_C1.predict(X_test)
cross_validation_metrics(lr_under_C1,X_under_train,y_under_train)
print_accuracy_metrics(y_test,y_pred_full_model1)
Logistic Regression Cross Validation Score(Recall):  96.23%
Recall: 0.9115646258503401
Log Loss: 3.5007281778367507
Precision: 0.01526022093155677
Accurcay: 0.8986458808796508
AUC:  0.9050941212162975
F1 Score: 0.030017921146953407
Confusion matrix:
 [[76649  8647]
 [   13   134]]

4] Logistic regression with C=0.1
lr_under_C2 = LogisticRegression(C=0.1,penalty = 'l1')
lr_under_C2.fit(X_under_train,y_under_train)
LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
# Prediction on original dataset
y_pred_full_model2 = lr_under_C2.predict(X_test)
cross_validation_metrics(lr_under_C2,X_under_train,y_under_train)
print_accuracy_metrics(y_test,y_pred_full_model2)
Logistic Regression Cross Validation Score(Recall):  88.99%
Recall: 0.9115646258503401
Log Loss: 0.6209142741882784
Precision: 0.08086904043452021
Accurcay: 0.9820231031213792
AUC:  0.9468545789165413
F1 Score: 0.14855875831485585
Confusion matrix:
 [[83773  1523]
 [   13   134]]

5] Logistic regression with C=1
lr_under_C3 = LogisticRegression(C=1,penalty = 'l1')
lr_under_C3.fit(X_under_train,y_under_train)
LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
# Prediction on original dataset
y_pred_full_model3 = lr_under_C3.predict(X_test)
cross_validation_metrics(lr_under_C3,X_under_train,y_under_train)
print_accuracy_metrics(y_test,y_pred_full_model3)
Logistic Regression Cross Validation Score(Recall):  91.19%
Recall: 0.9259259259259259
Log Loss: 1.4306248896684717
Precision: 0.03190810465858328
Accurcay: 0.9585801233609739
AUC:  0.9422770694605426
F1 Score: 0.061690314620604564
Confusion matrix:
 [[70301  3034]
 [    8   100]]

6] Logistic regreesion with C=10
lr_under_C4 = LogisticRegression(C=10,penalty = 'l1')
lr_under_C4.fit(X_under_train,y_under_train)
LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l1', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
# Prediction on original dataset
y_pred_full_model4 = lr_under_C4.predict(X_test)
cross_validation_metrics(lr_under_C4,X_under_train,y_under_train)
print_accuracy_metrics(y_test,y_pred_full_model4)
Logistic Regression Cross Validation Score(Recall):  90.51%
Recall: 0.9259259259259259
Log Loss: 1.6827008224584545
Precision: 0.027247956403269755
Accurcay: 0.951281946543578
AUC:  0.9386226070619608
F1 Score: 0.05293806246691372
Confusion matrix:
 [[69765  3570]
 [    8   100]]

7] Decision Tree Classifier
DecisionTreeClassifier= DecisionTreeClassifier()
DecisionTreeClassifier.fit(X_under_train,y_under_train)
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
# Prediction on original dataset
y_pred_DecisionTree = DecisionTreeClassifier.predict(X_test)
cross_validation_metrics(DecisionTreeClassifier,X_under_train,y_under_train)
print_accuracy_metrics(y_test,y_pred_DecisionTree)
Logistic Regression Cross Validation Score(Recall):  87.12%
Recall: 0.9722222222222222
Log Loss: 3.9090580892603124
Precision: 0.012479201331114808
Accurcay: 0.886823795324265
AUC:  0.9294601259062294
F1 Score: 0.02464210279277165
Confusion matrix:
 [[65026  8309]
 [    3   105]]

plt.figure(0).clf()

fpr, tpr, thresh = metrics.roc_curve(y_test, y_balanced_pred)
auc = metrics.roc_auc_score(y_test, y_balanced_pred)
plt.plot(fpr,tpr,label="Logistic Regtession Class weight, auc="+ '{0:.3f}'.format(auc))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_full_model1)
auc = metrics.roc_auc_score(y_test, y_pred_full_model1)
plt.plot(fpr,tpr,label="Logistic regression(C=0.01), auc="+ '{0:.3f}'.format(auc))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_full_model2)
auc = metrics.roc_auc_score(y_test, y_pred_full_model2)
plt.plot(fpr,tpr,label="Logistic regression(C=0.1), auc="+'{0:.3f}'.format(auc))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_full_model3)
auc = metrics.roc_auc_score(y_test, y_pred_full_model3)
plt.plot(fpr,tpr,label="Logistic regression(C=1), auc="+'{0:.3f}'.format(auc))

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_full_model4)
auc = metrics.roc_auc_score(y_test, y_pred_full_model4)
plt.plot(fpr,tpr,label="Logistic regression(C=10), auc="+'{0:.3f}'.format(auc))

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve\n AUC={auc}'.format(auc = auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend(loc="lower right")
<matplotlib.legend.Legend at 0x2293b290>

Obviously, trying to increase recall, tends to come with a decrease of precision. However, in our case, if we predict that a transaction is fraudulent and turns out not to be, is not a massive problem compared to the opposite.

Predictions on new samples
data_test=pd.read_csv('creditcard_test_dataset.csv')
data_test.head()
id	Time	V1	V2	V3	V4	V5	V6	V7	V8	...	V20	V21	V22	V23	V24	V25	V26	V27	V28	Amount
0	30808	109847	1.930307	-0.234417	-1.583368	0.024786	1.016470	1.179664	-0.386242	0.327416	...	-0.437585	0.211402	1.015098	0.073241	-0.804496	0.072202	-0.383211	0.016787	-0.081815	1.00
1	174948	84730	-5.053316	-3.617236	-0.323455	1.447171	-0.097495	-0.154917	-2.076441	1.331307	...	-1.351008	-0.202483	0.036442	-1.901898	0.090641	-1.777126	0.947972	0.954071	-1.577919	333.48
2	203982	95957	0.090684	1.197902	-1.370219	0.770437	0.857203	-0.698486	1.350617	-0.952780	...	0.617074	-0.255027	0.378177	-0.126596	0.572250	0.347829	0.538705	-0.211582	-0.267241	50.14
3	283146	142109	1.808668	-1.197522	-2.291089	-0.690236	-0.312654	-1.756994	0.631548	-0.725965	...	0.457894	0.498576	0.927097	-0.283918	0.134011	0.372327	0.050185	-0.121994	-0.038088	254.04
4	47316	129034	-0.920372	1.003867	-0.110709	-2.771173	0.841827	-0.497278	0.769851	0.474075	...	-0.175248	-0.238352	-0.849203	-0.124786	0.086646	-0.007944	0.360582	0.090110	0.091000	1.00
5 rows × 31 columns

data_test.tail()
id	Time	V1	V2	V3	V4	V5	V6	V7	V8	...	V20	V21	V22	V23	V24	V25	V26	V27	V28	Amount
39995	33211	87180	-0.227015	1.066681	0.167350	0.847756	0.918577	-0.569415	1.412774	-0.317352	...	0.007818	0.115365	0.685676	-0.300523	-0.008611	-0.105765	-0.433028	0.129265	0.010438	13.99
39996	226678	66735	1.080631	0.072792	0.193645	1.189376	-0.379073	-0.941658	0.310233	-0.237765	...	0.009391	0.075056	0.056154	-0.170659	0.388427	0.646854	-0.336122	-0.004275	0.034426	89.16
39997	91144	69391	-0.366346	-0.123654	1.574127	-1.620753	-1.152640	-0.860367	-0.407476	-0.123524	...	-0.106231	-0.167896	-0.034830	-0.125103	0.357849	-0.137647	-0.134737	0.253740	0.180461	10.00
39998	151948	84607	1.245859	-0.208636	-0.083039	-0.792677	-0.621157	-1.219052	0.085329	-0.130222	...	-0.217895	0.033446	0.241614	-0.169362	0.550034	0.853734	-0.576327	0.022538	0.001399	7.71
39999	80174	70403	-0.633590	0.812096	2.561737	0.446063	-0.448034	-0.683693	0.267501	-0.008028	...	0.149068	0.007932	-0.015742	-0.131304	0.736886	-0.036651	0.325312	0.051134	0.087213	6.47
5 rows × 31 columns

data_test['normal_amount'] = StandardScaler().fit_transform(data_test['Amount'].values.reshape(-1,1))
data_test1 = data_test.drop(['Amount','Time','id'], axis=1)
new_pred_class = lr_under_C2.predict(data_test1)
new_pred_class
array([0, 0, 0, ..., 0, 0, 0], dtype=int64)
new_pred_class_prob = lr_balanced.predict_proba(data_test1)[:, 1]
new_pred_class_prob
array([0.04646066, 0.00050197, 0.05586541, ..., 0.01563743, 0.03944736,
       0.06019283])
pd.DataFrame({'ID':data_test.id,'Class':new_pred_class}).set_index('ID').to_csv('submission_credit1.csv')
pd.DataFrame({'ID':data_test.id,'Class':new_pred_class_prob}).set_index('ID').to_csv('submission_credit_prob1.csv')
