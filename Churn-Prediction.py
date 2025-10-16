import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\Banavand\Desktop\File\Telco-Customer-Churn.csv")

print(df.head())

print(df.info())
'''
    RangeIndex: 7032 entries, 0 to 7031
    Data columns (total 21 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   customerID        7032 non-null   object 
     1   gender            7032 non-null   object 
     2   SeniorCitizen     7032 non-null   int64  
     3   Partner           7032 non-null   object 
     4   Dependents        7032 non-null   object 
     5   tenure            7032 non-null   int64  
     6   PhoneService      7032 non-null   object 
     7   MultipleLines     7032 non-null   object 
     8   InternetService   7032 non-null   object 
     9   OnlineSecurity    7032 non-null   object 
     10  OnlineBackup      7032 non-null   object 
     11  DeviceProtection  7032 non-null   object
     12  TechSupport       7032 non-null   object
     13  StreamingTV       7032 non-null   object
     14  StreamingMovies   7032 non-null   object
     15  Contract          7032 non-null   object
     16  PaperlessBilling  7032 non-null   object
     17  PaymentMethod     7032 non-null   object
     18  MonthlyCharges    7032 non-null   float64
     19  TotalCharges      7032 non-null   float64
     20  Churn             7032 non-null   object
    dtypes: float64(2), int64(2), object(17)
    memory usage: 1.1+ MB
    None
'''

print(df.describe())
'''
           SeniorCitizen       tenure  MonthlyCharges  TotalCharges
    count    7032.000000  7032.000000     7032.000000   7032.000000
    mean        0.162400    32.421786       64.798208   2283.300441
    std         0.368844    24.545260       30.085974   2266.771362
    min         0.000000     1.000000       18.250000     18.800000
    25%         0.000000     9.000000       35.587500    401.450000
    50%         0.000000    29.000000       70.350000   1397.475000
    75%         0.000000    55.000000       89.862500   3794.737500
    max         1.000000    72.000000      118.750000   8684.800000
'''

print(df.isna().sum())
'''
    customerID          0
    gender              0
    SeniorCitizen       0
    Partner             0
    Dependents          0
    tenure              0
    PhoneService        0
    MultipleLines       0
    InternetService     0
    OnlineSecurity      0
    OnlineBackup        0
    DeviceProtection    0
    TechSupport         0
    StreamingTV         0
    StreamingMovies     0
    Contract            0
    PaperlessBilling    0
    PaymentMethod       0
    MonthlyCharges      0
    TotalCharges        0
    Churn               0
    dtype: int64
'''

# ----------------------------------------------------------------------------

# Task: Display the distribution of class labels (churn) using Count Plot.
sns.countplot(data=df, x='Churn', hue='Churn')
plt.show()

# Task: Check the distribution of Churn among the TotalCharges task categories using a Violin Plot.
sns.violinplot(data=df, x='Churn', y='TotalCharges', hue='Churn')
plt.show()

# Task: Create a box plot that shows the distribution of TotalCharges for each contract type and also add a color scheme based on the Churn class.
sns.boxplot(data=df, x='Contract', y='TotalCharges', hue='Churn')
plt.show()

# Task: Create a bar chart that shows the correlation of the following features with class labels. Remember that for categorical
# features you must first convert them to DUMMY variables because correlation can only be calculated for numeric features.
corr = pd.get_dummies(df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                      'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']]).corr()
corr_yes = corr['Churn_Yes'].sort_values().iloc[1:-1]
print(corr_yes)
'''
    Contract_Two year                         -0.301552
    InternetService_No                        -0.227578
    DeviceProtection_No internet service      -0.227578
    TechSupport_No internet service           -0.227578
    OnlineSecurity_No internet service        -0.227578
    OnlineBackup_No internet service          -0.227578
    StreamingMovies_No internet service       -0.227578
    PaperlessBilling_No                       -0.191454
    Contract_One year                         -0.178225
    OnlineSecurity_Yes                        -0.171270
    TechSupport_Yes                           -0.164716
    Dependents_Yes                            -0.163128
    Partner_Yes                               -0.149982
    PaymentMethod_Credit card (automatic)     -0.134687
    InternetService_DSL                       -0.124141
    PaymentMethod_Bank transfer (automatic)   -0.118136
    PaymentMethod_Mailed check                -0.090773
    OnlineBackup_Yes                          -0.082307
    DeviceProtection_Yes                      -0.066193
    MultipleLines_No                          -0.032654
    MultipleLines_No phone service            -0.011691
    PhoneService_No                           -0.011691
    gender_Male                               -0.008545
    gender_Female                              0.008545
    PhoneService_Yes                           0.011691
    MultipleLines_Yes                          0.040033
    StreamingMovies_Yes                        0.060860
    StreamingMovies_No                         0.130920
    Partner_No                                 0.149982
    SeniorCitizen                              0.150541
    Dependents_No                              0.163128
    PaperlessBilling_Yes                       0.191454
    DeviceProtection_No                        0.252056
    OnlineBackup_No                            0.267595
    PaymentMethod_Electronic check             0.301455
    InternetService_Fiber optic                0.307463
    TechSupport_No                             0.336877
    OnlineSecurity_No                          0.342235
    Contract_Month-to-month                    0.404565
    Name: Churn_Yes, dtype: float64
'''
sns.barplot(x=corr_yes.index, y=corr_yes.values, palette='viridis')
plt.title("Feature correlation to yes churn")
plt.xticks(rotation=90)
plt.show()

# ----------------------------------------------------------------------------

print(f"array({df['Contract'].unique()}, dtype={df['Contract'].dtype})")
'''
    array(['Month-to-month' 'One year' 'Two year'], dtype=object)
'''

# Task: Create a histogram  to display the distribution of the tenure column, which is the number of months a customer has been or is a customer.
print(df['tenure'].unique())
sns.histplot(data=df, x='tenure', bins=30)
plt.yticks(range(0, 801, 100))
plt.show()

# Task: Now use the SEABORN library to create separate histograms based on two additional features Churn and Contract.
sns.displot(data=df, x="tenure", bins=70, col='Contract', row='Churn')
plt.show()
'''
    grid = sns.FacetGrid(data=df, col='Contract', row='Churn', margin_titles=True)
    grid.map(sns.histplot, 'tenure', bins=30)
    grid.set_axis_labels("tenure", "Count")
    plt.subplots_adjust(top=0.9)
    grid.fig.suptitle('Distribution of Tenure by Churn and Contract Type')
    plt.show()
'''

# Task: Create a scatter chart  of TotalCharges vs. Monthly Charges and set the color of the dots based on Churn.
plt.figure(figsize=(10, 8), dpi=150)
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Churn', alpha=0.6)
plt.show()


# Task: Now that you have the churn rate for each tenure group from 1 to 72 months, create a chart to show the churn rate for each month of the membership period.
no_churn = df.groupby(["Churn", "tenure"]).count().transpose()['No']
yes_churn = df.groupby(["Churn", "tenure"]).count().transpose()['Yes']

churn_rate = 100 * yes_churn / (yes_churn + no_churn)
d = churn_rate.transpose()["customerID"]
d = pd.DataFrame(data=d)
print(d)
'''
    tenure   customerID

    1        61.990212
    2        51.680672
    3        47.000000
    4        47.159091
    5        48.120301
    ...            ...
    68        9.000000
    69        8.421053
    70        9.243697
    71        3.529412
    72        1.657459

    [72 rows x 1 columns]
'''
plt.figure(figsize=(10, 8), dpi=150)
sns.lineplot(data=d, x='tenure', y='customerID')
plt.show()


# Task: Create a new column called Tenure Cohort based on the values ​​of the tenure column, which creates 4 separate cohorts.
def c(i):
    if i < 13:
        return '0-12 Months'
    elif i < 25:
        return '12-24 Months'
    elif i < 48:
        return '24-48 Months'
    else:
        return 'Over 48 Months'

df['total tenure'] = df['tenure'].apply(c)


# Task: Create a scatter plot of TotalCharges vs. Monthly Charges colored based on the Tenure Cohort defined in the previous task.
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='total tenure', alpha=0.5)
plt.show()


# Task: Create a bar chart  to display churn by tenure cohort.
sns.countplot(data=df, x='total tenure', hue='Churn')
plt.show()


# Task: Create a grid of bar charts that display totals by Tenure Cohort, separated by Contract type, and color-coded by Chum.
sns.catplot(data=df, x='total tenure', kind='count', col='Contract', hue='Churn')
plt.show()


# ----------------------------------------------------------------------------

# Split the data task into features X and labels Y. Create dummy variables if necessary and note which features are not useful and should be removed.
# Then do a test training split so that 10% of the data is kept for testing. We will use random state equal to 101 in the Solution Video Notebook.
X = pd.get_dummies(df.drop(['Churn', 'customerID'], axis=1), drop_first=True)
Y = df['Churn'].map({'Yes': 1, 'No': 0})

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=101, test_size=0.2)


# Complete the following tasks:
# 1: Train a single decision tree model. You can use grid search to find optimal hyperparameters.
# 2: Evaluate performance metrics from the decision tree, including classification reporting and drawing an error matrix.
# 3: Calculate the importance of features from the decision tree.
# 4: Draw your tree.

# 1:
de_model = DecisionTreeClassifier(random_state=101)

de_param_grid = {
    'criterion': ["gini", "entropy", "log_loss"],
    'max_depth': [3, 4, 5]
}
de_grid = GridSearchCV(estimator=de_model, param_grid=de_param_grid, cv=3, verbose=1)
de_grid.fit(x_train, y_train)

# 2:
print(f"classification_report for DecisionTree:\n{classification_report(y_test, de_grid.predict(x_test))}")
print(f"best params: {de_grid.best_params_}")
print(f"best params: {de_grid.best_params_}")

ConfusionMatrixDisplay.from_predictions(y_test, de_grid.predict(x_test))
plt.show()

# 3:
fi = de_grid.best_estimator_.feature_importances_
fi = pd.DataFrame(data=fi, index=X.columns, columns=['feature_importances']).sort_values('feature_importances', ascending=True)
fi = fi[fi['feature_importances'] > 0]

sns.barplot(data=fi, x=fi.index, y='feature_importances', palette='Reds', hue='feature_importances')
plt.xticks(rotation = 90)
plt.show()

# 4:
plt.figure(figsize=(10, 8))
plot_tree(de_grid.best_estimator_, filled=True, feature_names=X.columns)
plt.show()


# Task: Create a random forest model and generate a classification report and error matrix of its predicted results on the test set.
ra_model = RandomForestClassifier(random_state=101)
ra_model.fit(x_train, y_train)

print(f"classification_report for RandomForest:\n{classification_report(y_test, ra_model.predict(x_test))}")

ConfusionMatrixDisplay.from_predictions(y_test, ra_model.predict(x_test))
plt.show()


# Task: Use 1: AdaBoost and 2: Gradient Boosting to create a model and return the classification report and plot an error matrix for its predicted results.
# 1) AdaBoost:
ad_model = AdaBoostClassifier(random_state=101)
ad_model.fit(x_train, y_train)

print(f"classification_report for AdaBoost:\n{classification_report(y_test, ad_model.predict(x_test))}")
fiw = pd.DataFrame(data=ad_model.feature_importances_, index=X.columns, columns=['feature_importances']).sort_values('feature_importances', ascending=True)
print(fiw)
ConfusionMatrixDisplay.from_predictions(y_test, ad_model.predict(x_test))
plt.show()

# 2) Gradient:
gr_model = GradientBoostingClassifier(random_state=101)
gr_model.fit(x_train, y_train)

print(f"classification_report for GradientBoosting:\n{classification_report(y_test, gr_model.predict(x_test))}")

ConfusionMatrixDisplay.from_predictions(y_test, gr_model.predict(x_test))
plt.show()


# Task: Use Logistic to create a model and return the classification report and plot an error matrix for its predicted results.
lo_model = LogisticRegression(random_state=101)
lo_model.fit(x_train, y_train)

print(f"classification_report for LogisticRegression:\n{classification_report(y_test, lo_model.predict(x_test))}")

ConfusionMatrixDisplay.from_predictions(y_test, lo_model.predict(x_test))
plt.show()


# Task: Use SVM to create a model and return the classification report and plot an error matrix for its predicted results.
svc_model = SVC(random_state=101)
svc_model.fit(x_train, y_train)

print(f"classification_report for SVC:\n{classification_report(y_test, svc_model.predict(x_test))}")

ConfusionMatrixDisplay.from_predictions(y_test, svc_model.predict(x_test))
plt.show()


# Results:
results = {
    "Model": ["Decision Tree", "Random Forest", "AdaBoost", "GradientBoosting", "Logistic", "SVM"],
    "Accuracy": [
        de_grid.score(x_test, y_test),
        ra_model.score(x_test, y_test),
        ad_model.score(x_test, y_test),
        gr_model.score(x_test, y_test),
        lo_model.score(x_test, y_test),
        svc_model.score(x_test, y_test)
    ]
}
df_results = pd.DataFrame(results)
sns.barplot(data=df_results, x="Model", y="Accuracy", palette="crest")
plt.xticks(rotation=45)
plt.show()


'''
    The AdaBoost model performed best with an accuracy of 81%, and the biggest impact was on the Contract and tenure features.
'''
