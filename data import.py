##import libraries and data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

adult = fetch_ucirepo(id=2) 
# data importing
other_variables = adult.data.features 
income = adult.data.targets 
whole_df=pd.concat([other_variables,income],axis=1)

###FIRST EXERCISE###
#data cleaning.

#we will found possible nan values
nan_counts=whole_df.isnull().sum()
nan_counts
#we find the most frecunece value in each  column
most_frequent_values = {
    "native-country": whole_df["native-country"].mode()[0],#unites-states
    "occupation": whole_df["occupation"].mode()[0],#prof-specialty
    "workclass": whole_df["workclass"].mode()[0]#private
}
print(most_frequent_values)
#now we will change the nan values with the most frequence values that we found above
whole_df["native-country"].fillna(whole_df["native-country"].mode()[0],inplace=True)
whole_df["workclass"].fillna(whole_df["workclass"].mode()[0],inplace=True)
#in occupation column we will drop nas beacuse values are even shown
whole_df.dropna(subset=["occupation"], inplace=True)

nan_counts=whole_df.isnull().sum()
nan_counts

#lets check if we have some extra noise
#LETS CHECK COLUMN AGE
whole_df['age'].unique()

#LETS CHECK COLUMN WORKCLASS
whole_df['workclass'].unique()
#we notice the question mark
whole_df["workclass"] = whole_df["workclass"].replace("?", most_frequent_values["workclass"])
#LETS CHECK COLUMN fnlwgt
sorted(whole_df['fnlwgt'].unique())

#LETS CHECK COLUMN education
whole_df['education'].unique()

#LETS CHECK COLUMN education-num
sorted(whole_df['education-num'].unique())

#LETS CHECK COLUMN marital-status
whole_df['marital-status'].unique()

#LETS CHECK COLUMN occupation
whole_df['occupation'].unique()
#we notice the question mark
whole_df = whole_df[whole_df["occupation"] != "?"]

#LETS CHECK COLUMN relationship
whole_df['relationship'].unique()

#LETS CHECK COLUMN race
whole_df['race'].unique()

#LETS CHECK COLUMN sex
whole_df['sex'].unique()

#LETS CHECK COLUMN capital-gain
whole_df['capital-gain'].unique()
zero_count = (whole_df['capital-gain'] == 99999).sum()
zero_count

#LETS CHECK COLUMN capital-loss
whole_df['capital-loss'].unique()
zero_count = (whole_df['capital-loss'] == 0).sum()
zero_count

#LETS CHECK COLUMN hours-per-week
whole_df['hours-per-week'].unique()

#LETS CHECK COLUMN native-country
whole_df['native-country'].unique()
#we notice the question mark
whole_df["native-country"] = whole_df["native-country"].replace("?", most_frequent_values["native-country"])

#LETS CHECK COLUMN income
whole_df['income'].unique()
#we notice 2 diff types of the same value >50k &>50k. and <=50k &<=50k.
whole_df['income'] = whole_df['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})


#data normalization.

# We will attempt to normalize certain variables to make them more useful for the machine learning model we will use later.
# The variables being normalized are quantitative, so we focus on the numerical ones. 
# We will definitely normalize  age,weight, education number, and hours per week.
# However, we will  normalizing capital gain and capital loss with log scaling because we observed that they contain many zero values and many outliers.
# Using Min-Max Scaling in this case would create a numerical variable centered around zero with several extreme values.
columns_to_normalize = ["age","fnlwgt", "education-num", "hours-per-week"]
scaler = MinMaxScaler()
whole_df[columns_to_normalize] = scaler.fit_transform(whole_df[columns_to_normalize])

columns_to_log_scale = ["capital-gain", "capital-loss",]
for col in columns_to_log_scale:
    whole_df[col] = np.log1p(whole_df[col])


#now we will also use label encoding to our nominal variables 
categorical_columns=['workclass','education','marital-status','occupation','relationship',
                     'race','sex','native-country','income']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    whole_df[col] = le.fit_transform(whole_df[col]) 
    label_encoders[col] = le  
    
#making a df of encodings to help us in future cases
encoding_mappings = {}
for col in categorical_columns:
    encoding_mappings[col] = {category: i for i, category in enumerate(label_encoders[col].classes_)}
encoding_df = pd.DataFrame.from_dict(encoding_mappings, orient="index").transpose()


#data reduction.

#we will use a copy of our original data to use data reduction,because our data are
#already reducted and in future cases we will have to use our raw data so we dont have to
#change them
# This code snippet is performing the following tasks:
whole_df_copy=whole_df.copy()
feature = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
           'fnlwgt', 'education-num', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
           'native-country']
x = whole_df_copy.loc[:, feature]  # Separating features
y = whole_df_copy.loc[:, 'income']  # Separating target
# Apply PCA while keeping 95% of the variance
pca = PCA(n_components=0.95)  # Automatically selects the number of components
pct = pca.fit_transform(x)
pct.shape[0]
# Get the number of selected principal components
num_components = pca.n_components_
# Create a DataFrame with the principal components
pca_columns = [f'pc{i+1}' for i in range(num_components)]
principal_df = pd.DataFrame(pct, columns=pca_columns)
principal_df = principal_df.reset_index(drop=True)
income_df = whole_df_copy[['income']].reset_index(drop=True)
finaldf = pd.concat([principal_df, income_df], axis=1)
# Print the transformed dataset
print(f"Number of selected principal components: {num_components}")
print(finaldf)
#in order to keep 0.95 of our variance only 6 feature stay


###SECOND EXERCISE###

#Future Selection.

#Feature Selection using SelectKBest
features = whole_df.drop(columns=["income"]) 
target= whole_df["income"] 
selector = SelectKBest(score_func=f_classif, k=5) 
features_new = selector.fit_transform(X, y)
selected_features_kbest = features.columns[selector.get_support()]
selected_features_df = pd.DataFrame({
    "Method": ["SelectKBest"] * 5 ,
    "Feature": selected_features_kbest.tolist() 
})
print(selected_features_df)

#Training Models Full and with 5 Features

features_full = whole_df.drop(columns=["income"])
features_reduced = whole_df[selected_features_kbest]
target= whole_df["income"]

X_train_full, X_test_full, y_train, y_test = train_test_split(features_full, target, test_size=0.2, random_state=42)
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(features_reduced, target, test_size=0.2, random_state=42)

# Training Decision Tree  Full
start_time = time.time()
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train_full, y_train)
y_pred_full = dt_full.predict(X_test_full)
accuracy_full = accuracy_score(y_test, y_pred_full)
time_full = time.time() - start_time

# Decision Tree 5 Variables
start_time = time.time()
dt_reduced = DecisionTreeClassifier(random_state=42)
dt_reduced.fit(X_train_reduced, y_train_reduced)
y_pred_reduced = dt_reduced.predict(X_test_reduced)
accuracy_reduced = accuracy_score(y_test_reduced, y_pred_reduced)
time_reduced = time.time() - start_time

# Print Results
comparison_df = pd.DataFrame({
    "Model": ["Full Feature Set", "Reduced Feature Set (Top 5)"],
    "Accuracy": [accuracy_full, accuracy_reduced],
    "Training Time (seconds)": [time_full, time_reduced]
})
print(comparison_df)
#we notice that dont only with 5 variables we get a better accuracy index but its also a lot more quicker


#Managing Outliers.

#detect outliers
selected_columns = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
def detect_outliers(data, selected_columns):
    outliers_dict = {}
    for column in selected_columns:
        if column in data.select_dtypes(include=np.number).columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 4 * IQR
            upper_bound = Q3 + 4 * IQR
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            outliers_dict[column] = len(outliers)  
    return pd.DataFrame(list(outliers_dict.items()), columns=["Column", "Number of Outliers"])
outliers_summary = detect_outliers(whole_df, selected_columns)
print(outliers_summary)

#plot out outliers
def plot_outliers_boxplots(data, selected_columns):
    for column in selected_columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=data[column])
        plt.title(f"Boxplot of {column} (Detecting Outliers)")
        plt.show()
plot_outliers_boxplots(whole_df, selected_columns)

#manage outliers
#checking our outliers we see that we have 19787 in 46033 ,42% of our data.Removing them would be a bad idea for our future model
#So we must find a way not to remove them but edit them .
#AFTER LOG AND CAPPING AND SOM OTHER METHOD I SEE NO RESULTS TO MY OULLIERS
#As a solution i will make my bounders really flexible only to take out the most extreme values.I do that by multipyling 
#iqr by 3.5.I see with way that age ouliers disappeared ,fnlwght reduced by far,hour  are 5.351 and capital-gain and capital-loss stay the same.
#i SEE THAT I CAN NOT USE capital gai and capital loss columns any more after i already made the log transform so i will throw them from my data.
#5.500 thousands outliers remaing ,almost 7 percent of our data.But we now know that they are extremely outliers so we will remove them
print(whole_df[["capital-gain", "capital-loss", "income"]].corr())


# next move its to drom capital-gain,capital-loss and remove outliers