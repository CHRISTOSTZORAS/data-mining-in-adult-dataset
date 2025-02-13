##import libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix,ConfusionMatrixDisplay
import time
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.stats import zscore

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
    "workclass": whole_df["workclass"].mode()[0]#private
}
print(most_frequent_values)
#now we will change the nan values with the most frequence values that we found above
whole_df["native-country"].fillna(whole_df["native-country"].mode()[0],inplace=True)
whole_df["workclass"].fillna(whole_df["workclass"].mode()[0],inplace=True)
whole_df.dropna(subset=["occupation"], inplace=True)
nan_counts=whole_df.isnull().sum()

#lets check if we have some extra noise
#LETS CHECK COLUMN AGE
whole_df['age'].unique()

#LETS CHECK COLUMN WORKCLASS
whole_df['workclass'].unique()
whole_df.loc[:, "workclass"] = whole_df["workclass"].replace("?", most_frequent_values["workclass"])

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

#LETS CHECK COLUMN capital-loss
whole_df['capital-loss'].unique()
zero_count = (whole_df['capital-loss'] == 0).sum()

#LETS CHECK COLUMN hours-per-week
whole_df['hours-per-week'].unique()

#LETS CHECK COLUMN native-country
whole_df['native-country'].unique()
whole_df.loc[:, "native-country"] = whole_df["native-country"].replace("?", most_frequent_values["native-country"])

#LETS CHECK COLUMN income
whole_df['income'].unique()
whole_df.loc[:, 'income'] = whole_df['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})


#data normalization.
columns_to_normalize = ["age", "fnlwgt", "education-num", "hours-per-week"]
scaler = MinMaxScaler()
whole_df.loc[:, columns_to_normalize] = scaler.fit_transform(whole_df[columns_to_normalize])

columns_to_log_scale = ["capital-gain", "capital-loss"]
for col in columns_to_log_scale:
    whole_df.loc[:, col] = np.log1p(whole_df[col])


categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country','income']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    whole_df.loc[:, col] = le.fit_transform(whole_df[col]) 
    label_encoders[col] = le  
encoding_mappings = {col: {category: i for i, category in enumerate(label_encoders[col].classes_)} 
                     for col in categorical_columns}


#data reduction.

# make 2 datasets have the same index
feature = ["age", "fnlwgt", "education-num", "hours-per-week",'workclass', 'education','marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country','capital-gain','capital-loss']
x = whole_df.loc[:,feature] # separating features
y = whole_df.loc[:,'income']# separating target
# PCA while keeping 95% of the variance
pca = PCA(n_components=0.95) 
pct = pca.fit_transform(x)
pca_columns = [f'pc{i+1}' for i in range(pct.shape[1])]
principal_df = pd.DataFrame(pct, columns=pca_columns)
principal_df = principal_df.reset_index(drop=True)
y= y.reset_index(drop=True)
finaldf = pd.concat([principal_df, y], axis=1)
print(f"Number of selected principal components: {pca_columns}")
print(finaldf)

###SECOND EXERCISE###
#Future Selection.
#Feature Selection using SelectKBest
feature = ["age", "fnlwgt", "education-num", "hours-per-week",'workclass', 'education','marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country','capital-gain','capital-loss']
features= whole_df.loc[:,feature] 
target = whole_df.loc[:,'income']
selector = SelectKBest(score_func=f_classif, k=5) 
features_new = selector.fit_transform(features, target)
selected_features_kbest = features.columns[selector.get_support()]
selected_features_df = pd.DataFrame({
    "Method": ["SelectKBest"] * 5 ,
    "Feature": selected_features_kbest.tolist() 
})
print(selected_features_df)

#Training Models Full and with 5 Features

features_full = features
features_reduced = features[selected_features_kbest]
target = target.astype('category')
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
    "Model": ["Full Feature ", "Reduced Feature"],
    "Accuracy": [accuracy_full, accuracy_reduced],
    "Training Time (seconds)": [time_full, time_reduced]
})
print(comparison_df)



#Managing Outliers.

#detect outliers with z score
selected_columns = ["age", "capital-gain", "capital-loss", "hours-per-week",'fnlwgt']
z_scores = np.abs(stats.zscore(whole_df[selected_columns]))
threshold = 3
outliers = (z_scores > threshold).sum(axis=0)
outliers_summary_zscore = pd.DataFrame({
    "Column": selected_columns,
    "Number of Outliers": outliers
})
print(outliers_summary_zscore)

#plot out outliers - box plots
def plot_outliers_boxplots(data, selected_columns):
    for column in selected_columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=data[column])
        plt.title(f"Boxplot of {column}")
        plt.show()
plot_outliers_boxplots(whole_df, selected_columns)

# Compute correlation matrix

correlation_matrix = whole_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.show()

#we take out fnlwgt
whole_df = whole_df.drop(['fnlwgt'],axis=1)
print(whole_df)
columns_to_remove_outliers = ["hours-per-week"]
z_scores = np.abs(zscore(whole_df[columns_to_remove_outliers]))
threshold = 3
whole_df = whole_df[(z_scores < threshold).all(axis=1)]

selected_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
z_scores = np.abs(stats.zscore(whole_df[selected_columns]))
threshold = 3
outliers = (z_scores > threshold).sum(axis=0)
outliers_summary_zscore = pd.DataFrame({
    "Column": selected_columns,
    "Number of Outliers": outliers
})
print(outliers_summary_zscore)

###THIRD EXERCISE###
k = 2
X_clustering=whole_df.copy()
#K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_clustering)
silhouette_kmeans = silhouette_score(X_clustering, kmeans_labels)
#Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=k)
agglo_labels = agglo.fit_predict(X_clustering)
silhouette_agglo = silhouette_score(X_clustering, agglo_labels)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Scatter plot  K-Means
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="viridis", ax=axes[0], s=10)
axes[0].set_title(f"K-Means Clustering με k={k}\nSilhouette Score: {silhouette_kmeans:.4f}")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].legend(title="Clusters")
# Scatter plot Agglomerative
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=agglo_labels, palette="magma", ax=axes[1], s=10)
axes[1].set_title(f"Agglomerative Clustering με k={k}\nSilhouette Score: {silhouette_agglo:.4f}")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].legend(title="Clusters")
plt.tight_layout()
plt.show()
silhouette_scores = pd.DataFrame({
    "Algorithm": ["K-Means", "Agglomerative"],
    "Silhouette Score": [silhouette_kmeans, silhouette_agglo]
})
print(silhouette_scores)

#we will try for k=3 and for k=4 to see what is happening
cluster_numbers = [3, 4]
silhouette_scores = {"Algorithm": [], "Clusters": [], "Silhouette Score": []}

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)
fig, axes = plt.subplots(len(cluster_numbers), 2, figsize=(14, 6 * len(cluster_numbers)))

for idx, k in enumerate(cluster_numbers):
    # K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_clustering)
    silhouette_kmeans = silhouette_score(X_clustering, kmeans_labels)  # FIXED

    # Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=k)
    agglo_labels = agglo.fit_predict(X_clustering)
    silhouette_agglo = silhouette_score(X_clustering, agglo_labels)

    # Store silhouette scores
    silhouette_scores["Algorithm"].extend(["K-Means", "Agglomerative"])
    silhouette_scores["Clusters"].extend([k, k])
    silhouette_scores["Silhouette Score"].extend([silhouette_kmeans, silhouette_agglo])

    # Scatter plot for K-Means
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="viridis", ax=axes[idx, 0], s=10)
    axes[idx, 0].set_title(f"K-Means Clustering with k={k}\nSilhouette Score: {silhouette_kmeans:.4f}")
    axes[idx, 0].set_xlabel("Principal Component 1")
    axes[idx, 0].set_ylabel("Principal Component 2")
    axes[idx, 0].legend(title="Clusters")

    # Scatter plot for Agglomerative Clustering
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=agglo_labels, palette="magma", ax=axes[idx, 1], s=10)
    axes[idx, 1].set_title(f"Agglomerative Clustering with k={k}\nSilhouette Score: {silhouette_agglo:.4f}")
    axes[idx, 1].set_xlabel("Principal Component 1")
    axes[idx, 1].set_ylabel("Principal Component 2")
    axes[idx, 1].legend(title="Clusters")

plt.tight_layout()
plt.show()

# Convert silhouette scores to DataFrame
results_df = pd.DataFrame(silhouette_scores)
print(results_df)


###FORTH EXERCISE###

# 1️⃣ Define features and target
new_feature = ["age", "education-num", "hours-per-week", 'workclass', 'education', 
               'marital-status', 'occupation', 'relationship', 'race', 'sex', 
               'native-country', 'capital-gain', 'capital-loss']
X = whole_df.loc[:, new_feature]  
y = whole_df.loc[:, 'income'].astype(int)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}
results = []
for model_name, model in models.items():
    print(f"\n🔹 Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix - {model_name}:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"])
    disp.plot(cmap="Blues", values_format="d") 
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    accuracy = accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["precision"]
    recall = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["recall"]
    f1 = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["f1-score"]
    results.append({"Model": model_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1})
machine_learning_results_df = pd.DataFrame(results)
print("\n📊 Final Results Summary:")
print(machine_learning_results_df)

# making some extra experiments 
gb_versions = {
    "GB_n_estimators_100": GradientBoostingClassifier(n_estimators=100),
    "GB_n_estimators_300": GradientBoostingClassifier(n_estimators=300), 
    "GB_n_estimators_500": GradientBoostingClassifier(n_estimators=500), 

    "GB_max_depth_3": GradientBoostingClassifier(max_depth=3), 
    "GB_max_depth_5": GradientBoostingClassifier(max_depth=5), 
    "GB_max_depth_7": GradientBoostingClassifier(max_depth=7), 

    "GB_learning_rate_0.1": GradientBoostingClassifier(learning_rate=0.1),
    "GB_learning_rate_0.05": GradientBoostingClassifier(learning_rate=0.05), 
    "GB_learning_rate_0.01": GradientBoostingClassifier(learning_rate=0.01)  
}
results = []
for version_name, model in gb_versions.items():
    print(f"\n Training {version_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix - {version_name}:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"])
    disp.plot(cmap="Blues", values_format="d")  
    plt.title(f"Confusion Matrix - {version_name}")
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["precision"]
    recall = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["recall"]
    f1 = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["f1-score"]
    results.append({"Model": version_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1})

machine_learning_results_df = pd.DataFrame(results)
print("\n📊 Final Results Summary:")
print(machine_learning_results_df)

###Fifth EXERCISE###
import tensorflow as tf
new_feature = ["age", "education-num", "hours-per-week", 'workclass', 'education', 
               'marital-status', 'occupation', 'relationship', 'race', 'sex', 
               'native-country', 'capital-gain', 'capital-loss']

X = whole_df.loc[:, new_feature]
y = whole_df.loc[:, 'income'].astype(int)  

# turning our da ta to float 32 type to be compatible with tensor
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42)  
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.75, random_state=42)
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_temp=X_temp.astype(np.float32)
y_temp=y_temp.astype(np.float32)
X_test=X_test.astype(np.float32)
y_test=y_test.astype(np.float32)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.float32)
def create_model(hidden_layer_1, hidden_layer_2):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_layer_1, activation='relu', input_shape=(X_train.shape[1],)),  
        tf.keras.layers.Dense(hidden_layer_2, activation='relu'), 
        tf.keras.layers.Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
hidden_layer_configs = [
    (16, 8),  
    (32, 16), 
    (64, 32)   
]
results = []
for config in hidden_layer_configs:
    print(f"\n🔹 Training MLP with {config[0]}-{config[1]} hidden neurons...")
    model = create_model(*config)
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))
    y_pred = (model.predict(X_test) > 0.5).astype("int32")  
    accuracy = accuracy_score(y_test, y_pred)
    results.append({"Model": f"MLP_{config[0]}-{config[1]}", "Accuracy": accuracy})

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - MLP {config[0]}-{config[1]}")
    plt.legend()
    plt.show()
results_df = pd.DataFrame(results)
print("\n📊 Final Model Comparison:")
print(results_df)

























