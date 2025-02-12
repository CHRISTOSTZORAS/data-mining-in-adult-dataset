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
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
import time
from scipy import stats
from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
adult = fetch_ucirepo(id=2) 
# data importing
other_variables = adult.data.features 
income = adult.data.targets 
whole_df=pd.concat([other_variables,income],axis=1)

###FIRST EXERCISE###
#data cleaning.

#we will found possible nan values
nan_counts=other_variables.isnull().sum()
nan_counts
#we find the most frecunece value in each  column
most_frequent_values = {
    "native-country": other_variables["native-country"].mode()[0],#unites-states
    "workclass": other_variables["workclass"].mode()[0]#private
}
print(most_frequent_values)
#now we will change the nan values with the most frequence values that we found above
other_variables["native-country"].fillna(other_variables["native-country"].mode()[0],inplace=True)
other_variables["workclass"].fillna(other_variables["workclass"].mode()[0],inplace=True)
other_variables.dropna(subset=["occupation"], inplace=True)
nan_counts=other_variables.isnull().sum()

#lets check if we have some extra noise
#LETS CHECK COLUMN AGE
other_variables['age'].unique()

#LETS CHECK COLUMN WORKCLASS
other_variables['workclass'].unique()
other_variables.loc[:, "workclass"] = other_variables["workclass"].replace("?", most_frequent_values["workclass"])

#LETS CHECK COLUMN fnlwgt
sorted(other_variables['fnlwgt'].unique())

#LETS CHECK COLUMN education
other_variables['education'].unique()

#LETS CHECK COLUMN education-num
sorted(other_variables['education-num'].unique())

#LETS CHECK COLUMN marital-status
other_variables['marital-status'].unique()

#LETS CHECK COLUMN occupation
other_variables['occupation'].unique()
other_variables = other_variables[other_variables["occupation"] != "?"]

#LETS CHECK COLUMN relationship
other_variables['relationship'].unique()

#LETS CHECK COLUMN race
other_variables['race'].unique()

#LETS CHECK COLUMN sex
other_variables['sex'].unique()

#LETS CHECK COLUMN capital-gain
other_variables['capital-gain'].unique()
zero_count = (other_variables['capital-gain'] == 99999).sum()

#LETS CHECK COLUMN capital-loss
other_variables['capital-loss'].unique()
zero_count = (other_variables['capital-loss'] == 0).sum()

#LETS CHECK COLUMN hours-per-week
other_variables['hours-per-week'].unique()

#LETS CHECK COLUMN native-country
other_variables['native-country'].unique()
other_variables.loc[:, "native-country"] = other_variables["native-country"].replace("?", most_frequent_values["native-country"])

#LETS CHECK COLUMN income
income['income'].unique()
income.loc[:, 'income'] = income['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})


#data normalization.
columns_to_normalize = ["age", "fnlwgt", "education-num", "hours-per-week"]
scaler = MinMaxScaler()
other_variables.loc[:, columns_to_normalize] = scaler.fit_transform(other_variables[columns_to_normalize])

columns_to_log_scale = ["capital-gain", "capital-loss"]
for col in columns_to_log_scale:
    other_variables.loc[:, col] = np.log1p(other_variables[col])


categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                       'relationship', 'race', 'sex', 'native-country']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    other_variables.loc[:, col] = le.fit_transform(other_variables[col]) 
    label_encoders[col] = le  
encoding_mappings = {col: {category: i for i, category in enumerate(label_encoders[col].classes_)} 
                     for col in categorical_columns}

encoding_df = pd.DataFrame.from_dict(encoding_mappings, orient="index").transpose()
income_encoder = LabelEncoder()
income_encoded = income_encoder.fit_transform(income.values.ravel()) 
income = pd.DataFrame(income_encoded, columns=['income']) 
label_encoders['income'] = income_encoder

#data reduction.

# make 2 datasets have the same index
other_variables = other_variables.reset_index(drop=True)
income = income.reset_index(drop=True)
common_index = other_variables.index.intersection(income.index)
# keep common rows
other_variables = other_variables.loc[common_index].reset_index(drop=True)
income = income.loc[common_index].reset_index(drop=True)
# PCA while keeping 95% of the variance
pca = PCA(n_components=0.95) 
pct = pca.fit_transform(other_variables)
num_components = pca.n_components_
pca_columns = [f'pc{i+1}' for i in range(num_components)]
principal_df = pd.DataFrame(pct, columns=pca_columns)
finaldf = pd.concat([principal_df, income], axis=1)
print(f"Number of selected principal components: {num_components}")
print(finaldf)

###SECOND EXERCISE###
#Future Selection.
#Feature Selection using SelectKBest
features = other_variables
target = income.values.ravel()
selector = SelectKBest(score_func=f_classif, k=5) 
features_new = selector.fit_transform(features, target)
selected_features_kbest = features.columns[selector.get_support()]
selected_features_df = pd.DataFrame({
    "Method": ["SelectKBest"] * 5 ,
    "Feature": selected_features_kbest.tolist() 
})
print(selected_features_df)

#Training Models Full and with 5 Features

features_full = other_variables
features_reduced = other_variables[selected_features_kbest]
target= income
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
selected_columns = ["age", "capital-gain", "capital-loss", "hours-per-week","fnlwgt"]
z_scores = np.abs(stats.zscore(whole_df[selected_columns]))
threshold = 3.5
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
plot_outliers_boxplots(other_variables, selected_columns)

#correlation matrix plot
correlation_matrix = whole_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.show()


#we take out fnlwgt
other_variables = other_variables.drop(columns=["fnlwgt"])
#for age and hours per week we will cap extreme values at the 99th precentile 
for col in ["age", "hours-per-week"]:
    upper_limit = other_variables[col].quantile(0.99)
    other_variables[col] = np.where(other_variables[col] > upper_limit, upper_limit, other_variables[col]) 
# for capital gain and capital loss ,we already have logarithm used we will cap them too in 5 and 99 percentile
for col in ["capital-gain", "capital-loss"]:
    lower_limit = other_variables[col].quantile(0.05) 
    upper_limit = other_variables[col].quantile(0.95) 
    other_variables[col] = np.clip(other_variables[col], lower_limit, upper_limit) 
    
selected_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
z_scores = np.abs(stats.zscore(other_variables[selected_columns]))
threshold = 3.5
outliers = (z_scores > threshold).sum(axis=0)
outliers_summary_zscore = pd.DataFrame({
    "Column": selected_columns,
    "Number of Outliers": outliers
})
print(outliers_summary_zscore)

###THIRD EXERCISE###
k = 2
X_clustering=other_variables.copy()
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
silhouette_scores3_4 = {"Algorithm": [], "Clusters": [], "Silhouette Score": []}
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)
fig, axes = plt.subplots(len(cluster_numbers), 2, figsize=(14, 6 * len(cluster_numbers)))

for idx, k in enumerate(cluster_numbers):
    #K-Means
    kmeans3_4 = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels3_4 = kmeans3_4.fit_predict(X_clustering)
    silhouette_kmeans3_4 = silhouette_scores3_4(X_clustering, kmeans_labels3_4)

    #Agglomerative Clustering
    agglo = AgglomerativeClustering(n_clusters=k)
    agglo_labels = agglo.fit_predict(X_clustering)
    silhouette_agglo = silhouette_score(X_clustering, agglo_labels)
    silhouette_scores["Algorithm"].extend(["K-Means", "Agglomerative"])
    silhouette_scores["Clusters"].extend([k, k])
    silhouette_scores["Silhouette Score"].extend([silhouette_kmeans, silhouette_agglo])

    #Scatter plot  K-Means
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette="viridis", ax=axes[idx, 0], s=10)
    axes[idx, 0].set_title(f"K-Means Clustering με k={k}\nSilhouette Score: {silhouette_kmeans:.4f}")
    axes[idx, 0].set_xlabel("Principal Component 1")
    axes[idx, 0].set_ylabel("Principal Component 2")
    axes[idx, 0].legend(title="Clusters")

    #Scatter plot Agglomerative Clustering
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=agglo_labels, palette="magma", ax=axes[idx, 1], s=10)
    axes[idx, 1].set_title(f"Agglomerative Clustering με k={k}\nSilhouette Score: {silhouette_agglo:.4f}")
    axes[idx, 1].set_xlabel("Principal Component 1")
    axes[idx, 1].set_ylabel("Principal Component 2")
    axes[idx, 1].legend(title="Clusters")

plt.tight_layout()
plt.show()

results_df = pd.DataFrame(silhouette_scores)
print(results_df)
###FORTH EXERCISE###
X = other_variables.copy()  # Features
y = income.values.ravel() # Target variable
pd.Series(y).value_counts()
# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    "Support Vector Machine": SVC(class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []

for model_name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluate results
    accuracy = accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["precision"]
    recall = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["recall"]
    f1 = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["f1-score"]
    
    # Store results
    results.append({"Model": model_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1})
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# 3️⃣ Display Results
machine_learning_results_df = pd.DataFrame(results)
print(machine_learning_results_df)

# Οι confusion matrices δείχνουν την απόδοση κάθε μοντέλου σε δύο κατηγορίες:

# <=50K (Χαμηλό Εισόδημα)
# >50K (Υψηλό Εισόδημα)
# 🔹 Logistic Regression
# True Positives (TP) = 2934
# False Negatives (FN) = 0 (άρα δεν προβλέπει σχεδόν καθόλου σωστά το >50K)
# False Positives (FP) = 2934 (πολλά λάθος predictions για <=50K)
# Accuracy = 54.2% (χαμηλό)
# 📌 Συμπέρασμα: Το Logistic Regression δεν αποδίδει καλά, καθώς μπερδεύει αρκετά τις κατηγορίες και έχει χαμηλό Accuracy.

# 🔹 Random Forest
# True Positives (TP) = 809 (λίγες σωστές προβλέψεις >50K)
# False Negatives (FN) = 0 (ακόμα σημαντικό πρόβλημα)
# Accuracy = 69.7% (μέτριο)
# 📌 Συμπέρασμα: Το Random Forest είναι σημαντικά καλύτερο από το Logistic Regression, αλλά δυσκολεύεται ακόμα να προβλέψει τα άτομα με υψηλό εισόδημα.

# 🔹 Support Vector Machine (SVM)
# True Positives (TP) = 1693 (καλύτερη απόδοση στο >50K)
# False Negatives (FN) = 0 (καμία σωστή πρόβλεψη για >50K)
# Accuracy = 63.5% (μέτριο)
# 📌 Συμπέρασμα: Το SVM αποδίδει λίγο καλύτερα από το Logistic Regression, αλλά δεν διαχωρίζει καλά τις δύο κατηγορίες.

# 🔹 Gradient Boosting
# True Positives (TP) = 2 (ουσιαστικά αποτυγχάνει να προβλέψει την κατηγορία >50K)
# False Negatives (FN) = 0
# Accuracy = 75.7% (υψηλό)
# 📌 Συμπέρασμα: Παρόλο που έχει το υψηλότερο Accuracy, δεν προβλέπει σχεδόν καθόλου σωστά την κατηγορία >50K, που είναι μεγάλο πρόβλημα


from imblearn.over_sampling import SMOTE
import sklearn
import imblearn

# Εφαρμογή SMOTE για την εξισορρόπηση των κλάσεων
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Επιβεβαίωση της νέας κατανομής των κλάσεων
unique, counts = np.unique(y_resampled, return_counts=True)
balanced_class_distribution = dict(zip(unique, counts))

# Επανεκπαίδευση των μοντέλων με τα εξισορροπημένα δεδομένα
new_results = []

for model_name, model in models.items():
    # Εκπαίδευση με τα νέα δεδομένα
    model.fit(X_resampled, y_resampled)
    
    # Πρόβλεψη στο test set
    y_pred_smote = model.predict(X_test)
    
    # Αξιολόγηση αποτελεσμάτων
    accuracy = accuracy_score(y_test, y_pred_smote)
    precision = classification_report(y_test, y_pred_smote, output_dict=True)["weighted avg"]["precision"]
    recall = classification_report(y_test, y_pred_smote, output_dict=True)["weighted avg"]["recall"]
    f1 = classification_report(y_test, y_pred_smote, output_dict=True)["weighted avg"]["f1-score"]
    
    # Αποθήκευση αποτελεσμάτων
    new_results.append({"Model": model_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1})
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_smote)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name} (SMOTE Applied)")
    plt.show()

# Προβολή αποτελεσμάτων μετά την εφαρμογή SMOTE
new_machine_learing_results_df = pd.DataFrame(new_results)
print(new_machine_learing_results_df)

# 1️⃣ Logistic Regression (Παλινδρόμηση)
# 📊 Confusion Matrix

# 3721 σωστές προβλέψεις για <=50K (True Negatives)
# 3252 λανθασμένες προβλέψεις όπου <=50K προβλέφθηκαν ως >50K
# 0 σωστές προβλέψεις για >50K (False Negatives)
# Η ακρίβεια έπεσε στο 51.9%
# 🔍 Ανάλυση

# Ο ταξινομητής κάνει πάρα πολλά λάθη στην κατηγορία >50K.
# Πιθανότατα δεν μπορεί να προσαρμοστεί σωστά στα νέα δεδομένα που δημιούργησε το SMOTE.
# Γενικά, η Logistic Regression δεν είναι τόσο ισχυρός ταξινομητής για περίπλοκα προβλήματα.
# 🛠 Τι μπορούμε να κάνουμε;

# Δοκιμή διαφορετικών κανονικοποιήσεων ή διαγραφή περιττών χαρακτηριστικών.
# 2️⃣ Random Forest
# 📊 Confusion Matrix

# 5542 σωστές προβλέψεις για <=50K
# 1431 λανθασμένες προβλέψεις για <=50K
# Η ακρίβεια βελτιώθηκε στο 65.4% (συγκριτικά με πριν)
# 🔍 Ανάλυση

# Το Random Forest ανταποκρίθηκε πολύ καλύτερα από τη Logistic Regression.
# Έχει περισσότερη ισορροπία στις προβλέψεις, αλλά ακόμα υπάρχουν λάθη.
# 🛠 Τι μπορούμε να κάνουμε;

# Αύξηση του αριθμού των δέντρων (n_estimators).
# Δοκιμή διαφορετικών υπερπαραμέτρων (max_depth, min_samples_split, κτλ.).
# 3️⃣ Support Vector Machine
# 📊 Confusion Matrix

# 4227 σωστές προβλέψεις για <=50K
# 2746 λανθασμένες προβλέψεις για <=50K
# Ακρίβεια στο 55.2%
# 🔍 Ανάλυση

# Ακόμα πολλά λάθη και στις δύο κατηγορίες.
# Ο αλγόριθμος δεν ανταποκρίθηκε καλά στο SMOTE.
# 🛠 Τι μπορούμε να κάνουμε;

# Δοκιμή διαφορετικών kernels (RBF, polynomial).
# Ρύθμιση της παραμέτρου C.
# 4️⃣ Gradient Boosting
# 📊 Confusion Matrix

# 6362 σωστές προβλέψεις για <=50K
# 611 λανθασμένες προβλέψεις για <=50K
# Η καλύτερη ακρίβεια στο 71.1%!
# 🔍 Ανάλυση

# Το Gradient Boosting αποδίδει το καλύτερο αποτέλεσμα.
# Έχει την υψηλότερη ακρίβεια, recall και F1-score.
# 🛠 Τι μπορούμε να κάνουμε;

# Αύξηση των n_estimators.
# Δοκιμή Learning Rate (μείωση για καλύτερη γενίκευση)
####βλεςπουμε οτι ουτε με την εξσισοροπηση δεδομενων βγαζουμε καρη οποτε παμε να δοκιμασουμε μονο τις 5 μεταβλητες απο το select k best
selected_features_kbest = ["education-num", "marital-status", "race", "capital-gain", "native-country"]
X_selected = other_variables[selected_features_kbest]
X_train, X_test, y_train, y_test = train_test_split(X_selected, income.values.ravel(), test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Support Vector Machine": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []

for model_name, model in models.items():
    # Εκπαίδευση με τα νέα δεδομένα
    model.fit(X_train, y_train)
    
    # Πρόβλεψη στο test set
    y_pred = model.predict(X_test)
    
    # Αξιολόγηση αποτελεσμάτων
    accuracy = accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["precision"]
    recall = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["recall"]
    f1 = classification_report(y_test, y_pred, output_dict=True)["weighted avg"]["f1-score"]
    
    # Αποθήκευση αποτελεσμάτων
    results.append({"Model": model_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1})
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name} (5best)")
    plt.show()

# Προβολή των αποτελεσμάτων
results_df_5best = pd.DataFrame(results)
print(results_df_5best)
#oute edw vriskoume kapoio montelo pou na einai deleastiko
# 1️⃣ Feature Selection with SelectKBest
# You used SelectKBest to reduce the number of features to the top 5 most relevant based on their correlation with the target.
# The models trained on these selected features show an accuracy comparable to those trained on the full dataset.
# However, the confusion matrices indicate that some models perform extremely poorly in predicting the ">50K" class.
# 2️⃣ Confusion Matrices (5 Best Features)
# Logistic Regression, SVM: These models completely fail to predict the ">50K" class (no values in the true positive quadrant).
# Random Forest & Gradient Boosting: Show minor improvements in detecting the ">50K" class, but the number of false negatives remains high.
# 🔴 Interpretation:

# The drastic class imbalance is likely causing these models to be biased towards predicting only "<=50K".
# The removal of important features might have stripped away crucial information needed to separate the two income classes effectively.
#  Observations:

# Accuracy is not significantly affected by the feature reduction.
# Gradient Boosting shows the highest Precision (0.677), indicating it is more confident in its positive predictions.
# Random Forest shows similar performance to the full-feature model, meaning it still retains good predictive power.

#συνολικο συμπερασμα 
# Με βάση τα αποτελέσματα των τριών διαφορετικών προσεγγίσεων (αρχικά δεδομένα, εξισορροπημένα με SMOTE, και χρήση των 5 καλύτερων χαρακτηριστικών), η καλύτερη επιλογή εξαρτάται από την προτεραιότητά σου.

# Αν δίνουμε έμφαση στην ακρίβεια (Accuracy):

# Gradient Boosting στις αρχικές συνθήκες και με τα 5 καλύτερα χαρακτηριστικά έχει την υψηλότερη ακρίβεια (0.757).
# Random Forest επίσης έχει καλή ακρίβεια, αλλά χαμηλότερη από το Gradient Boosting.
# Αν δίνουμε έμφαση στην ανάκληση (Recall) για την ανίχνευση των υψηλών εισοδημάτων (>50K):

# Μετά την εφαρμογή του SMOTE, το Logistic Regression και το Random Forest έχουν καλύτερη ισορροπία recall, αλλά μειωμένη ακρίβεια.
# Αν δίνουμε έμφαση στην ισορροπία μεταξύ ακρίβειας και ανάκλησης (F1-score):

# Gradient Boosting διατηρεί την υψηλότερη F1-score στις περισσότερες περιπτώσεις.
# Τελική επιλογή:
# Gradient Boosting χωρίς SMOTE και χωρίς τη μείωση χαρακτηριστικών (με όλα τα features).

# Έχει την καλύτερη συνολική απόδοση, με υψηλή ακρίβεια, recall, precision και F1-score.
# Δεν χάνει πληροφορία από το dataset.
# Αν όμως δίνεις έμφαση στο να μειώσεις τη διάσταση των δεδομένων και να βελτιώσεις την ταχύτητα, τότε η επιλογή του Random Forest με τα 5 καλύτερα χαρακτηριστικά είναι καλή εναλλακτική, καθώς κρατάει αξιοπρεπή ακρίβεια χωρίς περιττά χαρακτηριστικά.
#παμε τωρα με το gradient boosting να προσπαθησουμε να ο βελτιστοποιησουμε
from sklearn.model_selection import GridSearchCV
X = other_variables.copy()  # Features
y = income.values.ravel()  # Target variable

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ορισμός του Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=42)

# Ορισμός του grid για hyperparameter tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 1.0]
}

# GridSearchCV για την εύρεση των καλύτερων παραμέτρων
grid_search = GridSearchCV(gb, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Καλύτερες παράμετροι
best_params = grid_search.best_params_
best_params
# {'learning_rate': 0.2,
#  'max_depth': 7,
#  'min_samples_split': 10,
#  'n_estimators': 500,
#  'subsample': 0.8}

#δοκιμη τνω καλυερων παραμετρων 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
best_gb = GradientBoostingClassifier(
    learning_rate=0.2,
    max_depth=7,
    min_samples_split=10,
    n_estimators=500,
    subsample=0.8,
    random_state=42
)

# Εκπαίδευση του βελτιωμένου μοντέλου
best_gb.fit(X_train, y_train)

# Πρόβλεψη στο test set
y_pred_gb = best_gb.predict(X_test)

# Υπολογισμός μετρικών απόδοσης
accuracy = accuracy_score(y_test, y_pred_gb)
precision = precision_score(y_test, y_pred_gb)
recall = recall_score(y_test, y_pred_gb)
f1 = f1_score(y_test, y_pred_gb)

# Εμφάνιση των αποτελεσμάτων
print(f"Βελτιστοποιημένο Gradient Boosting Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Εμφάνιση Confusion Matrix
cm = confusion_matrix(y_test, y_pred_gb)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - Optimized Gradient Boosting")
plt.show()
#done with 4 ,no best params first gradient boosting correct




