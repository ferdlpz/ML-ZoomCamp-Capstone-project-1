import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# parameters
min_samples_leaf = 1
max_depth = 5
n_estimators = 40
output_file = f'model_min_samples_leaf={min_samples_leaf}_max_depth={max_depth}_n_estimators={n_estimators}.bin'
path_save = '/Users/fdl/Repos/ML-ZoomCamp-Capstone-project-1/04 Script predict/'

# data loading and preparation
df = pd.read_csv(r'/Users/fdl/Repos/ML-ZoomCamp-Capstone-project-1/01 Data/Heart_Disease_Prediction.csv')
# rename target
df = df.rename(columns={'Heart Disease': 'y'})
df['y'] = df['y'].map({'Presence': 1, 'Absence': 0})

feature_selection = ['ST depression', 'Chest pain type', 'Exercise angina', 'Sex',
            'Number of vessels fluro', 'BP', 'Max HR', 'EKG results', 'FBS over 120', 'Cholesterol', 'Age']

df = df[feature_selection + ['y']]

# train final model
dv = DictVectorizer(sparse=False)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.y.values

df_test = df_test.reset_index(drop=True)
y_test = df_test.y.values

X_train = dv.fit_transform(df_full_train[feature_selection].to_dict(orient='records'))
X_val = dv.fit_transform(df_test[feature_selection].to_dict(orient='records'))

rf = RandomForestClassifier(n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=1)

rf.fit(X_train, y_full_train)

y_pred = rf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"RandomForest AUC: {auc:.3f}")

print("RandomForest Accuracy Report:")
print(classification_report(y_test, (y_pred >= 0.5).astype(int)))

print(f'Saving the model to {output_file}')
with open(path_save + output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)
print(f'Model saved')