from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import pickle
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt

C=1.0
output_file = f'model_C={C}.bin'
model_file = 'model_C=1.0.bin'

path = kagglehub.dataset_download("blastchar/telco-customer-churn")
df = pd.read_csv('/home/codespace/.cache/kagglehub/datasets/blastchar/telco-customer-churn/versions/1/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes.values == "object"].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
df['churn'] = np.where(df['churn'] == 'no', 0, 1)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

y_train = df_train.churn.values
y_test = df_test.churn.values
y_val = df_val.churn.values

numerical = ['totalcharges', 'monthlycharges', 'tenure']


categorical = ['gender', 'seniorcitizen', 'partner', 'dependents', 
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
               'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling','paymentmethod']


def train(df_train, y_train, C=1.0):
    # get the necessary columns, convert to dictionary
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False) # Transforms lists of feature-value mappings to vectors.
    X_train = dv.fit_transform(dicts) # dv transforms the feature matrix to vectors

    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X_train, y_train)
    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.fit_transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred


kfold = KFold(n_splits = 5, shuffle=True, random_state=1)
dv, model = train(df_full_train, df_full_train.churn.values, C=C)
y_predict= predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_predict)
print(auc)
# ### Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"The model is saved to {output_file}")
