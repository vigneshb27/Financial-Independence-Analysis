import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
import pickle
import numpy as np

df = pd.read_csv('final_fi_data.csv')

label_encoder = preprocessing.LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

df2 = df.drop(['gender', 'no_of_people_contribute_income', 'relationship_status', 'children_status', 'occupation', 'housing situation', 'have_roomates', 'country', 'living environment', 'commute', 'highest level of education attained?', 'amount_for_re', 'frequency_of_checking_balance', 'do_you_move_when_fi', 'how_financially_stable_childhood_home', 'parent/guardian_financial_status',
              'did_parent/guardian_teach_about_money', 'age_when_financially_literate', 'age_when_learnt_about_fi', 'education', 'job', 'housing_situation', 'location', 'spending_habits', 'frugality', 'logical_thinking', 'creativity', 'drive_for_self_improvement', 'importance_of_relationship', 'internet_consumption', 'introversion', 'religiousness_level'], axis=1)

X = df2.drop('financially_independent', axis=1)
y = df2['financially_independent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0)

# Assign feature names as a list
feature_names = ['age', 'gross_annual_wage', 'total_portfolio', 'amount_for_fi',
                 'curr_percentage_of_fi', 'response_release_consent']

X_train = X_train.to_numpy()
X_train = pd.DataFrame(X_train, columns=feature_names)

X_test = X_test.to_numpy()
X_test = pd.DataFrame(X_test, columns=feature_names)

clf = svm.SVC()
clf.fit(X_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

print(model.predict([[37, 128000, 300000, 1000000, 40.0, 0]]))
