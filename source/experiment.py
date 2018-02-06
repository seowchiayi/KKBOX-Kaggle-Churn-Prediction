from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
import time
from datetime import datetime

start_time=time.time()
def run_cv(X,y,test,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()

    # Iterate through folds
    print 'Training...'
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        classifier = clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    y_pred = pd.DataFrame(classifier.predict_proba(test),columns=clf.classes_)
    print y_pred.head(n=100)
    return y_pred

def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

train = pd.read_csv('../input/train_27.csv',na_values='NaN')
test = pd.read_csv('../input/test_27.csv',na_values='NaN')
target = train['is_churn']

# print 'Hot encoding'
# combined = pd.get_dummies(pd.concat([train, test], keys = [0,1]), columns = [c for c in train.columns if c in ['is_one_timer','registered_via', 'city']])
# train, test = combined.xs(0), combined.xs(1)

norm_cols = [(train.columns.get_loc(c))-2 for c in train.columns if c not in ['is_churn','msno','streak','one_timer_trans','is_one_timer','registered_via', 'city']]
cols = [c for c in train.columns if c not in ['is_churn','msno']]
concat_cols = [(train.columns.get_loc(c))-2 for c in train.columns if c in ['streak','one_timer_trans','is_one_timer','registered_via', 'city']]

print 'Filling missing data'
imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
imp_train = imp.fit_transform(train[cols])
imp_test = imp.fit_transform(test[cols])
train_fillna = pd.DataFrame(imp_train)
test_fillna = pd.DataFrame(imp_test)

print 'Normalizing data'
scaler = MinMaxScaler()
norm_X = scaler.fit_transform(train_fillna[norm_cols])
norm_Y = scaler.fit_transform(test_fillna[norm_cols])
norm_X = pd.DataFrame(norm_X)
norm_Y = pd.DataFrame(norm_Y)
train_norm = pd.concat([norm_X,train_fillna[concat_cols]],axis=1)
test_norm = pd.concat([norm_Y,test_fillna[concat_cols]],axis=1)

X = train_norm.as_matrix().astype(np.float)
test_set = test_norm.as_matrix().astype(np.float)
target = target.as_matrix().astype(np.float)

print "Random forest:"
y_pred = run_cv(X,target,test_set,RF)
test['is_churn'] = y_pred[1.0]
#print "%.3f" % accuracy(target, run_cv(X,target,test,RF))
test[['msno','is_churn']].to_csv('submission_rf.csv', index=False)
print((time.time()-start_time)/60)
