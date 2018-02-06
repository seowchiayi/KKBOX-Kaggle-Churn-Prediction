import pandas as pd
from sklearn import *
import sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.rename(columns={'payment_method_id_<lambda>' : 'payment_method_id'}, inplace=True)
test.rename(columns={'payment_method_id_<lambda>' : 'payment_method_id'}, inplace=True)

#train.drop(['actual_amount_paid_min', 'actual_amount_paid_max', 'plan_list_price_min', 'plan_list_price_max', 'payment_plan_days_min', 'payment_plan_days_max', 'amt_per_day_max', 'amt_per_day_min', 'num_25_sum', 'num_50_sum', 'num_75_sum','num_985_sum'],axis=1, inplace=True)
#test.drop(['actual_amount_paid_min', 'actual_amount_paid_max', 'plan_list_price_min', 'plan_list_price_max', 'payment_plan_days_min', 'payment_plan_days_max', 'amt_per_day_max', 'amt_per_day_min', 'num_25_sum', 'num_50_sum', 'num_75_sum','num_985_sum'],axis=1, inplace=True)


# combined = pd.get_dummies(pd.concat([train, test], keys = [0,1]), columns = ['gender', 'registered_via', 'city', 'is_one_timer'])
# train, test = combined.xs(0), combined.xs(1)
cols = [c for c in train.columns if c not in ['is_churn','msno']]

fold = 5
for i in range(fold):
    params = {
        'eta': 0.02, #best 0.08
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': i,
        'gamma':1,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 150,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500
    if i != 0:
        pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
pred /= fold
plt.rcParams['figure.figsize'] = (7.0, 7.0)
xgb.plot_importance(booster=model)
plt.show()
test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)

# test['is_churn'] = test['is_churn'].apply(lambda x: 0 if x <= 0.00001 else x)
# test['is_churn'] = test['is_churn'].apply(lambda x: 1 if x >= 0.99 else x)
test[['msno','is_churn']].to_csv('submission1.csv', index=False)
