import numpy as np
import pandas as pd

def change_datatype(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)


df_transactions = pd.read_csv('../input/transactions.csv')
df_transactions = df_transactions.append(pd.read_csv('../input/transactions_v2.csv'))
print "Finish uploaded"
change_datatype(df_transactions)
change_datatype_float(df_transactions)
print "Finish changing datatype"


df_transactions_for_crack = df_transactions.groupby('msno')['membership_expire_date'].max().reset_index(name='membership_expire_date')

df_transactions_train_crack = df_transactions_for_crack.copy()
df_transactions_train_crack['check_churn'] = (df_transactions_train_crack['membership_expire_date'] < 20170101).astype(int)
df_transactions_train_crack.drop(['membership_expire_date'],axis=1,inplace=True)

df_transactions_cv_crack = df_transactions_for_crack.copy()
df_transactions_cv_crack['check_churn'] = (df_transactions_cv_crack['membership_expire_date'] < 20170201).astype(int)
df_transactions_cv_crack.drop(['membership_expire_date'],axis=1,inplace=True)

df_transactions_test_crack = df_transactions_for_crack.copy()
df_transactions_test_crack['check_churn'] = (df_transactions_test_crack['membership_expire_date'] < 20170301).astype(int)
df_transactions_test_crack.drop(['membership_expire_date'],axis=1,inplace=True)


transactions_train = df_transactions.loc[df_transactions.transaction_date < 20170201].copy()
transactions_cv = df_transactions.loc[df_transactions.transaction_date < 20170301].copy()
transactions_test = df_transactions.loc[df_transactions.transaction_date < 20170401].copy()

del df_transactions, df_transactions_for_crack
transactions_train = transactions_train.loc[transactions_train.membership_expire_date >= 20150101]
transactions_cv = transactions_cv.loc[transactions_cv.membership_expire_date >= 20150201]
transactions_test = transactions_test.loc[transactions_test.membership_expire_date >= 20150301]


date_cols = ['transaction_date', 'membership_expire_date']
for col in date_cols:
    transactions_train[col] = pd.to_datetime(transactions_train[col], format='%Y%m%d')
    transactions_cv[col] = pd.to_datetime(transactions_cv[col], format='%Y%m%d')
    transactions_test[col] = pd.to_datetime(transactions_test[col], format='%Y%m%d')


transactions_train['emome']  = ((transactions_train['plan_list_price'] == 0) & (transactions_train['actual_amount_paid'] > 0)).astype(int)
transactions_cv['emome']  = ((transactions_cv['plan_list_price'] == 0) & (transactions_cv['actual_amount_paid'] > 0)).astype(int)
transactions_test['emome']  = ((transactions_test['plan_list_price'] == 0) & (transactions_test['actual_amount_paid'] > 0)).astype(int)


transactions_train['discount'] = (transactions_train['plan_list_price'] - transactions_train['actual_amount_paid']).clip_lower(0)
transactions_cv['discount'] = (transactions_cv['plan_list_price'] - transactions_cv['actual_amount_paid']).clip_lower(0)
transactions_test['discount'] = (transactions_test['plan_list_price'] - transactions_test['actual_amount_paid']).clip_lower(0)

transactions_train['is_discount'] = transactions_train.discount.apply(lambda x: 1 if x > 0 else 0)
transactions_cv['is_discount'] = transactions_cv.discount.apply(lambda x: 1 if x > 0 else 0)
transactions_test['is_discount'] = transactions_test.discount.apply(lambda x: 1 if x > 0 else 0)


transactions_train['loyalty'] = transactions_train['membership_expire_date'].sub(transactions_train['transaction_date']).dt.days
transactions_cv['loyalty'] = transactions_cv['membership_expire_date'].sub(transactions_cv['transaction_date']).dt.days
transactions_test['loyalty'] = transactions_test['membership_expire_date'].sub(transactions_test['transaction_date']).dt.days


transactions_train['free_trial'] = ((transactions_train['actual_amount_paid']==0) & (transactions_train['payment_plan_days']==0)).astype(int)
transactions_cv['free_trial'] = ((transactions_cv['actual_amount_paid']==0) & (transactions_cv['payment_plan_days']==0)).astype(int)
transactions_test['free_trial'] = ((transactions_test['actual_amount_paid']==0) & (transactions_test['payment_plan_days']==0)).astype(int)


transactions_train['amt_per_day']=transactions_train['actual_amount_paid'].div(transactions_train['payment_plan_days'])
transactions_train['amt_per_day'].replace(np.inf, 0, inplace=True)
transactions_train['amt_per_day'].fillna(0, inplace=True)

transactions_cv['amt_per_day']=transactions_cv['actual_amount_paid'].div(transactions_cv['payment_plan_days'])
transactions_cv['amt_per_day'].replace(np.inf, 0, inplace=True)
transactions_cv['amt_per_day'].fillna(0, inplace=True)

transactions_test['amt_per_day']=transactions_test['actual_amount_paid'].div(transactions_test['payment_plan_days'])
transactions_test['amt_per_day'].replace(np.inf, 0, inplace=True)
transactions_test['amt_per_day'].fillna(0, inplace=True)


combined = pd.get_dummies(pd.concat([transactions_train, transactions_cv, transactions_test], keys = [0,1,2]), columns = ['payment_method_id'])
transactions_train, transactions_cv, transactions_test = combined.xs(0), combined.xs(1), combined.xs(2)

transactions_train['late_count'] = transactions_train.loyalty.apply(lambda x: 1 if x < 0 else 0)
transactions_cv['late_count'] = transactions_cv.loyalty.apply(lambda x: 1 if x < 0 else 0)
transactions_test['late_count'] = transactions_test.loyalty.apply(lambda x: 1 if x < 0 else 0)

def process_trans(chunk):
    grouped_object=chunk.groupby('msno',sort=False) # not sorting results in a minor speedup
    func = {
        'payment_plan_days': ['sum'], 'plan_list_price': ['sum'], 'actual_amount_paid': ['mean'],
        'is_auto_renew':['sum'], 'is_cancel':['sum'], 'emome': ['sum'], 'discount':['sum'], 'is_discount':['sum'],
        'loyalty': ['sum'], 'free_trial': ['sum'], 'amt_per_day': ['mean'],
        'payment_method_id_1':['sum'],
        'payment_method_id_2':['sum'], 'payment_method_id_3':['sum'],'payment_method_id_4':['sum'],
        'payment_method_id_5':['sum'],'payment_method_id_6':['sum'],'payment_method_id_7':['sum'],
        'payment_method_id_8':['sum'],'payment_method_id_10':['sum'],'payment_method_id_11':['sum'],
        'payment_method_id_12':['sum'],'payment_method_id_13':['sum'],'payment_method_id_14':['sum'],
        'payment_method_id_15':['sum'],'payment_method_id_16':['sum'],'payment_method_id_17':['sum'],
        'payment_method_id_18':['sum'],'payment_method_id_19':['sum'],'payment_method_id_20':['sum'],
        'payment_method_id_21':['sum'],'payment_method_id_22':['sum'],'payment_method_id_23':['sum'],
        'payment_method_id_24':['sum'],'payment_method_id_25':['sum'],'payment_method_id_26':['sum'],
        'payment_method_id_27':['sum'],'payment_method_id_28':['sum'],'payment_method_id_29':['sum'],
        'payment_method_id_30':['sum'],'payment_method_id_31':['sum'],'payment_method_id_32':['sum'],
        'payment_method_id_33':['sum'],'payment_method_id_34':['sum'],'payment_method_id_35':['sum'],
        'payment_method_id_36':['sum'],'payment_method_id_37':['sum'],'payment_method_id_38':['sum'],
        'payment_method_id_39':['sum'],'payment_method_id_40':['sum'],'payment_method_id_41':['sum'],
        'transaction_date': ['max'], 'membership_expire_date': ['max'], 'late_count': ['sum']
        }
    result=grouped_object.agg(func)
    return result


transactions_train = process_trans(transactions_train)
transactions_train.columns = ['_'.join(col).strip() for col in transactions_train.columns.values]
transactions_train.reset_index(inplace=True)

transactions_cv = process_trans(transactions_cv)
transactions_cv.columns = ['_'.join(col).strip() for col in transactions_cv.columns.values]
transactions_cv.reset_index(inplace=True)

transactions_test = process_trans(transactions_test)
transactions_test.columns = ['_'.join(col).strip() for col in transactions_test.columns.values]
transactions_test.reset_index(inplace=True)


transactions_train = pd.merge(transactions_train, df_transactions_train_crack, how='left', on='msno')
transactions_cv = pd.merge(transactions_cv, df_transactions_cv_crack, how='left', on='msno')
transactions_test = pd.merge(transactions_test, df_transactions_test_crack, how='left', on='msno')



transactions_train.to_csv('../processed/transaction_train_final.csv',index=False)
transactions_cv.to_csv('../processed/transaction_cv_final.csv',index=False)
transactions_test.to_csv('../processed/transactions_test_final.csv',index=False)


print list(transactions_train)
