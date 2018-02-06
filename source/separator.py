import numpy as np
import pandas as pd

#df_members = pd.read_csv('../processed/members_final.csv')
#df_transactions = pd.read_csv('../processed/transactions_final.csv')
train = pd.read_csv('../processed/train.csv')

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

# separate out to predict with those cases
# case 1) only member missing
# case 2) only userlog missing
# case 3) member, userlog missing
# case 4) only transactions missing
# case 5) perfect case, all exists

na_free = train.dropna(subset=['city','bd','gender','registered_via','payment_method_id','payment_plan_days','plan_list_price',
'actual_amount_paid','is_auto_renew','transaction_date','membership_expire_date','is_cancel','discount','is_discount','amt_per_day','loyalty','num_75_sum',
'date_count','num_unq_sum','total_secs_sum','num_50_sum','num_100_sum','num_25_sum','num_985_sum','activity_period','days_inactive',
'is_one_timer','average_time','average_unique'],how='all')

#rows with null values in all columns except msno and is churn

#rows = train[~train.index.isin(na_free.index)]


train1 = na_free
train1 = train1[(train1['city'].isnull()) &
(train1['bd'].isnull()) &
(train1['gender'].isnull()) &
(train1['registered_via'].isnull()) &
(train1['payment_method_id'].notnull()) &
(train1['payment_plan_days'].notnull()) &
(train1['plan_list_price'].notnull()) &
(train1['actual_amount_paid'].notnull()) &
(train1['is_auto_renew'].notnull()) &
(train1['transaction_date'].notnull()) &
(train1['membership_expire_date'].notnull()) &
(train1['is_cancel'].notnull()) &
(train1['discount'].notnull()) &
(train1['is_discount'].notnull()) &
(train1['amt_per_day'].notnull()) &
(train1['loyalty'].notnull()) &
(train1['num_75_sum'].notnull()) &
(train1['date_count'].notnull()) &
(train1['num_unq_sum'].notnull()) &
(train1['total_secs_sum'].notnull()) &
(train1['num_50_sum'].notnull()) &
(train1['num_100_sum'].notnull()) &
(train1['num_25_sum'].notnull()) &
(train1['num_985_sum'].notnull()) &
(train1['activity_period'].notnull()) &
(train1['days_inactive'].notnull()) &
(train1['is_one_timer'].notnull()) &
(train1['average_time'].notnull()) &
(train1['average_unique'].notnull())]

train1.drop(['city','bd','gender','registered_via'],axis=1,inplace=True)

print train1.isnull().any()
train1.to_csv('../processed/train1.csv',index=False)

train2 = na_free
train2 = train2[(train2['city'].notnull()) &
(train2['bd'].notnull()) &
(train2['gender'].notnull()) &
(train2['registered_via'].notnull()) &
(train2['payment_method_id'].notnull()) &
(train2['payment_plan_days'].notnull()) &
(train2['plan_list_price'].notnull()) &
(train2['actual_amount_paid'].notnull()) &
(train2['is_auto_renew'].notnull()) &
(train2['transaction_date'].notnull()) &
(train2['membership_expire_date'].notnull()) &
(train2['is_cancel'].notnull()) &
(train2['discount'].notnull()) &
(train2['is_discount'].notnull()) &
(train2['amt_per_day'].notnull()) &
(train2['loyalty'].notnull()) &
(train2['num_75_sum'].isnull()) &
(train2['date_count'].isnull()) &
(train2['num_unq_sum'].isnull()) &
(train2['total_secs_sum'].isnull()) &
(train2['num_50_sum'].isnull()) &
(train2['num_100_sum'].isnull()) &
(train2['num_25_sum'].isnull()) &
(train2['num_985_sum'].isnull()) &
(train2['activity_period'].isnull()) &
(train2['days_inactive'].isnull()) &
(train2['is_one_timer'].isnull()) &
(train2['average_time'].isnull()) &
(train2['average_unique'].isnull())]

train2.drop(['num_75_sum','date_count','num_unq_sum','total_secs_sum','num_50_sum','num_100_sum','num_25_sum','num_985_sum',
'activity_period','days_inactive','is_one_timer','average_time','average_unique'],axis=1,inplace=True)

print train2.isnull().any()
train2.to_csv('../processed/train2.csv',index=False)


train3 = na_free
train3 = train3[(train3['city'].isnull()) &
(train3['bd'].isnull()) &
(train3['gender'].isnull()) &
(train3['registered_via'].isnull()) &
(train3['payment_method_id'].notnull()) &
(train3['payment_plan_days'].notnull()) &
(train3['plan_list_price'].notnull()) &
(train3['actual_amount_paid'].notnull()) &
(train3['is_auto_renew'].notnull()) &
(train3['transaction_date'].notnull()) &
(train3['membership_expire_date'].notnull()) &
(train3['is_cancel'].notnull()) &
(train3['discount'].notnull()) &
(train3['is_discount'].notnull()) &
(train3['amt_per_day'].notnull()) &
(train3['loyalty'].notnull()) &
(train3['num_75_sum'].isnull()) &
(train3['date_count'].isnull()) &
(train3['num_unq_sum'].isnull()) &
(train3['total_secs_sum'].isnull()) &
(train3['num_50_sum'].isnull()) &
(train3['num_100_sum'].isnull()) &
(train3['num_25_sum'].isnull()) &
(train3['num_985_sum'].isnull()) &
(train3['activity_period'].isnull()) &
(train3['days_inactive'].isnull()) &
(train3['is_one_timer'].isnull()) &
(train3['average_time'].isnull()) &
(train3['average_unique'].isnull())]

train3.drop(['city','bd','gender','registered_via','num_75_sum','date_count','num_unq_sum','total_secs_sum','num_50_sum','num_100_sum','num_25_sum','num_985_sum',
'activity_period','days_inactive','is_one_timer','average_time','average_unique'],axis=1,inplace=True)

print train3.isnull().any()
train3.to_csv('../processed/train3.csv',index=False)

train4 = na_free
train4 = train4[(train4['city'].notnull()) &
(train4['bd'].notnull()) &
(train4['gender'].notnull()) &
(train4['registered_via'].notnull()) &
(train4['payment_method_id'].isnull()) &
(train4['payment_plan_days'].isnull()) &
(train4['plan_list_price'].isnull()) &
(train4['actual_amount_paid'].isnull()) &
(train4['is_auto_renew'].isnull()) &
(train4['transaction_date'].isnull()) &
(train4['membership_expire_date'].isnull()) &
(train4['is_cancel'].isnull()) &
(train4['discount'].isnull()) &
(train4['is_discount'].isnull()) &
(train4['amt_per_day'].isnull()) &
(train4['loyalty'].isnull()) &
(train4['num_75_sum'].notnull()) &
(train4['date_count'].notnull()) &
(train4['num_unq_sum'].notnull()) &
(train4['total_secs_sum'].notnull()) &
(train4['num_50_sum'].notnull()) &
(train4['num_100_sum'].notnull()) &
(train4['num_25_sum'].notnull()) &
(train4['num_985_sum'].notnull()) &
(train4['activity_period'].notnull()) &
(train4['days_inactive'].notnull()) &
(train4['is_one_timer'].notnull()) &
(train4['average_time'].notnull()) &
(train4['average_unique'].notnull())]

train4.drop(['payment_method_id','payment_plan_days','plan_list_price','actual_amount_paid','is_auto_renew',
'transaction_date','membership_expire_date','is_cancel','discount','is_discount','amt_per_day','loyalty'],axis=1,inplace=True)

print train4.isnull().any()
train4.to_csv('../processed/train4.csv',index=False)

train5 = na_free
train5 = train5[(train5['city'].notnull()) &
(train5['bd'].notnull()) &
(train5['gender'].notnull()) &
(train5['registered_via'].notnull()) &
(train5['payment_method_id'].notnull()) &
(train5['payment_plan_days'].notnull()) &
(train5['plan_list_price'].notnull()) &
(train5['actual_amount_paid'].notnull()) &
(train5['is_auto_renew'].notnull()) &
(train5['transaction_date'].notnull()) &
(train5['membership_expire_date'].notnull()) &
(train5['is_cancel'].notnull()) &
(train5['discount'].notnull()) &
(train5['is_discount'].notnull()) &
(train5['amt_per_day'].notnull()) &
(train5['loyalty'].notnull()) &
(train5['num_75_sum'].notnull()) &
(train5['date_count'].notnull()) &
(train5['num_unq_sum'].notnull()) &
(train5['total_secs_sum'].notnull()) &
(train5['num_50_sum'].notnull()) &
(train5['num_100_sum'].notnull()) &
(train5['num_25_sum'].notnull()) &
(train5['num_985_sum'].notnull()) &
(train5['activity_period'].notnull()) &
(train5['days_inactive'].notnull()) &
(train5['is_one_timer'].notnull()) &
(train5['average_time'].notnull()) &
(train5['average_unique'].notnull())]

print train5.isnull().any()
train5.to_csv('../processed/train5.csv',index=False)
