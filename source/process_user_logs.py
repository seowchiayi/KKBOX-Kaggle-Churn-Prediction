import numpy as np
import pandas as pd
from datetime import datetime

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

def f(x):
    rng = pd.date_range(x.min(),x.max())
    return len(rng.difference(x))

def days_inactive(x):
    if(x>0):
        if (x>0.25 and x <=0.75):
            return 1
        elif x>0.75:
            return 2
        else:
            return 0
    else:
        return 0

def one_timer(x):
    if x<=1:
        return 1
    else:
        return 0

def day_of_week(chunk):
    chunk['weekday'] = pd.to_datetime(chunk.date, format='%Y%m%d').apply(lambda x: 1 if x.weekday() else 0)
    chunk['weekend'] = pd.to_datetime(chunk.date, format='%Y%m%d').apply(lambda x: 0 if x.weekday() else 1)
    # chunk['day_of_week'] = chunk['date'].dt.dayofweek
    # days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    # for i in range(len(days)):
    #     chunk[days[i]] = chunk.day_of_week.apply(lambda x: 0 if x!=i else 1)
    return chunk

def process_user_log(chunk):
    grouped_object=chunk.groupby('msno',sort=False) # not sorting results in a minor speedup
    func = {'date':['min','max','count'],
       'num_25':['sum'],'num_50':['sum'],
       'num_75':['sum'],'num_985':['sum'],
       'num_100':['sum'],'num_unq':['sum'],'total_secs':['sum'],
       'weekday':['sum'], 'weekend':['sum']}
    result=grouped_object.agg(func)
    return result

def process_user_log_L2(chunk):
    grouped_object=chunk.groupby('msno',sort=False) # not sorting results in a minor speedup
    func = {'date_min':['min'], 'date_max':['max'], 'date_count':['sum'],
       'num_25_sum':['sum'],'num_50_sum':['sum'],
       'num_75_sum':['sum'],'num_985_sum':['sum'],
       'num_100_sum':['sum'],'num_unq_sum':['sum'],'total_secs_sum':['sum'],
       'weekday_sum':['sum'], 'weekend_sum':['sum']}
    result=grouped_object.agg(func)
    return result

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df_user_logs = pd.read_csv('../input/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
df_user_logs_v2 = pd.read_csv('../input/user_logs_v2.csv')

temp_df = pd.DataFrame()
for chunk in df_user_logs:
    print "Processing. . ."
    change_datatype(chunk)
    change_datatype_float(chunk)
    chunk = chunk.loc[chunk.date < 20150201]
    chunk.total_secs = chunk.total_secs.clip(lower=0.0)
    chunk = day_of_week(chunk)
    chunk = process_user_log(chunk)
    temp_df = temp_df.append(chunk)

df_user_logs = temp_df
change_datatype(df_user_logs_v2)
change_datatype_float(df_user_logs_v2)
df_user_logs_v2 = df_user_logs_v2.loc[df_user_logs_v2.date < 20150201]
df_user_logs_v2 = df_user_logs_v2.clip(lower=0.0)
df_user_logs_v2 = day_of_week(df_user_logs_v2)
df_user_logs_v2 = process_user_log(df_user_logs_v2)
df_user_logs.append(df_user_logs_v2)
df_user_logs.columns = ['_'.join(col).strip() for col in df_user_logs.columns.values]
df_user_logs = process_user_log_L2(df_user_logs)
df_user_logs.columns = df_user_logs.columns.droplevel(1)
df_user_logs.reset_index(inplace=True)


print "Sorting"
df_user_logs = df_user_logs.sort_values(['msno'],ascending=True)

print "Convert to datetime"
df_user_logs.date_min = pd.to_datetime(df_user_logs.date_min, format='%Y%m%d')
df_user_logs.date_max = pd.to_datetime(df_user_logs.date_max, format='%Y%m%d')
#Feature 1: the range of user's activity days
df_user_logs['activity_period'] = (df_user_logs.date_max-df_user_logs.date_min).dt.days + 1

#Feature 2: days not active, days where no activity were recorded
df_user_logs['days_inactive'] = df_user_logs.activity_period - df_user_logs.date_count

#Feature 3: type of user, where he rarely uses kkbox, uses it moderately or frequently
# 0 rare 1 moderate 2 frequent
df_user_logs['is_one_timer'] = df_user_logs.activity_period.apply(one_timer)

#Feature 4: time spent per day
df_user_logs['average_time'] =df_user_logs.total_secs_sum / df_user_logs.date_count

#Feature 5: Average unique song played
df_user_logs['average_unique'] = df_user_logs.num_unq_sum / df_user_logs.date_count
df_user_logs['average_25'] = df_user_logs.num_25_sum / df_user_logs.date_count
df_user_logs['average_50'] = df_user_logs.num_50_sum / df_user_logs.date_count
df_user_logs['average_75'] = df_user_logs.num_75_sum / df_user_logs.date_count
df_user_logs['average_985'] = df_user_logs.num_985_sum / df_user_logs.date_count
df_user_logs['average_100'] = df_user_logs.num_100_sum / df_user_logs.date_count

df_user_logs.to_csv('../processed/user_logs_final_training.csv',index=False)
