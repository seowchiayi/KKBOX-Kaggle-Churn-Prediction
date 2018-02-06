import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
from subprocess import check_output
from collections import Counter
import scipy.stats as st


pd.set_option('display.width', 120)

df_train = pd.read_csv('../input/train.csv')
df_train = df_train.append(pd.read_csv('../input/train_v2.csv'))
df_members = pd.read_csv('../processed/members_final.csv')
df_transactions_train = pd.read_csv('../processed/transaction_train_final.csv')
df_transactions_test = pd.read_csv('../processed/transactions_test_final.csv')
df_sample = pd.read_csv('../input/sample_submission_v2.csv')
df_user_logs = pd.read_csv('../processed/user_logs_final.csv')

df_train.drop_duplicates(subset='msno', keep='last', inplace=True)
df_train = pd.merge(df_train, df_members, how='left', on='msno')
df_sample = pd.merge(df_sample, df_members, how='left', on='msno')


df_train = pd.merge(df_train, df_transactions_train, how='left', on='msno')
df_sample = pd.merge(df_sample, df_transactions_test, how='left', on='msno')


df_train = pd.merge(df_train, df_user_logs, how='left', on='msno')
df_sample = pd.merge(df_sample, df_user_logs, how='left', on='msno')

df_train.to_csv('train.csv', index=False)
df_sample.to_csv('test.csv', index=False)
