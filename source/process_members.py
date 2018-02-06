import numpy as np
import pandas as pd

df_members = pd.read_csv('../input/members_v3.csv')
df_members['bd'] = df_members['bd'].clip_lower(0)
df_members['bd'] = df_members['bd'].clip_upper(75)
df_members['registration_init_time'] = pd.to_datetime(df_members['registration_init_time'], format='%Y%m%d')

gender = {'male':1, 'female':2}
df_members['gender'] = df_members['gender'].map(gender)
df_members = df_members.fillna(0)
df_members['gender'] = df_members['gender'].astype(int)
df_members.to_csv('./processed/members_final.csv', index=False)
