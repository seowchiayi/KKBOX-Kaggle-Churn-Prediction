from sklearn.preprocessing import MinMaxScaler

def normalize(train,test):
    scaler = MinMaxScaler()
    norm_train = scaler.fit_transform(train[norm_cols])
    norm_test = scaler.fit_transform(test[norm_cols])
    norm_train = pd.DataFrame(norm_train)
    norm_test = pd.DataFrame(norm_test)

train = pd.read_csv('../input/train_27.csv')
test = pd.read_csv('../input/test_27.csv')

combined = pd.get_dummies(pd.concat([train, test], keys = [0,1]), columns = ['registered_via', 'city', 'is_one_timer'])
train, test = combined.xs(0), combined.xs(1)
norm_cols = [c for c in train.columns if c not in ['is_churn','msno','streak','one_timer_trans','is_one_timer','registered_via', 'city']]
train = train.fillna(0)
test = test.fillna(0)
