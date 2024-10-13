import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %matplotlib inline

!gdown --id "11vCKuk81QvTQcwibMX7Wo6k4q0HDzLOz"

df = pd.read_csv("/content/House_Rent_Dataset.csv")
df.head()

"""# Data Understanding & Removing Outlier"""

df.shape

df.info()

# The 'Point of Contact' and 'Posted On' features do not affect the rent price model, so they will be dropped
df = df.drop(['Posted On', 'Point of Contact'], axis='columns')

"""## Univariate Analysis"""

df.groupby('Area Type')['Area Type'].agg('count')

# The 'Area Type' feature has only 2 samples of 'Built Area', so these samples will be removed
df.drop(df.index[df['Area Type'] == 'Built Area'], inplace=True)

df.groupby('Area Type')['Area Type'].agg('count')

df.groupby('City')['City'].agg('count')

df.groupby('Furnishing Status')['Furnishing Status'].agg('count')

df.groupby('Tenant Preferred')['Tenant Preferred'].agg('count')

df.groupby('Floor')['Floor'].agg('count')

df.groupby('Area Locality')['Area Locality'].agg('count')

# The 'Floor' and 'Area Locality' features have too many unique values, so they will be dropped
df = df.drop(['Floor', 'Area Locality'], axis='columns')

df.head()

df.hist(bins=50, figsize=(10,10))
plt.ticklabel_format(useOffset=False, style='plain')
plt.show()

df.Rent.describe().apply(lambda x: format(x, 'f'))

"""## Multivariate Analysis"""

# Adding a new feature: price per sqft
df['Price_per_sqft'] = df['Rent'] * 1000 / df['Size']

df.head()

# Detecting size per BHK outliers
# 100 sqft for 1 BHK is uncommon, so we consider the threshold to be 300 sqft/BHK

df[(df.Size / df.BHK) < 300].head()

df.shape

# Removing size per BHK outliers
df1 = df[~(df.Size / df.BHK < 300)]
df1.head()

df1.shape

# Detecting price per sqft outliers
df1.Price_per_sqft.describe().apply(lambda x: format(x, 'f'))

"""A price of 571 per sqft is very low, and a price of 1400000 per sqft is very high."""

# Removing price per sqft outliers using mean and one standard deviation
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('City'):
        m = np.mean(subdf.Price_per_sqft)
        st = np.std(subdf.Price_per_sqft)
        reduced_df = subdf[(subdf.Price_per_sqft > (m - st)) & (subdf.Price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df2 = remove_pps_outliers(df1)
df2.shape

# Detecting bathroom outliers
# Having 4 bathrooms for a 2 BHK is unusual, so we consider the limit to be the number of BHK + 2

df2[df2.Bathroom > df2.BHK + 2]

# Removing bathroom outliers
df2 = df2[~(df2.Bathroom > df2.BHK + 2)]
df2.head()

df2.shape

# Dropping the 'Price_per_sqft' feature as it is no longer needed
df3 = df2.drop(['Price_per_sqft'], axis='columns')

# Viewing the correlation between numerical features and the target feature (rent)
plt.figure(figsize=(10, 8))
correlation_matrix = df3.corr().round(2)

# To print the values inside the boxes, use the parameter annot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix for Numerical Features", size=20)

# Viewing the correlation between categorical features and the target feature (rent)
cat_features = df2.select_dtypes(include='object').columns.to_list()

for col in cat_features:
    sns.catplot(x=col, y="Rent", kind="bar", dodge=False, height=4, aspect=3, data=df2, palette="Set3")
    plt.title("Average 'Rent' Relative to - {}".format(col))

"""# Data Preparation

## One hot encoding
"""

df3 = pd.get_dummies(data=df3, columns=['Area Type'])
df3 = pd.get_dummies(data=df3, columns=['City'])
df3 = pd.get_dummies(data=df3, columns=['Furnishing Status'])
df3 = pd.get_dummies(data=df3, columns=['Tenant Preferred'])

df3.head()

"""## Train Test Split"""

from sklearn.model_selection import train_test_split

X = df3.drop(["Rent"], axis=1)
y = df3["Rent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=123)

print(f'Total # of samples in whole dataset: {len(X)}')
print(f'Total # of samples in train dataset: {len(X_train)}')
print(f'Total # of samples in test dataset: {len(X_test)}')

"""## Normalization"""

from sklearn.preprocessing import StandardScaler

# Normalizing the training data
numerical_features = ['BHK', 'Size', 'Bathroom']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

# Normalizing the test data
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

"""# Modeling

## Grid Search
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

def grid_search_model(X, y):
    algos = {
        'knn': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            }
        },
        'boosting': {
            'model': AdaBoostRegressor(),
            'params': {
                'learning_rate': [0.1, 0.05, 0.01, 0.05, 0.001],
                'n_estimators': [25, 50, 75, 100],
                'random_state': [11, 33, 55, 77]
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [25, 50, 75, 100],
                'max_depth': [8, 16, 32, 64],
                'random_state': [11, 33, 55, 77],
            }
        }

    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.05, random_state=123)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

grid_search_model(X, y)

"""## Model with best parameter"""

acc = pd.DataFrame(index=['accuracy'])

from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train, y_train)
acc.loc['accuracy', 'knn'] = knn.score(X_test, y_test)
knn.score(X_test, y_test)

rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=11)
rf.fit(X_train, y_train)
acc.loc['accuracy', 'rf'] = rf.score(X_test, y_test)
rf.score(X_test, y_test)

boosting = AdaBoostRegressor(n_estimators=25, learning_rate=0.001, random_state=11)                             
boosting.fit(X_train, y_train)
acc.loc['accuracy', 'boosting'] = boosting.score(X_test, y_test)
boosting.score(X_test, y_test)

"""# Evaluation"""

# Accuracy of the models
acc

# Mean squared error of the models
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN', 'RF', 'Boosting'])

model_dict = {'KNN': knn, 'RF': rf, 'Boosting': boosting}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train)) / 1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test)) / 1e3
 
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

prediksi = X_test.iloc[5:10].copy()
pred_dict = {'y_true': y_test[5:10]}
for name, model in model_dict.items():
    pred_dict['prediksi_' + name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)
