import streamlit as st
import plotly.graph_objects as go
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


data = pd.read_csv("C:/Users/Checkout/PycharmProjects/CMPE 257/Dataset/smogn_clean_train.csv")
dataset = pd.read_csv("C:/Users/Checkout/PycharmProjects/CMPE 257/Dataset/train.csv")
print(data.head())
print(dataset.columns)
data = data.drop(['Unnamed: 0'], axis=1)


st.title("House Prediction Dashboard")
st.markdown("The dashboard will help to get to know more about the given datasets and it's output")

options = st.sidebar.multiselect("Select Attributes", data.columns)

names = ["Random Forest Regressor", "KNeighbours Regressor", "Gradient Boosting Regressor", "XG Boost Regressor", "Linear Regression"]
classifiers = [
    RandomForestRegressor(max_depth=3, random_state=1, n_estimators=10),
    KNeighborsRegressor(n_neighbors=3),
    GradientBoostingRegressor(random_state=1),
    XGBRegressor(),
    LinearRegression()
]

# iterate over classifiers
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


X = data.drop(['SalePrice'], axis=1)
y = data['SalePrice']

normalized_X=(X-X.min())/(X.max()-X.min())

# split the features and labels into  train and test data
X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.2, random_state=1)


m = pd.DataFrame(columns=["Model", "R2- Score"])

if st.sidebar.button("Predict"):

    max_score = 1
    max_class = ''

    X_train_temp = X_train[options]
    X_test_temp = X_test[options]

    for name, clf in zip(names, classifiers):

        start_time = time.time()
        clf.fit(X_train_temp, y_train)
        score = 100.0 * clf.score(X_test_temp, y_test)

        y_pred = clf.predict(X_test_temp)

        mse = mean_squared_error(y_test, y_pred)
        rmse_val = rmse(y_pred, y_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        m.loc[len(m.index)] = [ name, r2]

        if r2 < max_score:
            clf_best = clf
            max_score = r2
            max_class = name

        # print('Best --> Classifier = %s, Score (test, r2) = %.2f' % (max_class, max_score))

    st.markdown('Best Classifier = %s ' % max_class)
    st.markdown('Best r2 Score on Test Data = %.2f' % max_score)
    st.table(m)


selected_status = st.sidebar.selectbox('Select Features',
                                       options=['Over all Quality',
                                                'Year Built',
                                                'Over All Condition',
                                                'Neighborhood',
                                                'Building Type',
                                                'Roof Style',
                                                'Basement Quality',
                                                'Electrical System',
                                                'Garage Cars',
                                                ])


fig = go.Figure()


if selected_status == 'Building Type':
    fig.add_trace(go.Bar(x=dataset.BldgType, y=dataset.SalePrice))
if selected_status == 'Over all Quality':
    fig.add_trace(go.Bar(x=dataset.OverallQual, y=dataset.SalePrice))
if selected_status == 'Year Built':
    fig.add_trace(go.Bar(x=dataset.YearBuilt, y=dataset.SalePrice))
if selected_status == 'Over All Condition':
    fig.add_trace(go.Bar(x=dataset.OverallCond, y=dataset.SalePrice))
if selected_status == 'Neighborhood':
    fig.add_trace(go.Bar(x=dataset.Neighborhood, y=dataset.SalePrice))
if selected_status == 'Roof Style':
    fig.add_trace(go.Bar(x=dataset.RoofStyle, y=dataset.SalePrice))
if selected_status == 'Basement Quality':
    fig.add_trace(go.Bar(x=dataset.BsmtQual, y=dataset.SalePrice))
if selected_status == 'Electrical System':
    fig.add_trace(go.Bar(x=dataset.Electrical, y=dataset.SalePrice))
if selected_status == 'Garage Cars':
    fig.add_trace(go.Bar(x=dataset.GarageCars, y=dataset.SalePrice))


st.plotly_chart(fig, use_container_width=True)