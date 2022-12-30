import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score, accuracy_score
from function_list import data_expand_p, data_expand_h
from datetime import datetime


url='https://raw.githubusercontent.com/pwgajlee/streamlit_deploy_test/main/contributions_w.csv'
data = pd.read_csv(url)
data['ConsolidatedContribution']=data['ConsolidatedContribution'].str.replace(',','')
data.ConsolidatedContribution = data.ConsolidatedContribution.astype('float32')
data['HealthContribution']=data['HealthContribution'].str.replace(',','')
data.HealthContribution = data.HealthContribution.astype('float32')
data['PensionContribution']=data['PensionContribution'].str.replace(',','')
data.PensionContribution = data.PensionContribution.astype('float32')
data.columns = ['Quarter', 'Consolidated Contributions', 'Health Contributions', 'Pension Contributions']



if datetime.now().month < 4:
    n = 1
elif datetime.now().month < 7:
    n = 2
elif datetime.now().month < 10:
    n = 3
elif datetime.now().month < 13:
    n = 4
    
pred_year = []
pred_month = []

begin_year = datetime.now().year
if n==4:
    for i in range(0,2):
        begin_year += 1
        for j in range(1,13):
            pred_q = int(f'{begin_year}')
            pred_year.append(pred_q)
    pred_year.append(int(f'{begin_year+1}'))


begin_year = datetime.now().year
if n==4:
    for i in range(0,2):
        begin_year += 1
        for j in range(1,13):
            pred_m = int(f'{j}')
            pred_month.append(pred_m)
    pred_month.append(1)

data_p = data.drop(columns=['Health Contributions','Consolidated Contributions'])
data_p = data_expand_p(25, 25, data_p)

data_h = data.drop(columns=['Pension Contributions','Consolidated Contributions'])
data_h = data_expand_h(25, 25, data_h)

X_cols_p = [col for col in data_p.columns if col.startswith('x')]
X_cols_p.insert(0, 'Pension Contributions')
y_cols_p = [col for col in data_p.columns if col.startswith('y')]
X_train_p = data_p[X_cols_p][:-35].values
y_train_p = data_p[y_cols_p][:-35].values
X_test_p = data_p[X_cols_p][-35:].values
y_test_p = data_p[y_cols_p][-35:].values

X_cols_h = [col for col in data_h.columns if col.startswith('x')]
X_cols_h.insert(0, 'Health Contributions')
y_cols_h = [col for col in data_h.columns if col.startswith('y')]
X_train_h = data_h[X_cols_h][:-35].values
y_train_h = data_h[y_cols_h][:-35].values
X_test_h = data_h[X_cols_h][-35:].values
y_test_h = data_h[y_cols_h][-35:].values
    
dt_p = DecisionTreeRegressor(random_state=1)
dt_p.fit(X_train_p, y_train_p)
dt_p_pred = dt_p.predict(X_test_p)

dt_h = DecisionTreeRegressor(random_state=1)
dt_h.fit(X_train_h, y_train_h)
dt_h_pred = dt_h.predict(X_test_h)

dt_p_pred_g = pd.DataFrame(dt_p_pred[34])
dt_p_pred_g['Year']=pred_year
dt_p_pred_g['Month']=pred_month
dt_p_pred_g.columns=['Pension Prediction', 'Year', 'Month']
pred_quarter =[]
for i in dt_p_pred_g['Month']:
    if 0<i<4:
        pred_q = 1
        pred_quarter.append(pred_q)
    elif 3<i<7:
        pred_q = 2
        pred_quarter.append(pred_q)
    elif 6<i<10:
        pred_q = 3
        pred_quarter.append(pred_q)
    elif 9<i<13:
        pred_q = 4
        pred_quarter.append(pred_q)
dt_p_pred_g['Quarter1']=pred_quarter
dt_p_pred_g['Quarter']=(dt_p_pred_g['Year'].astype(str)+dt_p_pred_g['Quarter1'].astype(str)).astype(int)
dt_p_pred_g=dt_p_pred_g.drop(columns=['Year', 'Month', 'Quarter1'])
dt_p_pred_g=dt_p_pred_g[['Quarter','Pension Prediction']]

dt_h_pred_g = pd.DataFrame(dt_h_pred[34])
dt_h_pred_g['Year']=pred_year
dt_h_pred_g['Month']=pred_month
dt_h_pred_g.columns=['Health Prediction', 'Year', 'Month']
pred_quarter =[]
for i in dt_h_pred_g['Month']:
    if 0<i<4:
        pred_q = 1
        pred_quarter.append(pred_q)
    elif 3<i<7:
        pred_q = 2
        pred_quarter.append(pred_q)
    elif 6<i<10:
        pred_q = 3
        pred_quarter.append(pred_q)
    elif 9<i<13:
        pred_q = 4
        pred_quarter.append(pred_q)
dt_h_pred_g['Quarter1']=pred_quarter
dt_h_pred_g['Quarter']=(dt_h_pred_g['Year'].astype(str)+dt_h_pred_g['Quarter1'].astype(str)).astype(int)
dt_h_pred_g=dt_h_pred_g.drop(columns=['Year', 'Month', 'Quarter1'])
dt_h_pred_g=dt_h_pred_g[['Quarter','Health Prediction']]

dt_c_pred_g = pd.DataFrame(dt_p_pred[34])
dt_c_pred_g.columns=['Consolidated Prediction']
dt_c_pred_g['Quarter']=dt_p_pred_g['Quarter']
dt_c_pred_g['Consolidated Prediction']=dt_p_pred_g['Pension Prediction']+dt_h_pred_g['Health Prediction']

dt_pred_g = pd.concat([dt_p_pred_g, dt_h_pred_g['Health Prediction'], dt_c_pred_g['Consolidated Prediction']], axis=1)
dt_pred_g_index = dt_pred_g.groupby(['Quarter']).agg('sum')
dt_pred_g = dt_pred_g.groupby(['Quarter']).agg('sum')




dt_p_pred = pd.DataFrame(dt_p_pred[34])
dt_p_pred['Pension Contributions'] = 'Pension Prediction'
dt_p_pred.columns = ['Amount', 'Contribution']
dt_p_pred['Quarter']=''
dt_p_pred['Year']=pred_year
dt_p_pred['Month']=pred_month
dt_p_pred=dt_p_pred[['Contribution', 'Year', 'Month', 'Quarter', 'Amount']]

dt_h_pred = pd.DataFrame(dt_h_pred[34])
dt_h_pred['Health Contributions'] = 'Health Prediction'
dt_h_pred.columns = ['Amount', 'Contribution']
dt_h_pred['Quarter']=''
dt_h_pred['Year']=pred_year
dt_h_pred['Month']=pred_month
dt_h_pred=dt_h_pred[['Contribution', 'Year', 'Month', 'Quarter', 'Amount']]

prediction=[dt_p_pred, dt_h_pred]
contrib_prediction=pd.concat(prediction)

consolidated=contrib_prediction.groupby(['Year','Month'], as_index=False).agg('sum')
consolidated['Contribution']='Consolidated Prediction'
consolidated=consolidated[['Contribution', 'Year', 'Month', 'Amount']]

prediction=[dt_p_pred, dt_h_pred, consolidated]
contrib_prediction=pd.concat(prediction)

pred_quarter =[]
for i in contrib_prediction['Month']:
    if 0<i<4:
        pred_q = 1
        pred_quarter.append(pred_q)
    elif 3<i<7:
        pred_q = 2
        pred_quarter.append(pred_q)
    elif 6<i<10:
        pred_q = 3
        pred_quarter.append(pred_q)
    elif 9<i<13:
        pred_q = 4
        pred_quarter.append(pred_q)
contrib_prediction['Quarter1']=pred_quarter
contrib_prediction['Quarter']=(contrib_prediction['Year'].astype(str)+contrib_prediction['Quarter1'].astype(str)).astype(int)
contrib_prediction=contrib_prediction.drop(columns=['Year', 'Month', 'Quarter1'])
contrib_prediction=contrib_prediction[['Contribution','Quarter', 'Amount']]
contrib_prediction=contrib_prediction.groupby(['Contribution', 'Quarter']).agg('sum')

