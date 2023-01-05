import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score, accuracy_score
from function_list import data_expand_p, data_expand_h, month_to_quarter, quarter_to_date, date_to_datetime
from datetime import datetime

#gathering data, both for table and graph (alt is for graph)
url='https://raw.githubusercontent.com/pwgajlee/streamlit_deploy_test/main/contributions_w.csv'
data = pd.read_csv(url)
url_alt = 'https://raw.githubusercontent.com/pwgajlee/streamlit_deploy_test/main/contributions_w_altair.csv'
data_alt = pd.read_csv(url_alt)

#find the quarter of the current month
n = month_to_quarter(datetime.now().month)

#find current year
y = datetime.now().year

#find current month
m = datetime.now().month

#function to change amount to $0.0M
def change_to_mil(x):
    return '$' + (x/1000000).round(decimals=1).astype(str) + 'M'

###################################
#SETTING UP PREDICTIONS


#change amounts to float-type
#change column names
data['ConsolidatedContribution']=data['ConsolidatedContribution'].str.replace(',','')
data.ConsolidatedContribution = data.ConsolidatedContribution.astype('float32')
data['HealthContribution']=data['HealthContribution'].str.replace(',','')
data.HealthContribution = data.HealthContribution.astype('float32')
data['PensionContribution']=data['PensionContribution'].str.replace(',','')
data.PensionContribution = data.PensionContribution.astype('float32')
data.columns = ['Quarter', 'Consolidated Contributions', 'Health Contributions', 'Pension Contributions']


#generating quarters and years for the prediction period (25 months)
pred_year = []
pred_month = []

pred_m = m + 1
while len(pred_month) <= 24:
    if pred_m <= 12:
        pred_month.append(pred_m)
        pred_m += 1
    else:
        pred_m = 1
        
last_month = pred_month[-1]
last_quarter = month_to_quarter(last_month)
#print(last_quarter)

while len(pred_year) <= 24:
    for i in range(0,12-m):
        pred_year.append(y)
    for i in range(0,12):
        pred_year.append(y+1)
    for i in range(0,25-len(pred_year)):
        pred_year.append(y+2)

last_year = pred_year[-1]
#print(last_year)


#PREDICTION MODEL

##separating pension data for prediction model
##set train and test set
data_p = data.drop(columns=['Health Contributions','Consolidated Contributions'])
data_p = data_expand_p(25, 25, data_p)
X_cols_p = [col for col in data_p.columns if col.startswith('x')]
X_cols_p.insert(0, 'Pension Contributions')
y_cols_p = [col for col in data_p.columns if col.startswith('y')]
X_train_p = data_p[X_cols_p][:-35].values
y_train_p = data_p[y_cols_p][:-35].values
X_test_p = data_p[X_cols_p][-35:].values
y_test_p = data_p[y_cols_p][-35:].values

##predict pension with prediction model
dt_p = DecisionTreeRegressor(random_state=1)
dt_p.fit(X_train_p, y_train_p)
dt_p_pred = dt_p.predict(X_test_p)

##fill in period for the prediction using already generated values above
##clean up prediction data for future use
pred_p = pd.DataFrame(dt_p_pred[34])
pred_p['Year']=pred_year
pred_p['Month']=pred_month
pred_p.columns=['Pension Contributions', 'Year', 'Month']
pred_quarter =[]
for i in pred_p['Month']:
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
pred_p['Quarter1']=pred_quarter
pred_p['Quarter']=(pred_p['Year'].astype(str)+pred_p['Quarter1'].astype(str)).astype(int)
pred_p=pred_p.drop(columns=['Year', 'Month', 'Quarter1'])
pred_p=pred_p[['Quarter','Pension Contributions']]
#print(pred_p)


##separating health data for prediction model
##set train and test set
data_h = data.drop(columns=['Pension Contributions','Consolidated Contributions'])
data_h = data_expand_h(25, 25, data_h)
X_cols_h = [col for col in data_h.columns if col.startswith('x')]
X_cols_h.insert(0, 'Health Contributions')
y_cols_h = [col for col in data_h.columns if col.startswith('y')]
X_train_h = data_h[X_cols_h][:-35].values
y_train_h = data_h[y_cols_h][:-35].values
X_test_h = data_h[X_cols_h][-35:].values
y_test_h = data_h[y_cols_h][-35:].values

##predict health with prediction model
dt_h = DecisionTreeRegressor(random_state=1)
dt_h.fit(X_train_h, y_train_h)
dt_h_pred = dt_h.predict(X_test_h)

##fill in period for the prediction using already generated values above
##clean up prediction data for future use
pred_h = pd.DataFrame(dt_h_pred[34])
pred_h['Year']=pred_year
pred_h['Month']=pred_month
pred_h.columns=['Health Contributions', 'Year', 'Month']
pred_quarter =[]
for i in pred_h['Month']:
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
pred_h['Quarter1']=pred_quarter
pred_h['Quarter']=(pred_h['Year'].astype(str)+pred_h['Quarter1'].astype(str)).astype(int)
pred_h=pred_h.drop(columns=['Year', 'Month', 'Quarter1'])
pred_h=pred_h[['Quarter','Health Contributions']]
#print(pred_h)


##add pension and health predictions to get consolidated prediction
pred_c = pd.DataFrame(dt_p_pred[34])
pred_c.columns=['Consolidated Contributions']
pred_c['Quarter']=pred_p['Quarter']
pred_c['Consolidated Contributions']=pred_p['Pension Contributions']+pred_h['Health Contributions']
#print(pred_c)


##combine all predictions into one table
##this is for table display and indexing
pred_all = pd.concat([pred_p, pred_h['Health Contributions'], pred_c['Consolidated Contributions']], axis=1)
pred_all_index = pred_all.groupby(['Quarter'], as_index=False).agg('sum') ###this one has index column in the front
pred_all = pred_all.groupby(['Quarter']).agg('sum') ###this one does not have index column, starts with quarter column
#print(pred_all)
#print(pred_all_index)


##generating separate table for graph (altair has different formatting requirements)
##pension table for graph
pred_p_alt = pd.DataFrame(dt_p_pred[34])
pred_p_alt['Pension Contributions'] = 'Pension Plan'
pred_p_alt.columns = ['Amount', 'Contribution']
pred_p_alt['Quarter']=pred_p['Quarter']
pred_p_alt=pred_p_alt[['Contribution', 'Quarter', 'Amount']]
#print(pred_p_alt)

##health table for graph
pred_h_alt = pd.DataFrame(dt_h_pred[34])
pred_h_alt['Health Contributions'] = 'Health Fund'
pred_h_alt.columns = ['Amount', 'Contribution']
pred_h_alt['Quarter']=pred_h['Quarter']
pred_h_alt=pred_h_alt[['Contribution', 'Quarter', 'Amount']]
#print(pred_h_alt)

##combine pension and health in preparation for consolidated table for graph
prediction=[pred_p_alt, pred_h_alt]
contrib_prediction=pd.concat(prediction)
#print(contrib_prediction)

##add the values under same quarter to get consolidated sum
pred_c_alt=contrib_prediction.groupby(['Quarter'], as_index=False).agg('sum')
pred_c_alt['Contribution']='Consolidated'
pred_c_alt=pred_c_alt[['Contribution', 'Quarter', 'Amount']]
#print(pred_c_alt)

##combine all predictions in one table for graph
prediction=[pred_p_alt, pred_h_alt, pred_c_alt]
pred_all_alt=pd.concat(prediction)
pred_all_alt=pred_all_alt.groupby(['Contribution', 'Quarter'], as_index=False).agg('sum')
#print(pred_all_alt)


#PREDICTION TABLES COMPLETE
###################################


#inserting slide bar to include other departments
#insert company logo
st.write("Contributions")
add_selectbox = st.sidebar.image('pwga_logo.png')
add_selectbox = st.sidebar.selectbox(
    "Categories",
    ("Overview", "Finance", "Pension", "Health", "Contributions", "Claims")
)

#inserting slider to select range of years
values = st.slider(
    'Select a range of years',
    2007, y+2, (y-4, y))


#actual data for graph
#change amount to float and combine with prediction table
data_alt['Amount']=data_alt['Amount'].str.replace(',','')
data_alt.Amount = data_alt.Amount.astype('float32')
data_alt=data_alt.groupby(['Contribution', 'Quarter'], as_index=False).agg('sum')
data_alt=pd.concat([data_alt, pred_all_alt]).reset_index()
data_alt=data_alt.drop(columns='index')
#print(data_alt)

#data for table
data_index = data.groupby(data['Quarter'], as_index=False).agg('sum')
data_index=pd.concat([data_index, pred_all_index]).reset_index()
data_index=data_index.drop(columns='index')
data = data.groupby(data['Quarter']).agg('sum')
data=pd.concat([data, pred_all])
#print(data_index)
#print(data)


#find index for the slider values to be used to slice data table
range_1 = int(f'{values[0]}1')
index_1 = data_index.index[data_index['Quarter']==range_1][0]
##if last range is prior year, pull up to Q4, if current year, pull up to current quarter, if future year, pull up to last pred quarter
if values[1] < y:
    range_2 = int(f'{values[1]}{4}')
    index_2 = data_index.index[data_index['Quarter']==range_2][0] + 1
elif values[1] == y:
    range_2 = int(f'{values[1]}{n}')
    index_2 = data_index.index[data_index['Quarter']==range_2][0] + 1
elif values[1] == y+1:
    range_2 = int(f'{values[1]}{4}')
    index_2 = data_index.index[data_index['Quarter']==range_2][0] + 1
elif values[1] == last_year:
    range_2 = int(f'{values[1]}{last_quarter}')
    index_2 = data_index.index[data_index['Quarter']==range_2][0] + 1

data_alt_low = data_alt['Quarter']>=range_1
data_alt_high = data_alt['Quarter']<=range_2
data_alt = data_alt[data_alt_low]
data_alt = data_alt[data_alt_high]

#selection box to choose what to show
all_symbols = data_alt.Contribution.unique()
symbols = st.multiselect("Choose Contribution to visualize", all_symbols, all_symbols[:3])

#change from quarter to date (20071 -> 1/1/2007), creating new column for tooltip to show $0.0M
data_alt['Period'] = data_alt.apply(lambda row: quarter_to_date(row), axis=1)
data_alt['Period']= pd.to_datetime(data_alt['Period'])
data_alt['Amount($M)'] = change_to_mil(data_alt['Amount'])
data_alt = data_alt[['Contribution', 'Period', 'Quarter', 'Amount', 'Amount($M)']]
#print(data_alt)

#data sliced to match slider values
data_sliced = data[index_1:index_2]

#create new dataframe just for table presentation using sliced data table
data_table = data_sliced.copy()


#formatting $ to millions, rounding to one decimal place, cleaning table, then transposing for viewing
data_table['Consolidated'] = change_to_mil(data_table['Consolidated Contributions'])
data_table['Health Fund'] = change_to_mil(data_table['Health Contributions'])
data_table['Pension Plan'] = change_to_mil(data_table['Pension Contributions'])
data_table = data_table.drop(columns=['Consolidated Contributions', 'Health Contributions', 'Pension Contributions'])
data_table = data_table.transpose()



#graph altair chart
def get_chart(data):
    hover = alt.selection_single(
        fields=["Period"],
        nearest=True,
        on="mouseover",
        empty="none",
    )
        
    lines = (
        alt.Chart(data, title="")
        .mark_line()
        .encode(
            x="yearquarter(Period)",
            y=alt.Y("Amount", axis=alt.Axis(format='$~s')),
            color="Contribution",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearquarter(Period)",
            y="Amount",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("yearquarter(Period)", title="Period"),
                alt.Tooltip("Amount($M)", title="Amount"),
            ],
        )
        .add_selection(hover)
    )

    return lines + points + tooltips

#altair interactive line graph
data_alt = data_alt[data_alt.Contribution.isin(symbols)]
chart = get_chart(data_alt)
st.altair_chart(
    chart,
    use_container_width=True
)

#display data table
st.table(data_table)
