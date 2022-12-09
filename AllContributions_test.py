import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from function_list import quarter_to_date, date_to_datetime
#from csv_file_read import data
#print(data.head())
#print(type(data))
#print(data.columns)

#setting quarter for the date of viewing (this is used for slicing data table)
##make this into a function in the future
if datetime.now().month < 4:
    n = 1
if datetime.now().month < 7:
    n = 2
if datetime.now().month < 10:
    n = 3
if datetime.now().month < 13:
    n = 4
    
#setting current year for the date of viewing (this is used for slider range)
cy = datetime.now().year

#inserting slide bar to include other departments
st.write("Contributions")
add_selectbox = st.sidebar.image('pwga_logo.png')
add_selectbox = st.sidebar.selectbox(
    "Categories",
    ("Overview", "Contributions", "Claims", "Pension Plan", "Health Fund")
)

#inserting slider to select range of years
values = st.slider(
    'Select a range of years',
    2007, cy, (cy-4, cy))

#reading csv file
url='https://raw.githubusercontent.com/pwgajlee/streamlit_deploy_test/main/contributions_w.csv'
data = pd.read_csv(url)
url_alt = 'https://raw.githubusercontent.com/pwgajlee/streamlit_deploy_test/main/contributions_w_altair.csv'
data_alt = pd.read_csv(url_alt)

#formatting contribution $ from string to float
##make this into a function
data['ConsolidatedContribution']=data['ConsolidatedContribution'].str.replace(',','')
data.ConsolidatedContribution = data.ConsolidatedContribution.astype('float32')
data['HealthContribution']=data['HealthContribution'].str.replace(',','')
data.HealthContribution = data.HealthContribution.astype('float32')
data['PensionContribution']=data['PensionContribution'].str.replace(',','')
data.PensionContribution = data.PensionContribution.astype('float32')

data_alt['Amount']=data_alt['Amount'].str.replace(',','')
data_alt.Amount = data_alt.Amount.astype('float32')


#rename column names
data.columns = ['Quarter', 'Consolidated Contributions', 'Health Contributions', 'Pension Contributions']

#combine quarters and sum up the contributions
agg_functions = {'Consolidated Contributions': 'sum', 'Health Contributions': 'sum', 'Pension Contributions': 'sum'}
data_clean_index = data.groupby(data['Quarter'], as_index=False).aggregate(agg_functions)
data_clean = data.groupby(data['Quarter']).aggregate(agg_functions)

data_clean_alt = data_alt.groupby(['Contribution','Quarter'], as_index=False).agg('sum')

#find index for the slider values to be used to slice data table
range_1 = int(f'{values[0]}1')
index_1 = data_clean_index.index[data_clean_index['Quarter']==range_1][0]
##if last range is prior year, pull up to Q4, if current year, pull up to current quarter
if values[1] < datetime.now().year:
    range_2 = int(f'{values[1]}{4}')
    index_2 = data_clean_index.index[data_clean_index['Quarter']==range_2][0] + 1
elif values[1] == datetime.now().year:
    range_2 = int(f'{values[1]}{n}')
    index_2 = data_clean_index.index[data_clean_index['Quarter']==range_2][0] + 1

data_alt_low = data_clean_alt['Quarter']>=range_1
data_alt_high = data_clean_alt['Quarter']<=range_2
data_clean_alt = data_clean_alt[data_alt_low]
data_clean_alt = data_clean_alt[data_alt_high]

#selection box to choose what to show
all_symbols = data_clean_alt.Contribution.unique()
symbols = st.multiselect("Choose Contribution to visualize", all_symbols, all_symbols[:3])

#function to change amount to $0.0M
def change_to_mil(x):
    return '$' + (x/1000000).round(decimals=1).astype(str) + 'M'

#change from quarter to date (20071 -> 1/1/2007), creating new column for tooltip to show $0.0M
data_clean_alt['Period'] = data_clean_alt.apply(lambda row: quarter_to_date(row), axis=1)
data_clean_alt['Period']= pd.to_datetime(data_clean_alt['Period'])
data_clean_alt['Amount($M)'] = change_to_mil(data_clean_alt['Amount'])
data_clean_alt = data_clean_alt[['Contribution', 'Period', 'Quarter', 'Amount', 'Amount($M)']]
print(data_clean_alt.info())

#data sliced to match slider values
data_graph_sliced = data_clean[index_1:index_2]

#create new dataframe just for table presentation using sliced data table
data_table = data_graph_sliced.copy()

#formatting $ to millions, rounding to one decimal place, cleaning table, then transposing for viewing
data_table['Consolidated'] = change_to_mil(data_table['Consolidated Contributions'])
data_table['Health Fund'] = change_to_mil(data_table['Health Contributions'])
data_table['Pension Plan'] = change_to_mil(data_table['Pension Contributions'])
data_table = data_table.drop(columns=['Consolidated Contributions', 'Health Contributions', 'Pension Contributions'])
data_table = data_table.transpose()

#renaming columns of data table for graph
data_graph_sliced.columns = ['Consolidated', 'Health Fund', 'Pension Plan']

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
data_clean_alt = data_clean_alt[data_clean_alt.Contribution.isin(symbols)]
chart = get_chart(data_clean_alt)
st.altair_chart(
    chart,
    use_container_width=True
)

#line graph
#st.line_chart(data_graph_sliced)
#data table
st.table(data_table)

