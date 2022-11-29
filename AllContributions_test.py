import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#reading data file (csv)
url = 'https://raw.githubusercontent.com/pwgajlee/streamlit_deploy_test/main/contributions_w.csv'
data = pd.read_csv(url, index_col=0)

#formatting $ from string to float
data['ConsolidatedContribution']=data['ConsolidatedContribution'].str.replace(',','')
data.ConsolidatedContribution = data.ConsolidatedContribution.astype('float32')
data['HealthContribution']=data['HealthContribution'].str.replace(',','')
data.HealthContribution = data.HealthContribution.astype('float32')
data['PensionContribution']=data['PensionContribution'].str.replace(',','')
data.PensionContribution = data.PensionContribution.astype('float32')

#rename column names
data.columns = ['Quarter', 'Consolidated Contributions', 'Health Contributions', 'Pension Contributions']

#combine quarters and sum contributions
agg_functions = {'Consolidated Contributions': 'sum', 'Health Contributions': 'sum', 'Pension Contributions': 'sum'}
data_clean = data.groupby(data['Quarter']).aggregate(agg_functions)

#create new dataframe just for table presentation
data_table = data_clean.copy()

#formatting $ to millions, rounding to one decimal place, and cleaning table
data_table['Consolidated'] = '$' + (data_table['Consolidated Contributions']/1000000).round(decimals=1).astype(str) + 'M'
data_table['Health Fund'] = '$' + (data_table['Health Contributions']/1000000).round(decimals=1).astype(str) + 'M'
data_table['Pension Plan'] = '$' + (data_table['Pension Contributions']/1000000).round(decimals=1).astype(str) + 'M'
data_table = data_table.drop(columns=['Consolidated Contributions', 'Health Contributions', 'Pension Contributions'])
data_table = data_table.transpose()

st.line_chart(data_clean)
st.table(data_table)

