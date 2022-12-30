import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from datetime import datetime

#re-arrange table for decision tree model for PENSION contributions
def data_expand_p(input_length: float, output_length: float, data: pd.DataFrame) -> pd.DataFrame:
    
    df = data.copy()
    
    i = 1
    while i < input_length:
        df[f'x_{i}'] = df['Pension Contributions'].shift(-i)
        i = i + 1
        
    j = 0
    while j < output_length:
        df[f'y_{j}'] = df['Pension Contributions'].shift(-output_length-j)
        j = j + 1
        
    df = df.dropna(axis=0)
    
    return df

#re-arrange table for decision tree model for HEALTH contributions
def data_expand_h(input_length: float, output_length: float, data: pd.DataFrame) -> pd.DataFrame:
    
    df = data.copy()
    
    i = 1
    while i < input_length:
        df[f'x_{i}'] = df['Health Contributions'].shift(-i)
        i = i + 1
        
    j = 0
    while j < output_length:
        df[f'y_{j}'] = df['Health Contributions'].shift(-output_length-j)
        j = j + 1
        
    df = df.dropna(axis=0)
    
    return df

#change quarter to month to prepare data to graph quarters on x-axis (20071 -> 2/1/2007)
def quarter_to_date(row):  
    if int(str(row['Quarter'])[-1])==1:
        return f'2/1/{str(row["Quarter"])[0:4]}'
    elif int(str(row['Quarter'])[-1])==2:
        return f'5/1/{str(row["Quarter"])[0:4]}'
    elif int(str(row['Quarter'])[-1])==3:
        return f'8/1/{str(row["Quarter"])[0:4]}'
    return f'11/1/{str(row["Quarter"])[0:4]}'

#change date type from str to datetime
def date_to_datetime(row):
    return datetime.strptime(row['date'], '%x')

#change amount to $0.0M
def change_to_mil(data):
    return '$' + (data/1000000).round(decimals=1).astype(str) + 'M'

#altair function to set graph parameters (AllContributions_test.py)
def get_chart(data):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="")
        .mark_line()
        .encode(
            x="yearquarter(date)",
            y="Amount",
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
            x="yearquarter(date)",
            y="Amount",
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("Quarter", title="Quarter"),
                alt.Tooltip("Amount", title="Amount"),
            ],
        )
        .add_selection(hover)
    )
    return (lines + points + tooltips).interactive()
