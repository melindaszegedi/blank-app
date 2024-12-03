import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import NearestNeighbors
import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import streamlit_option_menu
from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
    menu_title = "Project MAI3002",
    options = ["Introduction","Data preparation","Data exploration and cleaning","Describe and Visualize the data","Data Analysis", "Conclusion"],
    icons = ["chat-dots","list-task","search","bar-chart-line","graph-up", "folder"],
    menu_icon = "cast",
    default_index = 0,
    #orientation = "horizontal",
)
    
if selected == "Introduction":
    st.header('Introduction')
   # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    

if selected == "Data preparation":
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    st.write("### Summary Statistics of relevant rows")
    st.dataframe(df_rq.describe())


if selected == "Data exploration and cleaning":
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    
    missing_values_data = {
    "Column": ["Age", "Systolic Blood Pressure", "Diastolic Blood Pressure", "Cholesterol", "Smoking", "BMI" ],
    "Missing type": ["MCAR", "MAR or MCAR", "MAR or MCAR", "MAR or MNAR", "MNAR", "MAR"],
    "Reasoning": ["Generally easy to report, likely missing due to random error.", "Could be MAR if older or sicker participants avoid measurements, or MCAR if random errors occurred.", "Similar reasoning to sysBP.", "Could be MNAR if higher cholesterol individuals avoid reporting, or MAR if related to age/BMI.", "People may underreport smoking status due to social stigma.", "Missingness likely depends on variables like age or cholesterol, but not on BMI itself."],
    }
    # Create a DataFrame
    mdr = pd.DataFrame(missing_values_data)
    # Display the table
    st.write("### Interactive Table")
    st.dataframe(mdr)

    #Identify missing values in dataset
    missing_values = df_rq.isnull().sum().sum()
    # Display warning message if there are missing values
    if missing_values > 0:
        st.markdown(
         f"<span style='color:red; font-weight:bold;'>Warning: The dataset has {missing_values} missing values.</span>", 
        unsafe_allow_html=True
     )
    else:
        st.success("The dataset has no missing values.")


if selected == "Describe and Visualize the data":
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    st.write("Summary Statistics of relevant rows")
    st.dataframe(df_rq.describe())
   
if selected == "Data Analysis":
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    st.write("### Summary Statistics of relevant rows")
    st.dataframe(df_rq.describe())

if selected == "Conclusion":
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    st.write("### Summary Statistics of relevant rows")
    st.dataframe(df_rq.describe())

