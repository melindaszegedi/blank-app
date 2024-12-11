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
    st.title('Introduction')
   # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)





if selected == "Data preparation":
    st.title("Data preparation")
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

    
    #heatmap
    # Check if the dataset contains numeric data
    if df_rq.select_dtypes(include=[np.number]).empty:
        st.warning("The dataset does not contain numeric columns for correlation analysis.")
    else:
        # Calculate the correlation matrix
        correlation_matrix = df_rq.corr()

        # Create the heatmap
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(18, 12))
        sns.heatmap(
            correlation_matrix,
            annot=False,
            cmap="RdBu_r",
            linewidths=1,
            center=0,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            ax=ax,
        )

        # Annotate significant correlations
        for row in range(correlation_matrix.shape[0]):
            for col in range(correlation_matrix.shape[1]):
                correlation_value = correlation_matrix.iloc[row, col]
                if abs(correlation_value) >= 0.5 and row != col:
                    ax.text(
                        col + 0.5,
                        row + 0.5,
                        f"{correlation_value:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=12,
                        weight="bold",
                    )

        ax.set_title("Correlation Heatmap", fontsize=20, weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)
        st.pyplot(fig)
    



if selected == "Data exploration and cleaning":
    st.title("Data exploration and cleaning")
    with st.expander("##### Missing Data"):
        st.header("Missing Data")
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
    
        #missing data over period
        st.header("Further Missing Data Analysis")
        # Check for missing data
        missing_data = df_rq.isnull().sum()
        "Missing Data Count for each column:"
        st.write(missing_data[missing_data > 0])

        # Investigate missing values by period
        if 'PERIOD' in df_rq.columns:
            period_missing = df_rq.groupby('PERIOD').apply(lambda x: x.isnull().sum())

            # Plot heatmap
            st.write("Missing Data Across Examination Periods")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(period_missing, cmap="coolwarm", annot=True, fmt=".0f",
                        linewidths=0.5, annot_kws={"size": 8},
                        cbar_kws={'label': 'Count of Missing Values'})
            plt.title('Missing Data Across Examination Periods', fontsize=16)
            plt.xlabel('Variables', fontsize=12)
            plt.ylabel('Examination Period', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            st.pyplot(fig)
        else:
            st.write("The dataset does not contain a 'PERIOD' column.")
 
        st.write("## Imputation of missing values for different columns:")
        st.markdown("""
- **TOTCHOL**: The missing values of TOTCHOL are reasonable and can therefore be calculated with median imputation.   
- **BMI**: The missing values of BMI are reasonable and can therefore be calculated with median imputation.
- **HEARTRTE**: The missing values of HEARTRTE are reasonable and can therefore be calculated with median imputation.
- **BPMEDS**: The missing values of BPMEDS should be solved with categorical imputation. A new category, 'Unknown,' is created.
- **GLUCOSE**: There is a high amount of missing values in GLUCOSE. Therefore, K-Nearest Neighbors (KNN) is used for imputation.
""")
      # Imputation
        median_imputer = SimpleImputer(strategy='median')
        # Impute TOTCHOL
        df_rq['TOTCHOL'] = median_imputer.fit_transform(df_rq[['TOTCHOL']])
        # Impute BMI
        df_rq['BMI'] = median_imputer.fit_transform(df_rq[['BMI']])
        # Impute BPMEDS
        df_rq['BPMEDS'] = df_rq['BPMEDS'].fillna(-1)  # -1 as "Unknown"
        # Impute HEARTRTE
        df_rq['HEARTRTE'] = median_imputer.fit_transform(df_rq[['HEARTRTE']])
        st.write("Data After Imputation:")
        st.dataframe(df_rq.head())

      # Before imputation: Distribution of GLUCOSE
        col1, col2 = st.columns(2)
        with col1:
            st.write("Distribution of GLUCOSE Before Imputation:")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(df_rq['GLUCOSE'], bins=30, kde=True, ax=ax1)
            ax1.set_title('GLUCOSE Distribution Before Imputation')
            ax1.set_xlabel('GLUCOSE')
            ax1.set_ylabel('Frequency')
            st.pyplot(fig1)

        # KNN imputation for GLUCOSE (Only for GLUCOSE column)
        # Create a copy of the DataFrame to avoid modifying the original data
        df_rq = df_rq.copy()
        # Initialize KNN imputer
        imputer = KNNImputer(n_neighbors=5)
        df_rq['GLUCOSE'] = imputer.fit_transform(df_rq[['GLUCOSE']])
        with col2:
            st.write("Distribution of GLUCOSE Before Imputation:")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(df_rq['GLUCOSE'], bins=30, kde=True, ax=ax1)
            ax1.set_title('GLUCOSE Distribution Before Imputation')
            ax1.set_xlabel('GLUCOSE')
            ax1.set_ylabel('Frequency')
            st.pyplot(fig1)

    with st.expander("##### Identify, report, correct issues with erroneous data (if any)"):
        st.header("Identify, report, correct issues with erroneous data (if any)")
        # List of binary columns
        binary_columns = df_rq[['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS', 'ANYCHD', 'PERIOD']]

        # Streamlit UI for displaying value counts of binary columns
        st.write("### Value Counts for Binary Data Columns:")
        # display the columns in groups of three
        binary_column_names = binary_columns.columns
        for i in range(0, len(binary_column_names), 3):
            # Create columns for three tables side by side
            cols = st.columns(3)
    
            # Display the value counts in each column
            for col, binary_col in zip(cols, binary_column_names[i:i+3]):
                value_count = df_rq[binary_col].value_counts()
                with col:
                    st.write(f"#### {binary_col}:")
                    st.write(value_count)
        

    with st.expander("##### Identify and correct outliers"):
        st.header("Identify and correct outliers")
        # Boxplots to visualize outliers
        st.write("Interactive Variable Visualization")
        # Define color map for variables
        color_map = {
            'BMI': 'lightblue',
            'AGE': 'lightgreen',
            'TOTCHOL': 'salmon',
            'SYSBP': 'gold',
            'DIABP': 'orchid',
            'HEARTRTE': 'lightcoral',
            'GLUCOSE': 'lightpink'
        }
        # Select only numerical columns for visualization
        numerical_columns = ['BMI', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'HEARTRTE', 'GLUCOSE']
        df_rq_numeric = df_rq[numerical_columns]
        # Dropdown menu for selecting a variable
        selected_variable = st.selectbox("Select a variable to visualize:", numerical_columns)

        if selected_variable:
            # Plot the selected variable
            color = color_map.get(selected_variable, 'lightgray')
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.boxplot(data=df_rq_numeric, y=selected_variable, color=color, ax=ax)
            ax.set_title(f'{selected_variable} Boxplot', fontsize=16)
            ax.set_xlabel('Index')
            ax.set_ylabel(selected_variable)
            ax.grid(True)
            # Display the plot in Streamlit
            st.pyplot(fig)

        #impute outliers
        # Function to detect outliers using IQR
        def detect_outliers(df_rq, selected_columns):
            outliers = {}
            for col in selected_columns:
                Q1 = df_rq[col].quantile(0.2)
                Q3 = df_rq[col].quantile(0.8)
                IQR = Q3 - Q1

                # Define the outlier bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Detect outliers
                outliers[col] = df_rq[(df_rq[col] < lower_bound) | (df_rq[col] > upper_bound)]
            return outliers

        st.title("Outliers per column")
        # Step 2: Select columns for outlier detection
        selected_columns = ['BMI', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'HEARTRTE', 'GLUCOSE']
    
        if selected_columns:
            # Step 3: Detect outliers
            st.write("### Outlier Detection Results")
            outliers = detect_outliers(df_rq, selected_columns)
            total_outliers = 0

            # Display outliers for each column
            for col, outlier_data in outliers.items():
                num_outliers_col = len(outlier_data)
                total_outliers += num_outliers_col
                st.write(f"###### {col}: {num_outliers_col} outliers")

            st.write(f"###### Total Number of Outliers: {total_outliers}")

        #imputation of outliers:
        #determing outliers by IRQ:
        def detect_outliers(df_rq, column):
            Q1 = df_rq[column].quantile(0.2)
            Q3 = df_rq[column].quantile(0.8)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (df_rq[column] < lower_bound) | (df_rq[column] > upper_bound)

        st.title("Outlier Detection and KNN Imputation")
        st.dataframe(df_rq.describe())
        #Columns with outliers
        selected_columns = ['BMI', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'HEARTRTE', 'GLUCOSE']
        
        #Replace outliers with NaN
        df_imputed = df_rq.copy()
        for col in selected_columns:
            df_imputed[col] = df_rq[col].where(~detect_outliers(df, col), np.nan)

        st.write("### DataFrame with Outliers Replaced by NaN")            
        st.dataframe(df_imputed.describe())

        # Apply KNN Imputation to replace NaN values
        imputer = KNNImputer(n_neighbors=3)
        df_rqi = df_imputed.copy()
        df_rqi[selected_columns] = imputer.fit_transform(df_rqi[selected_columns])

        st.write("### DataFrame After KNN Imputation")
        st.dataframe(df_rqi.describe())

        
        #histogram data
        st.title("Histogram Plotter")
                # Define color map for variables
        # Select column for the histogram
        numerical_columnsi = ['BMI', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'HEARTRTE', 'GLUCOSE']
        df_rqi_numeric = df_rqi[numerical_columns]
        column_to_plot = st.selectbox("Select a column to plot:", numerical_columnsi)
        # Create the histogram
        bins = st.slider("Number of bins", min_value=10, max_value=50, value=30)
        fig, ax = plt.subplots()
        sns.histplot(df_rqi[column_to_plot], bins=bins, kde=True, ax=ax, color='blue')
        ax.set_title(f"Histogram of {column_to_plot}")
        ax.set_xlabel(column_to_plot)
        ax.set_ylabel("Frequency")
        # Display the plot
        st.pyplot(fig)



if selected == "Describe and Visualize the data":
    st.title("Describe and Visualize the data")
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    # Imputation missing values:
    df_rq = df_rq.copy()
    imputer = KNNImputer(n_neighbors=5)
    df_rq['GLUCOSE'] = imputer.fit_transform(df_rq[['GLUCOSE']])
    median_imputer = SimpleImputer(strategy='median')
    df_rq['TOTCHOL'] = median_imputer.fit_transform(df_rq[['TOTCHOL']])
    df_rq['BMI'] = median_imputer.fit_transform(df_rq[['BMI']])
    df_rq['BPMEDS'] = df_rq['BPMEDS'].fillna(-1)  # -1 as "Unknown"
    df_rq['HEARTRTE'] = median_imputer.fit_transform(df_rq[['HEARTRTE']])   
    #imputation outliers:
    def detect_outliers(df_rq, column):
        Q1 = df_rq[column].quantile(0.2)
        Q3 = df_rq[column].quantile(0.8)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df_rq[column] < lower_bound) | (df_rq[column] > upper_bound)
    selected_columns = ['BMI', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'HEARTRTE', 'GLUCOSE']
    df_imputed = df_rq.copy()
    for col in selected_columns:
        df_imputed[col] = df_rq[col].where(~detect_outliers(df, col), np.nan)
    imputer = KNNImputer(n_neighbors=3)
    df_rqi = df_imputed.copy()
    df_rqi[selected_columns] = imputer.fit_transform(df_rqi[selected_columns])

    #table with descriptive statistics 
    df_describe = df_rqi.describe()
    # Function to style entire rows based on condition
    def style_rows(row):
        return ["background-color: #f7aea8;" if row.name in ['count', 'std', '25%', '75%'] else "background-color: #ffd4d1;" for _ in row]
    # Apply style to specific rows by their index
    styled_df = df_describe.style.apply(style_rows, axis=1)
    st.header("Descriptive Statistics of final data")
    st.dataframe(styled_df)


    st.header("Proportion of each category")
    # Calculate proportions
    SEX_proportions = df_rqi['SEX'].value_counts(normalize=True) * 100
    CURSMOKE_proportions = df_rqi['CURSMOKE'].value_counts(normalize=True) * 100
    DIABETES_proportions = df_rqi['DIABETES'].value_counts(normalize=True) * 100
    BPMEDS_proportions = df_rqi['BPMEDS'].value_counts(normalize=True) * 100
    ANYCHD_proportions = df_rqi['ANYCHD'].value_counts(normalize=True) * 100
    PERIOD_proportions = df_rqi['PERIOD'].value_counts(normalize=True) * 100 
    
    # plot bar charts
    def plot_proportions(proportions, title, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(6, 4))
        proportions.plot(kind='bar', color=['lightskyblue', 'coral'], ax=ax)
        for i, v in enumerate(proportions):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        st.pyplot(fig)
    st.markdown("These graphs show the proportion of each category variables with perpcentages.")

    # Plot each proportion/percentage
    col1,col2,col3= st.columns(3)
    with col1:
        plot_proportions(SEX_proportions, "Sex Proportions", "Sex (1=male, 2=female)", "Percentage")
        plot_proportions(CURSMOKE_proportions, "Current Smoking Proportions", "Current Smoking (0=no, 1=yes)", "Percentage")
    with col2:
        plot_proportions(DIABETES_proportions, "Diabetes Proportions", "Diabetes (0=no, 1=yes)", "Percentage")
        plot_proportions(BPMEDS_proportions, "Blood Pressure Medication Proportions", "BPMEDS (0=no, 1=yes, -1=unknown)", "Percentage")
    with col3:
        plot_proportions(ANYCHD_proportions, "Any Coronary Heart Disease Proportions", "Any CHD (0=no, 1=yes)", "Percentage")
        plot_proportions(PERIOD_proportions, "Period Proportions", "Period", "Percentage")

    # Calculate the correlation matrix
    correlation_matrix = df_rqi.corr()
    # Create the heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap="RdBu_r",
        linewidths=1,
        center=0,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax,
    )

    # Annotate significant correlations
    for row in range(correlation_matrix.shape[0]):
        for col in range(correlation_matrix.shape[1]):
            correlation_value = correlation_matrix.iloc[row, col]
            if abs(correlation_value) >= 0.5 and row != col:
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    f"{correlation_value:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                    weight="bold",
                )

    ax.set_title("Correlation Heatmap", fontsize=20, weight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)
    st.pyplot(fig)

    # Visualize change in data over examination periods
    fig, ax = plt.subplots(figsize=(10, 6))
    df_rqi.groupby('PERIOD')[['AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'HEARTRTE', 'GLUCOSE', 'BMI']].mean().plot(
    marker='o', ax=ax
    )
    ax.set_title(
        'Average Values Over Examination Periods'
    )
    ax.set_xlabel('Examination Period')
    ax.set_ylabel('Average Value')
    ax.grid(True)
    st.pyplot(fig)

    # Streamlit Title
    st.header("BMI Calculator")
    st.write("*people < 18 years BMI calculation can be wrong")
    # Input fields for weight and height
    weight = st.number_input("Insert your weight in kilograms (kg):", min_value=0.0, format="%.2f")
    height = st.number_input("Insert your height in meters (m):", min_value=0.0, format="%.2f")
    # Calculate BMI if both inputs are valid
    if weight > 0 and height > 0:
        bmi = weight / (height * height)
        # Use columns for side-by-side display
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Your BMI:")
            st.write(round(bmi, 2))
        with col2:
            st.subheader("")
            if bmi < 18.5:
                st.write("Underweight")
            elif 18.5 <= bmi < 24.9:
                st.write("Normal weight")
            elif 25 <= bmi < 29.9:
                st.write("Overweight")
            elif 30 <= bmi < 34.9:
                st.write("Moderately Obese, consider losing weight")
            elif 35 <= bmi <39.9:
                st.write("Severely Obese, consider losing weight")
            else:
                st.write("Morbidly Obese, consider losing weight")
    else:
        st.write("Please enter valid values for weight and height.")





if selected == "Data Analysis":
    st.title("Describe and Visualize the data")
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    # Imputation missing values:
    df_rq = df_rq.copy()
    imputer = KNNImputer(n_neighbors=5)
    df_rq['GLUCOSE'] = imputer.fit_transform(df_rq[['GLUCOSE']])
    median_imputer = SimpleImputer(strategy='median')
    df_rq['TOTCHOL'] = median_imputer.fit_transform(df_rq[['TOTCHOL']])
    df_rq['BMI'] = median_imputer.fit_transform(df_rq[['BMI']])
    df_rq['BPMEDS'] = df_rq['BPMEDS'].fillna(-1)  # -1 as "Unknown"
    df_rq['HEARTRTE'] = median_imputer.fit_transform(df_rq[['HEARTRTE']])   
    #imputation outliers:
    def detect_outliers(df_rq, column):
        Q1 = df_rq[column].quantile(0.2)
        Q3 = df_rq[column].quantile(0.8)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df_rq[column] < lower_bound) | (df_rq[column] > upper_bound)
    selected_columns = ['BMI', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'HEARTRTE', 'GLUCOSE']
    df_imputed = df_rq.copy()
    for col in selected_columns:
        df_imputed[col] = df_rq[col].where(~detect_outliers(df, col), np.nan)
    imputer = KNNImputer(n_neighbors=3)
    df_rqi = df_imputed.copy()
    df_rqi[selected_columns] = imputer.fit_transform(df_rqi[selected_columns])






if selected == "Conclusion":
    st.title("Describe and Visualize the data")
    # Corrected URL for the raw CSV file
    url = 'https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/main/Framingham%20Dataset.csv'
    #allow all the columns to be visible
    pd.set_option('display.max_columns', None)
    # Read the CSV file from the URL
    df = pd.read_csv(url)
    #selection of relevant rows and columns for research question, put into new dataset
    df_rq=df[['BMI', 'AGE', 'SEX', 'TOTCHOL', 'SYSBP', 'DIABP', 'CURSMOKE','DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE','ANYCHD','PERIOD']]
    # Imputation missing values:
    df_rq = df_rq.copy()
    imputer = KNNImputer(n_neighbors=5)
    df_rq['GLUCOSE'] = imputer.fit_transform(df_rq[['GLUCOSE']])
    median_imputer = SimpleImputer(strategy='median')
    df_rq['TOTCHOL'] = median_imputer.fit_transform(df_rq[['TOTCHOL']])
    df_rq['BMI'] = median_imputer.fit_transform(df_rq[['BMI']])
    df_rq['BPMEDS'] = df_rq['BPMEDS'].fillna(-1)  # -1 as "Unknown"
    df_rq['HEARTRTE'] = median_imputer.fit_transform(df_rq[['HEARTRTE']])   
    #imputation outliers:
    def detect_outliers(df_rq, column):
        Q1 = df_rq[column].quantile(0.2)
        Q3 = df_rq[column].quantile(0.8)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df_rq[column] < lower_bound) | (df_rq[column] > upper_bound)
    selected_columns = ['BMI', 'AGE', 'TOTCHOL', 'SYSBP', 'DIABP', 'HEARTRTE', 'GLUCOSE']
    df_imputed = df_rq.copy()
    for col in selected_columns:
        df_imputed[col] = df_rq[col].where(~detect_outliers(df, col), np.nan)
    imputer = KNNImputer(n_neighbors=3)
    df_rqi = df_imputed.copy()
    df_rqi[selected_columns] = imputer.fit_transform(df_rqi[selected_columns])

    



