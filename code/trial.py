import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re 
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from PIL import Image
import pickle
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
import statistics 

st.set_page_config(page_title="Hackathon", layout="wide")

# Read the dataset
df = pd.read_excel("D:/company/Buckman/Sample Data for shortlisting.xlsx")
cols = df.columns
print(len(cols))
df.to_csv("D:/company/Buckman/Data.csv", index=False)

def execute_query(query):
    try:
        # Execute the query
        cur.execute(query)

        # Fetch the results
        results = cur.fetchall()

        # Close the cursor and database connection
        cur.close()
        mydb.close()
        return results

    except Exception as e:
        st.error(f"Error executing query: {e}")
mydb=mysql.connector.connect(host="localhost",user="kanishka",password="1234",database="investment",auth_plugin='mysql_native_password')
cur = mydb.cursor()

q1=f"SELECT * FROM investment.data"
res=execute_query(q1)
# Define range columns
df=pd.DataFrame(res,columns=cols)
range_cols = [df['Investment Experience'], df['Percentage of Investment'], df['Return Earned']]
experience = df['Investment Experience'].unique()
investment_percent = df['Percentage of Investment'].unique()
returns = df['Return Earned'].unique()
# Function for label encoding
def label_encoding(column):
    unique_values = column.unique()
    encoding_map = {}
    encoding_index = 1
    for value in unique_values:
        encoding_map[value] = encoding_index
        encoding_index += 1
    encoded_column = column.map(encoding_map)
    return encoded_column, encoding_map

# Perform label encoding
encoded_experience, experience_encoding_map = label_encoding(df['Investment Experience'])
df['Investment Experience Encoded'] = encoded_experience
encoded_percent, percent_encoding_map = label_encoding(df['Percentage of Investment'])
df['Percentage of Investment Encoded'] = encoded_percent
encoded_returns, return_encoding_map = label_encoding(df['Return Earned'])
df['Return Earned Encoded'] = encoded_returns

def convert_to_numeric(income_str):
    # Remove non-numeric characters and commas
    income_str = re.sub(r'[^\d]+', ' ', income_str)
    
    # If the string contains "Above", set the value to the provided number
    if "Above" in income_str:
        income_value = int(income_str.split()[-1])
    else:
        # Split the string into individual components
        income_values = income_str.strip().split()
        
        if len(income_values) == 1:
            # If only one value is present, it's already numeric
            income_value = int(income_values[0])
        elif len(income_values) == 2:
            # If a range is provided, take the median value
            try:
                income_value = (int(income_values[0]) + int(income_values[1])) // 2
            except ValueError:
                # Handle cases where non-numeric characters are present
                return None
        else:
            # Invalid format, return None
            return None
    
    return income_value

df_sample=df
for i in range(len(df['Household Income'])):
    df.loc[i, 'Household Income'] = convert_to_numeric(df.loc[i, 'Household Income'])
for i in range(len(df['Household Income'])):
    df.loc[i, 'Household Income Average'] = df.loc[i, 'Household Income']
#robust_scaler = RobustScaler()
#household_income_df = df[['Household Income Average']]
#household_income_scaled = robust_scaler.fit_transform(household_income_df)
# Replace the original column with the scaled values
#df['Household Income Average'] = household_income_scaled

# Function to clean the data
def clean_data(df):
    df.drop_duplicates(inplace=True)
    numerical_cols = df.select_dtypes(include=['int', 'float']).columns
    df[numerical_cols] = df[numerical_cols].fillna(method='ffill')
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    return df

# Clean the dataset
df_cleaned = clean_data(df)
numerical_cols = df.select_dtypes(include=['int', 'float']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Function to display boxplots
def display_boxplots(df):
    numerical_cols = df.select_dtypes(include=['int', 'float']).columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=df[col], ax=ax)
        ax.set_title(f"Boxplot for {col}")
        st.pyplot(fig)

# Function to detect outliers using Z-score
def detect_outliers_zscore(df, threshold=3):
    outliers = []
    for col in numerical_cols:
        col_z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        col_outliers = df[col][col_z_scores > threshold]
        outliers.extend(col_outliers.index.tolist())
    df = df.drop(index=outliers)
    return df.loc[outliers]

# Function to detect outliers using IQR
def detect_outliers_iqr(df):
    outliers = []
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    res = {}
    for col in numerical_cols:
        col_outliers = df[col][(df[col] < (Q1[col] - 1.5 * IQR[col])) | (df[col] > (Q3[col] + 1.5 * IQR[col]))]
        outliers.extend(col_outliers.index.tolist())
        res[col] = col_outliers.index.tolist()
    df_cleaned = df.drop(index=outliers)
    return df_cleaned, res

# Define Streamlit tab1 function
def tab1():
    st.title('Preprocessing')
    st.markdown('Displaying the dataset')
    st.write(df)
    st.markdown('Labels in investment experience')
    st.write(experience)
    st.markdown('Labels in investment percentage')
    st.write(investment_percent)
    st.markdown('Labels in investment returns')
    st.write(returns)
    
    st.write("Encoded 'Investment Experience' column:")
    st.write(df[['Investment Experience', 'Investment Experience Encoded']])
    st.write("\nEncoded 'Percentage of Investment' column:")
    st.write(df[['Percentage of Investment', 'Percentage of Investment Encoded']])
    st.write("\nEncoded 'Return Earned' column:")
    st.write(df[['Return Earned', 'Return Earned Encoded']])
    
    st.write("\nInvestment Experience Label Encoding Map:")
    st.write(experience_encoding_map)
    st.write("\nPercentage of Investment Label Encoding Map:")
    st.write(percent_encoding_map)
    st.write("\nReturn Earned Label Encoding Map:")
    st.write(return_encoding_map)
    
    #st.markdown('after scaling household income')
    st.write(df['Household Income'])
    st.subheader('Cleaning the dataset')
    st.write(df_cleaned)
    st.subheader('Data Types of Columns')
    st.write(df.dtypes)
    st.subheader('Shape of Original Dataset')
    st.write(df.shape)
    st.subheader('Number of Null Values in Original Dataset')
    st.write(df.isna().sum())
    st.subheader('Shape of Cleaned Dataset')
    st.write(df_cleaned.shape)
    
    st.header("Outliers Detection")
    st.subheader('Boxplot to detect outliers')
    display_boxplots(df_cleaned)
    
    st.subheader("Detected Outliers using IQR:")
    df_cleaned_iqr, res_iqr = detect_outliers_iqr(df_cleaned)
    for col, col_outliers in res_iqr.items():
        st.subheader(f"Column: {col}")
        if not col_outliers:
            st.write("No outliers detected.")
        else:
            outlier_df = df_cleaned_iqr.loc[col_outliers]
            st.write(outlier_df)
    
    st.subheader("Detected Outliers using Z-score:")
    df_cleaned_zscore = detect_outliers_zscore(df_cleaned)
    st.write(df_cleaned_zscore)

def explore_employment_details():
    st.title('Exploring Employment Details')

    # Employment Roles
    st.header('Employment Roles')
    role_counts = df['Role'].value_counts()
    st.write(role_counts)

    # Visualization: Employment Roles Bar Chart
    st.subheader('Employment Roles Distribution')
    fig, ax = plt.subplots()
    sns.countplot(y='Role', data=df, palette='pastel', order=role_counts.index, ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Role')
    ax.set_title('Employment Roles Distribution')
    st.pyplot(fig)

    # Career Stages
    st.header('Career Stages')
    career_stage_counts = df['Age'].value_counts()
    st.write(career_stage_counts)

    # Visualization: Career Stages Bar Chart
    st.subheader('Career Stages Distribution')
    fig, ax = plt.subplots()
    sns.countplot(y='Age', data=df, palette='pastel', order=career_stage_counts.index, ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Career Stage')
    ax.set_title('Career Stages Distribution')
    st.pyplot(fig)

    # Income Brackets
    st.header('Income Brackets')
    income_bracket_counts = df['Household Income'].value_counts()
    st.write(income_bracket_counts)

    # Visualization: Income Brackets Bar Chart
    st.subheader('Income Brackets Distribution')
    fig, ax = plt.subplots()
    sns.countplot(y='Household Income', data=df, palette='pastel', order=income_bracket_counts.index, ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Income Bracket')
    ax.set_title('Income Brackets Distribution')
    st.pyplot(fig)
def investigate_investment_behavior():
    st.title('Investment Behavior Insights')

    # Percentage of household income invested
    st.header('Percentage of Household Income Invested')
    st.write(df['Percentage of Investment'].value_counts())

    # Visualization: Percentage of Household Income Invested Pie Chart
    st.subheader('Percentage of Household Income Invested Distribution (Pie Chart)')
    fig, ax = plt.subplots()
    df['Percentage of Investment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette('pastel'))
    ax.set_ylabel('')
    ax.set_title('Percentage of Household Income Invested Distribution')
    st.pyplot(fig)

    # Sources of awareness about investments
    st.header('Sources of Awareness About Investments')
    st.write(df['Source of Awareness about Investment'].value_counts())

    # Visualization: Sources of Awareness About Investments Bar Chart
    st.subheader('Sources of Awareness About Investments Distribution (Bar Chart)')
    fig, ax = plt.subplots()
    sns.countplot(y='Source of Awareness about Investment', data=df, palette='pastel', ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Source of Awareness')
    ax.set_title('Sources of Awareness About Investments Distribution')
    st.pyplot(fig)

    # Knowledge levels
    st.header('Knowledge Levels')
    st.write(df['Knowledge level about different investment product'].value_counts())

    # Visualization: Knowledge Levels Bar Chart
    st.subheader('Knowledge Levels Distribution (Bar Chart)')
    fig, ax = plt.subplots()
    sns.countplot(y='Knowledge level about different investment product', data=df, palette='pastel', ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Knowledge Level')
    ax.set_title('Knowledge Levels Distribution')
    st.pyplot(fig)

    # Investment influencers
    st.header('Investment Influencers')
    st.write(df['Investment Influencer'].value_counts())

    # Visualization: Investment Influencers Horizontal Bar Chart
    st.subheader('Investment Influencers Distribution (Horizontal Bar Chart)')
    fig, ax = plt.subplots()
    sns.countplot(y='Investment Influencer', data=df, palette='pastel', ax=ax)
    ax.set_xlabel('Count')
    ax.set_ylabel('Influencer')
    ax.set_title('Investment Influencers Distribution')
    st.pyplot(fig)

    # Risk levels
    st.header('Risk Levels')
    st.write(df['Risk Level'].value_counts())

    # Visualization: Risk Levels Line Chart (Men vs Women)
    st.subheader('Risk Levels Distribution (Line Chart)')
    fig, ax = plt.subplots()
    sns.lineplot(x='Return Earned', y='Risk Level', hue='Gender', data=df, palette='pastel', ax=ax)
    ax.set_xlabel('Return Earned')
    ax.set_ylabel('Risk Level')
    ax.set_title('Risk Levels Distribution (Men vs Women)')
    st.pyplot(fig)

    # Reasons for investment
    st.header('Reasons for Investment')
    st.write(df['Reason for Investment'].value_counts())

    # Visualization: Reasons for Investment pie Chart
    st.subheader('Reasons for Investment Distribution (Pie Chart)')
    reasons_counts = df['Reason for Investment'].value_counts()
    colors = sns.color_palette('Set3')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(reasons_counts, labels=reasons_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.set_title('Reasons for Investment Distribution')
    st.pyplot(fig)
   
def tab2():
    st.title('Distribution Analysis')
    Demographic_distribution= st.button("Demographic distribution")
    Employment_details = st.button("Employment Details")
    Investment_behavior=st.button("Investment Behavior")
    # Gender distribution
    col1, col2 = st.columns(2)
    
    # Gender distribution pie chart
    if Demographic_distribution:
        with col1:
            st.header('Gender Distribution (Pie Chart)')
            gender_counts = df['Gender'].value_counts()
            st.write(gender_counts)
            fig, ax = plt.subplots()
            ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
            ax.legend(gender_counts.index, loc='center left', bbox_to_anchor=(1, 0.5))
            st.pyplot(fig)

        # Gender distribution bar chart
        with col2:
            st.header('Gender Distribution (Bar Chart)')
            st.bar_chart(gender_counts)
        # Marital status distribution
        st.header('Marital Status Distribution')
        marital_counts = df['Marital Status'].value_counts()
        st.write(marital_counts)
        fig, ax = plt.subplots()
        sns.barplot(x=marital_counts.index, y=marital_counts.values, palette='pastel', ax=ax)
        ax.set_xlabel('Marital Status')
        ax.set_ylabel('Count')
        ax.set_title('Marital Status Distribution')
        st.pyplot(fig)
    
        # Age group distribution - Pie chart
        st.header('Age Group Distribution (Pie Chart)')
        age_counts = df['Age'].value_counts()
        st.write(age_counts)
        fig, ax = plt.subplots()
        ax.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
        ax.legend(age_counts.index, loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig)
    
        # Age group distribution - Bar chart
        st.header('Age Group Distribution (Bar Chart)')
        fig, ax = plt.subplots()
        sns.barplot(x=age_counts.index, y=age_counts.values, palette='pastel', ax=ax)
        ax.set_ylabel('Count')
        ax.set_xlabel('Age Group')
        ax.set_title('Age Group Distribution')
        ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels vertically
        st.pyplot(fig)
    elif Employment_details:
        explore_employment_details()
    else:
        investigate_investment_behavior()

def identify_factors(df):
    st.title('Identifying Factors for Best Investment Decision')

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Generate a correlation heatmap
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Analyze the correlation matrix
    st.subheader('Factors Correlated with Investment Decision')
    target_correlation = corr_matrix['Return Earned Encoded'].sort_values(ascending=False)
    st.write(target_correlation)     
def best_investment_identification_factor(df):
    non_numeric_cols = df.select_dtypes(exclude=['int', 'float']).columns
    label_encoder = LabelEncoder()
    # Convert non-numeric columns to numeric
    for col in non_numeric_cols:
        df[col] = label_encoder.fit_transform(df[col])
    
    # Define numeric columns
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    demographic_cols = ['Gender', 'Marital Status', 'Age']
    employment_cols = ['Role', 'Household Income Average']
    behavioral_cols = ['Knowledge level about different investment product', 'Investment Experience Encoded', 'Risk Level']
    investment_outcome_col = ['Return Earned Encoded']
    
    # Exclude non-numeric columns from the factors
    factors = demographic_cols + employment_cols + behavioral_cols + investment_outcome_col
    numeric_factors = [col for col in factors if col in numeric_cols]

    # Select only numeric columns for correlation analysis
    df_numeric = df[numeric_factors]

    # Calculate the correlation matrix
    correlation_matrix = df_numeric.corr()

    # Visualize correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)  # Display the plot in Streamlit

    # Print factors contributing to the best investment decision
    st.write("Factors contributing to the best investment decision:")
    st.write(correlation_matrix['Return Earned Encoded'].sort_values(ascending=False))

    # Determine which demographic, employment, and behavioral characteristics correlate with successful investment outcomes
    st.write("\nCorrelation of demographic, employment, and behavioral characteristics with investment outcomes:")
    correlation_columns = demographic_cols + employment_cols + behavioral_cols
    correlation_values = correlation_matrix['Return Earned Encoded'][correlation_columns]
    st.write(correlation_values)
    
    demographic_sum = correlation_matrix['Return Earned Encoded'][demographic_cols].sum()
    employment_sum = correlation_matrix['Return Earned Encoded'][employment_cols].sum()
    behavioral_sum = correlation_matrix['Return Earned Encoded'][behavioral_cols].sum()
    st.write("Sum of correlation values for each group:")
    st.write("Demographic Columns:", demographic_sum)
    st.write("Employment Columns:", employment_sum)
    st.write("Behavioral Columns:", behavioral_sum)

    # Plot the correlation of the sums using a bar chart
    sums = {'Demographic': demographic_sum, 'Employment': employment_sum, 'Behavioral': behavioral_sum}
    st.bar_chart(sums)
    # Find the maximum sum and its corresponding name
    max_sum = max(demographic_sum, employment_sum, behavioral_sum)
    max_sum_name = {'Demographic': demographic_sum, 'Employment': employment_sum, 'Behavioral': behavioral_sum}
    max_sum_name = max(max_sum_name, key=max_sum_name.get)

    # Print the maximum sum and its corresponding name with larger font size
    st.markdown("<h2 style='color:green;'>Maximum Sum and Its Corresponding Name:</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:green;'>Maximum Sum:</h3> <p style='font-size:20px;'>{max_sum}</p>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:green;'>Corresponding Name:</h3> <p style='font-size:20px;'>{max_sum_name}</p>", unsafe_allow_html=True)


    # Find the maximum correlation within demographic, employment, and behavioral groups
    max_demographic_corr = correlation_matrix['Return Earned Encoded'][demographic_cols].max()
    max_employment_corr = correlation_matrix['Return Earned Encoded'][employment_cols].max()
    max_behavioral_corr = correlation_matrix['Return Earned Encoded'][behavioral_cols].max()

    max_corr_data = {
        'Group': ['Demographic', 'Employment', 'Behavioral'],
        'Maximum Correlation': [max_demographic_corr, max_employment_corr, max_behavioral_corr],
        'Column Name': [correlation_matrix['Return Earned Encoded'][demographic_cols].idxmax(),
                        correlation_matrix['Return Earned Encoded'][employment_cols].idxmax(),
                        correlation_matrix['Return Earned Encoded'][behavioral_cols].idxmax()]
    }
    max_corr_df = pd.DataFrame(max_corr_data)

    # Print the DataFrame
    st.write("Maximum Correlations within Each Group:")
    st.write(max_corr_df)
def tab3():
    st.title('BEST INVESTMENT IDENTIFICATION FACTOR')
    st.write(df.columns)
    # Add content for Tab 3
    df_numeric=df[numerical_cols]
    identify_factors(df_numeric)
    best_investment_identification_factor(df)

def evaluation(y_test, y_pred):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write("\nConfusion Matrix:")
    st.write(conf_matrix)

    # Calculate precision, recall, and f1-score from the classification report
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    precision = classification_rep['macro avg']['precision']
    recall = classification_rep['macro avg']['recall']
    f1_score = classification_rep['macro avg']['f1-score']
    accuracy = accuracy_score(y_test, y_pred)
    # Display precision, recall, and f1-score
    st.write("\nPrecision:", precision)
    st.write("Recall:", recall)
    st.write("F1-score:", f1_score)
    st.write("Accuracy:", accuracy)
def feature_importance(decision_tree, X_train):
    # Get feature importances from the decision tree model
    importances = decision_tree.feature_importances_
    features = X_train.columns

    # Create a DataFrame to store feature importances
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    ax.set_title('Feature Importances')
    st.pyplot(fig)  
def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot Seaborn bar chart with varied color palette
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'], palette='viridis', ax=ax)
    # Add labels to your graph
    ax.set_title(model_type + ' Feature Importance')
    ax.set_xlabel('Feature Importance Score')
    ax.set_ylabel('Features')
    st.pyplot(fig)

def tab4():
    robust_scaler = RobustScaler()
    df_actual=df
    household_income_df = df[['Household Income']]
    household_income_scaled = robust_scaler.fit_transform(household_income_df)
    # Replace the original column with the scaled values
    df['Household Income'] = household_income_scaled
    if 'Household Income Average' in df.columns:
        df.drop(columns=['Household Income Average'], inplace=True)
    non_numeric_cols = df.select_dtypes(exclude=['int', 'float']).columns
    label_encoder = LabelEncoder()
    for col in non_numeric_cols:
        df[col] = label_encoder.fit_transform(df[col])
    # Filter columns by checking if their names end with "Encoded"
    columns_to_drop = [col for col in df.columns if col.endswith('Encoded')]
# Drop the selected columns

    df_filtered = df.drop(columns=columns_to_drop)
    df_sample=pd.read_excel("D:/company/Buckman/Sample Data for shortlisting.xlsx")
    st.write(df_filtered.head(20))
    # Define features (X) and target variable (y)
    X = df_filtered.drop(columns=['Return Earned'])
    y = df_filtered['Return Earned']

    st.header('DECISION TREE')
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # Coefficients after Ridge regularization
    coefficients = ridge.coef_
    st.markdown('Ridge coefficients')
    # Print coefficients for each variable
    cols=X_train.columns
    for i, var in enumerate(cols):
        st.write(f'{var}: {coefficients[i]}')
    # Initialize the decision tree classifier
    decision_tree = DecisionTreeClassifier(random_state=42)

    # Train the decision tree classifier
    decision_tree.fit(X_train, y_train)

    # Make predictions using decision tree
    y_pred_decision_tree = decision_tree.predict(X_test)

    evaluation(y_test, y_pred_decision_tree)
    # Get feature importances
    feature_importance(decision_tree, X_train)
    
    st.header('Random Forest')
    # Initialize the random forest classifier with bagging
    random_forest_default = RandomForestClassifier(random_state=42)

    # Train the Random Forest classifier with default hyperparameters
    random_forest_default.fit(X_train, y_train)

    # Make predictions using the default Random Forest model
    y_pred_default = random_forest_default.predict(X_test)

    # Evaluate the default Random Forest model
    st.write("\nRandom Forest with Default Hyperparameters:")
    evaluation(y_test, y_pred_default)

    # Get feature importances of the default Random Forest model
    feature_importance_default = random_forest_default.feature_importances_
    feature_names = X.columns

    # Plot feature importance for default Random Forest model
    plot_feature_importance(feature_importance_default, feature_names, "Default")
    st.subheader('Random Forest with GridSearch ')
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Initialize the GridSearchCV
    grid_search = GridSearchCV(estimator=random_forest_default, param_grid=param_grid, cv=5)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Get the best estimator and its parameters from GridSearchCV
    best_estimator_grid = grid_search.best_estimator_
    best_params_grid = grid_search.best_params_

    # Make predictions using the best estimator from GridSearchCV
    y_pred_grid = best_estimator_grid.predict(X_test)

    # Evaluate the model performance after GridSearchCV
    st.write("\nRandom Forest after GridSearchCV:")
    evaluation(y_test, y_pred_grid)

    # Get feature importances of the model after GridSearchCV
    feature_importance_grid = best_estimator_grid.feature_importances_

    # Plot feature importance for model after GridSearchCV
    plot_feature_importance(feature_importance_grid, feature_names, "After GridSearchCV")

    st.subheader('Random Forest with RandomizedSearch ')
    # Define the parameter grid for RandomizedSearchCV
    param_grid_rand = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Initialize RandomizedSearchCV with the parameter grid
    random_search = RandomizedSearchCV(estimator=random_forest_default, 
                                    param_distributions=param_grid_rand, 
                                    n_iter=100, 
                                    cv=5, 
                                    scoring='accuracy', 
                                    random_state=42, 
                                    n_jobs=-1)

    # Fit RandomizedSearchCV to the training data
    random_search.fit(X_train, y_train)

    # Get the best estimator and its parameters from RandomizedSearchCV
    best_estimator_rand = random_search.best_estimator_
    best_params_rand = random_search.best_params_

    # Make predictions using the best estimator from RandomizedSearchCV
    y_pred_rand = best_estimator_rand.predict(X_test)

    # Evaluate the model performance after RandomizedSearchCV
    st.write("\nRandom Forest after RandomizedSearchCV:")
    evaluation(y_test, y_pred_rand)

    # Get feature importances of the model after RandomizedSearchCV
    feature_importance_rand = best_estimator_rand.feature_importances_

    # Plot feature importance for model after RandomizedSearchCV
    plot_feature_importance(feature_importance_rand, feature_names, "After RandomizedSearchCV")
    st.header('Neural Network')
    # Initialize the neural network classifier
    neural_network = MLPClassifier(random_state=42)
    neural_network.fit(X_train, y_train)
    y_pred_neural_network = neural_network.predict(X_test)

    evaluation(y_test, y_pred_neural_network)
    
    st.subheader("\nNeural Network with adam:")
     # Initialize the neural network classifier with Adam optimizer
    neural_network_adam = MLPClassifier(solver='adam', random_state=42)
    neural_network_adam.fit(X_train, y_train)
    y_pred_neural_network_adam = neural_network_adam.predict(X_test)
    evaluation(y_test, y_pred_neural_network_adam)
    
    neural_network_sgd = MLPClassifier(solver='sgd', learning_rate='constant', learning_rate_init=0.01, momentum=0.9, random_state=42)
    neural_network_sgd.fit(X_train, y_train)
    y_pred_neural_network_sgd = neural_network_sgd.predict(X_test)
    st.write("\nNeural Network with sgd:")
    evaluation(y_test, y_pred_neural_network_sgd)
    
    st.subheader('Adaboost classifier')
    # Train the neural network classifier with AdaBoost optimizer
    neural_network_adaboost = AdaBoostClassifier(random_state=42)
    neural_network_adaboost.fit(X_train, y_train)
    y_pred_neural_network_adaboost = neural_network_adaboost.predict(X_test)
    evaluation(y_test, y_pred_neural_network_adaboost)

    st.subheader('Gradient Boost classifier')
    # Train the neural network classifier with Gradient Boosting optimizer
    neural_network_grad_boost = GradientBoostingClassifier(random_state=42)
    neural_network_grad_boost.fit(X_train, y_train)
    y_pred_neural_network_grad_boost = neural_network_grad_boost.predict(X_test)
    evaluation(y_test, y_pred_neural_network_grad_boost)
    
    neural_network_models = {
    'Neural Network (Adam)': accuracy_score(y_test, y_pred_neural_network_adam),
    'Neural Network (SGD)': accuracy_score(y_test, y_pred_neural_network_sgd),
    'Neural Network (AdaBoost)': accuracy_score(y_test, y_pred_neural_network_adaboost),
    'Neural Network (Gradient Boosting)': accuracy_score(y_test, y_pred_neural_network_grad_boost),
    'Neural Network (Default)': accuracy_score(y_test, y_pred_neural_network)
}

    best_neural_network_model = max(neural_network_models, key=neural_network_models.get)
    st.write('The best neural network model is ',best_neural_network_model)
    if best_neural_network_model == 'Neural Network (Adam)':
        y_pred_best_neural_network = y_pred_neural_network_adam
        hidden_layer_weights = neural_network_adam.coefs_[0]
    elif best_neural_network_model == 'Neural Network (SGD)':
        y_pred_best_neural_network = y_pred_neural_network_sgd
        hidden_layer_weights = neural_network_sgd.coefs_[0]
    elif best_neural_network_model == 'Neural Network (AdaBoost)':
        y_pred_best_neural_network = y_pred_neural_network_adaboost
        hidden_layer_weights = neural_network_adaboost.coefs_[0]
    elif best_neural_network_model == 'Neural Network (Gradient Boosting)':
        y_pred_best_neural_network = y_pred_neural_network_grad_boost
        hidden_layer_weights = neural_network_grad_boost.coefs_[0]
    else:
        y_pred_best_neural_network = y_pred_neural_network
        hidden_layer_weights = neural_network.coefs_[0]
    feature_importance_nn = np.abs(hidden_layer_weights).sum(axis=1)
    plot_feature_importance(feature_importance_nn, X_train.columns, "Neural Network")
    
    # Compare models based on evaluation metrics
    st.subheader("\nModel Comparison:")
    models_comparison = {
        'Decision Tree': accuracy_score(y_test, y_pred_decision_tree),
        'Random Forest with Bagging': accuracy_score(y_test, y_pred_default),
        'Neural Network': accuracy_score(y_test, y_pred_best_neural_network)
    }
    models = {
    'Decision Tree': decision_tree,
    'Random Forest with Bagging': random_forest_default,
    'Neural Network': neural_network
    }
    best_model_name = max(models_comparison, key=models_comparison.get) 
    best_model = models[best_model_name]
    st.markdown('The best model is ')
    st.write(best_model_name)
    st.subheader('Cross-validation of the best model')
    scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
        
        # Print the cross-validation scores
    st.write(f'{best_model} Cross-Validation Scores:', scores)
        
        # Calculate and print the mean and standard deviation of the cross-validation scores
    st.write(f'{best_model} Mean Accuracy:', np.mean(scores))
    st.write(f'{best_model} Standard Deviation of Accuracy:', np.std(scores))
    
    st.write("Best Model:", best_model)
    # Save the best model to a file
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    selected_labels = {}
    data=df_sample
    for col in df_sample.columns:
        if col != 'Return Earned' :
            selected_labels[col] = st.selectbox(f'Select label for {col}', data[col].unique())
    label_encoder = LabelEncoder()
    for i in range(len(data['Household Income'])):
        data.loc[i, 'Household Income'] = convert_to_numeric(data.loc[i, 'Household Income'])

    for col, label in selected_labels.items():
        data[col] = label_encoder.fit_transform(data[col])
    # Convert the selected label to numeric value using the convert_to_numeric function
    # Predict the "Return Earned" based on the selected labels
    if 'Household Income Average' in df.columns:
        df.drop(df['Household Income Average'])
    if 'Household Income Average' in data.columns:
        df.drop(df['Household Income Average'])
    X = data.drop(columns=['Return Earned'])
    prediction = best_model.predict(X)
    majority_prediction = statistics.mode(prediction)
    # Display the prediction result
    st.write('Predicted Return Earned:', prediction)
    st.write('The prediction done is ',majority_prediction)
    decoded_label = return_encoding_map.get(majority_prediction)
    # Display the original label
    st.write('Original Label for Majority Predicted Return Earned:', decoded_label)

def tab0():
   st.title('ABOUT PAGE')
   st.write('Preprocessing - Cleaning and checking for outliers and label encoding ')
   st.write('Feature detection with correlation matrix')
   st.write('Model building with Decision tree , RandomForest and Neural network ')
   st.write('Evaluation and choosing of the best model')
   st.write('Predict for a given user input')
def main():
    st.sidebar.title('Navigation')
    tab = st.sidebar.radio('Go to', ['About','Preprocessing', 'Exploration', 'Best Investment Decision Identification', 'Machine Learning'])

    if tab == 'Preprocessing':
        tab1()
    elif tab == 'Exploration':
        tab2()
    elif tab == 'Best Investment Decision Identification':
        tab3()
    elif tab == 'Machine Learning':
        tab4()
    elif tab == 'About':
        tab0()

if __name__ == "__main__":
    main()
