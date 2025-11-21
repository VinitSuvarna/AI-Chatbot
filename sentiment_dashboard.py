import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading and Cleaning (optimized for Streamlit) ---
# Use st.cache_data to cache the data loading. This means it only runs once
# when the app starts or when the data file changes, making the app faster.
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('customer_interaction.csv')

    # 1. Convert 'Event Timestamp' to proper datetime objects and drop NaT
    # This handles inconsistent formats by coercing errors to NaT and then dropping those rows.
    df['Event Timestamp'] = pd.to_datetime(df['Event Timestamp'], errors='coerce')
    df.dropna(subset=['Event Timestamp'], inplace=True)

    # 2. Ensure 'Response Time (s)' is numeric and drop NaNs
    # Corrected column name to 'Response Time (s)'
    df['Response Time (s)'] = pd.to_numeric(df['Response Time (s)'], errors='coerce')
    df.dropna(subset=['Response Time (s)'], inplace=True)

    # 3. Handle Missing Values for 'Industry' and 'Customer Segment'
    # Fill NaN values with 'Unknown' to retain rows
    df['Industry'].fillna('Unknown', inplace=True)
    df['Customer Segment'].fillna('Unknown', inplace=True)

    # 4. Drop rows with missing 'Sentiment Score'
    # Sentiment Score is crucial; dropping NaNs is generally preferred here.
    df.dropna(subset=['Sentiment Score'], inplace=True)

    # 5. Convert 'User Name' and 'Record ID' to string type
    # Corrected 'User ID' to 'User Name' based on your provided columns.
    # Corrected 'Ticket ID' to 'Record ID' based on your provided columns.
    df['User Name'] = df['User Name'].astype(str)
    df['Record ID'] = df['Record ID'].astype(str) # This is your primary ID column

    # 6. Convert 'Department' column to string type for sorting in selectbox
    # Fixes TypeError: '<' not supported between instances of 'float' and 'str'
    df['Department'] = df['Department'].astype(str)

    return df

# Load and clean the data when the app starts
df = load_and_clean_data()

# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Use wide layout for better chart display
st.title("Customer Interaction Performance Dashboard")

# Sidebar for filters
st.sidebar.header("Filter Options")

# Department selection
# Now 'df['Department'].unique().tolist()' will only contain strings, so sorting works.
selected_department = st.sidebar.selectbox(
    "Select Department",
    options=['All Departments'] + sorted(df['Department'].unique().tolist()) # Add 'All' option
)

st.sidebar.write("---") # Separator in the sidebar

# Chart type selection
chart_type = st.sidebar.radio(
    "Choose metric for visualization:",
    ('Sentiment Distribution', 'Average Response Time')
)

# --- Main Content Area based on selections ---
if selected_department != 'All Departments':
    filtered_df = df[df['Department'] == selected_department]
    st.header(f"Analysis for Department: **{selected_department}**")
else:
    filtered_df = df
    st.header("Analysis Across All Departments")


# Display Sentiment Distribution Chart
if chart_type == 'Sentiment Distribution':
    st.subheader("Sentiment Score Distribution")
    if selected_department == 'All Departments':
        # Violin plot for all departments (shows distribution and density)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.violinplot(x='Department', y='Sentiment Score', data=filtered_df, ax=ax, inner='quartile')
        plt.xticks(rotation=45, ha='right') # Rotate department names for readability
        plt.title('Sentiment Score Distribution Across All Departments')
        plt.xlabel('Department')
        plt.ylabel('Sentiment Score')
        st.pyplot(fig) # Display the matplotlib plot in Streamlit
    else:
        # Histogram/KDE for a single selected department
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(filtered_df['Sentiment Score'], kde=True, ax=ax, bins=10)
        plt.title(f'Sentiment Score Distribution for {selected_department}')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        st.pyplot(fig)

# Display Average Response Time Comparison Chart
elif chart_type == 'Average Response Time':
    st.subheader("Average Response Time by Department")
    # Calculate average response time, ensuring correct column name 'Response Time (s)'
    avg_response_time = filtered_df.groupby('Department')['Response Time (s)'].mean().sort_values(ascending=False)

    # Streamlit's built-in bar_chart for quick visualization
    st.bar_chart(avg_response_time)
    st.write("Units: Seconds") # Inform the user about the units


st.markdown("---") # Horizontal line for visual separation
st.caption("Data last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
st.caption("Built with Streamlit")