import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data
file_path = "EComm_data.csv"
# Allow users to upload their own CSV file
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    data = load_data(file_path)
    st.info("Using preloaded dataset.")

# Display dataset preview
st.title("Streamlit App for Data Analysis")
st.write("### Dataset Preview")
st.dataframe(data.head())


#####  DATA CLEANING #####

st.write("### Data Cleaning")
st.write(f"Missing values per column:\n{data.isnull().sum()}")
data.fillna(method="ffill", inplace=True)
data.drop_duplicates(inplace=True)

columns = data.columns.tolist()
selected_columns = st.sidebar.multiselect("Select Columns to Display", columns, default=columns)
filtered_data = data[selected_columns]
st.write("### Filtered Dataset")
st.dataframe(filtered_data)

for col in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

target_column = st.sidebar.selectbox("Select Target Column", data.columns)
feature_columns = [col for col in data.columns if col != target_column]


#####  MODEL #####
X = data[feature_columns]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "SVM": SVR(),
    "XGBoost": xgb.XGBRegressor()
}

st.write("### Model Performance Metrics")
metrics = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    metrics[name] = {"MSE": mse, "RMSE": rmse, "R2 Score": r2}

metrics_df = pd.DataFrame(metrics).T
st.dataframe(metrics_df)

st.write("""
**Insights:**
- **MSE (Mean Squared Error)**: Measures the average squared difference between actual and predicted values. Lower values indicate better performance.
- **RMSE (Root Mean Squared Error)**: The square root of MSE, representing the standard deviation of residuals. Lower values are better.
- **R2 Score**: A measure of how well the model explains the variance in the target variable. Values closer to 1 indicate a better fit.
""")


#####  VISUALIZATIONS #####
st.sidebar.header("Select Visualization")
plot_type = st.sidebar.radio("Choose a plot type", ["Bar Chart", "Scatter Plot", "Box Plot", "Heatmap"])
#Bar Plot
if plot_type == "Bar Chart":
    st.write("### Bar Chart")
    bar_columns = filtered_data.select_dtypes(include=["object", "category"]).columns.tolist()
    selected_bar_column = st.sidebar.selectbox("Select column for bar chart", bar_columns)
    max_categories = 10
    bar_data = filtered_data[selected_bar_column].value_counts().head(max_categories)
    fig, ax = plt.subplots()
    bar_data.plot(kind="bar", ax=ax)
    ax.set_title(f"Bar Chart for {selected_bar_column}")
    st.pyplot(fig)
    st.write(f"""
    **Insight:** The bar chart shows the distribution of values in the selected column: **{selected_bar_column}**.
    Use this plot to:
    - Identify the most common categories or values.
    - Understand trends, such as popular product categories or customer preferences.
    - Make data-driven decisions, e.g., focusing on high-performing product categories.
    """)

# Scatter plot 
elif plot_type == "Scatter Plot":
    st.write("### Scatter Plot")
    numeric_columns = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if len(numeric_columns) >= 2:
        x_axis = st.sidebar.selectbox("Select X-axis", numeric_columns, key="scatter_x")
        y_axis = st.sidebar.selectbox("Select Y-axis", numeric_columns, key="scatter_y")
        categorical_columns = filtered_data.select_dtypes(include=['object', 'category']).columns.tolist()
        color_column = st.sidebar.selectbox("Select column for color (Grouping)", [None] + categorical_columns)
        if color_column: 
            fig = px.scatter(filtered_data, x=x_axis, y=y_axis, color=color_column, title=f"Scatter Plot of {x_axis} vs {y_axis}")
        else: 
            fig = px.scatter(filtered_data, x=x_axis, y=y_axis, title=f"Scatter Plot of {x_axis} vs {y_axis}")
        st.plotly_chart(fig)
        st.write(f"""
        **Insight:** This scatter plot visualizes the relationship between two numerical variables: **{x_axis}** and **{y_axis}**.
        Use this plot to:
        - Detect patterns or correlations between variables (e.g., higher discounts vs. net amount).
        - Compare customer spending habits based on {f'**{color_column}**' if color_column else '**no grouping**'}.
        - Explore outliers or clusters in the data.
        """)
    else:
        st.write("No data available for creating a scatter plot.")

# Box plot 
elif plot_type == "Box Plot":
    st.write("### Box Plot")
    numeric_columns = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_columns:
        box_column = st.sidebar.selectbox("Select column for box plot (Y-axis)", numeric_columns)
        categorical_columns = filtered_data.select_dtypes(include=['object', 'category']).columns.tolist()
        color_column = st.sidebar.selectbox("Select column for grouping (Color)", [None] + categorical_columns)
        fig = px.box(filtered_data, y=box_column, color=color_column, title=f"Box Plot of {box_column}")
        st.plotly_chart(fig)
        st.write(f"""
        **Insight:** The box plot visualizes the distribution of the selected numerical variable: **{box_column}**.
        - The selected color column **{color_column if color_column else 'None'}** allows grouping by categorical values.
        - Use this plot to detect medians, interquartile ranges, and potential outliers.
        - Compare distributions across categories or observe overall spread for decision-making.
        """)
    else:
        st.write("No data available for creating a box plot.")

# Heatmap 
elif plot_type == "Heatmap":
    st.write("### Heatmap of Correlations")
    numeric_columns = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if numeric_columns:
        selected_numeric_columns = st.sidebar.multiselect(
            "Select numerical columns for correlation heatmap", 
            numeric_columns, 
            default=numeric_columns[:5] 
        )
        
        if selected_numeric_columns:
            num_rows = len(filtered_data)
            sample_size = min(num_rows, 1000)
            sample_data = filtered_data[selected_numeric_columns].sample(n=sample_size, random_state=42)
            correlation_matrix = sample_data.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
            st.write(f"""
            **Insight:** This heatmap shows the pairwise correlation coefficients for the selected variables:
            **{', '.join(selected_numeric_columns)}**.
            - Correlation coefficients range from **-1** (strong negative correlation) to **1** (strong positive correlation).
            - Values close to **0** indicate weak or no linear relationship.
            - Use this heatmap to:
              - Identify variables that move together (positive correlation) or in opposite directions (negative correlation).
              - Pinpoint independent variables with low correlation for predictive modeling.
              - Highlight key relationships that could inform data strategies.
            """)
        else:
            st.write("Please select at least one numerical column to generate a heatmap.")
    else:
        st.write("No data available for correlation heatmap.")


st.sidebar.header("Advanced Data Filtering")
filter_column = st.sidebar.selectbox("Select Column to Filter", columns)
filter_value = st.sidebar.text_input("Enter Value to Filter")
if filter_value:
    filtered_data = filtered_data[filtered_data[filter_column].astype(str).str.contains(filter_value, case=False)]
    st.write(f"### Filtered Data by {filter_column} containing {filter_value}")
    st.dataframe(filtered_data)
