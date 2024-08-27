import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
from google.colab import drive
import streamlit as st


st.set_page_config(page_title="Flood Prediction Analysis", layout="wide")
st.markdown('<p class="big-font">Flood Prediction Analysis</p>', unsafe_allow_html=True)
# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

#file = st.file_uploader(" Upload your file (csv/txt/xlsx/xls)", type=["csv", "txt", "xlsx", "xls"])

default_file_path = r'/content/drive/MyDrive/Flood Prediction/flood_updated_1.csv'
file = st.file_uploader("📂 Upload your file (csv/txt/xlsx/xls)", type=["csv", "txt", "xlsx", "xls"])
# Function to load data from the uploaded file or default path
@st.cache_data
def load_data(file=None, default_path=None):
    try:
        if file is not None:
            return pd.read_csv(file, encoding="ISO-8859-1")
        else:
            st.error("No file provided and no default file available.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


if file is not None:
    df = load_data(file=file)
else:
    st.info(f"No file uploaded")
    df = load_data(default_path=default_file_path)



# Error handling if dataframe is empty
if df is None:
    st.stop()

#df = pd.read_csv(r'/content/drive/MyDrive/Flood Prediction/flood_updated_1.csv')

# Group the data by MonsoonIntensity and calculate the average FloodProbability
flood_prob_by_intensity = df.groupby('MonsoonIntensity')['FloodProbability'].mean().reset_index()

# Visualization using Plotly
fig = px.bar(flood_prob_by_intensity, x='MonsoonIntensity', y='FloodProbability', 
             title='Average Flood Probability by Monsoon Intensity')
st.plotly_chart(fig)

highest_prob_intensity = flood_prob_by_intensity[flood_prob_by_intensity['FloodProbability'] == flood_prob_by_intensity['FloodProbability'].max()]['MonsoonIntensity'].values[0]
highest_prob = flood_prob_by_intensity['FloodProbability'].max()

st.write("**Insights:**")
st.write(f"- Monsoon with intensity '{highest_prob_intensity}' has the highest average flood probability of {highest_prob:.2f}.")

from sklearn.preprocessing import LabelEncoder

# Select columns to encode (excluding 'FloodProbability')
columns_to_encode = df.columns[df.columns != 'FloodProbability']

# Apply label encoding to each selected column
label_encoders = {}
for column in columns_to_encode:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  label_encoders[column] = le  # Store the encoder
  # Correlation Matrix
st.subheader("Correlation Matrix")
corr_matrix = df.corr()
fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                 x=corr_matrix.columns,
                                 y=corr_matrix.columns,
                                 colorscale='Viridis'))
st.plotly_chart(fig)

# Decode the columns
for column, le in label_encoders.items():
  df[column] = le.inverse_transform(df[column])



# Apply label encoding to each selected column
label_encoders = {}
for column in columns_to_encode:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  label_encoders[column] = le  # Store the encoder

# Assuming 'Flood Probability' is the target variable and the rest are independent variables
X = df.drop('FloodProbability', axis=1)  
y = df['FloodProbability']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features (optional but often recommended for regression)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Flood Probability")
plt.ylabel("Predicted Flood Probability")
plt.title("Actual vs. Predicted Flood Probability")
plt.show()

# Decode the columns
for column, le in label_encoders.items():
  df[column] = le.inverse_transform(df[column])

# Dropdown for Political Factor (with unique key)
selected_factor1 = st.selectbox('Select Political Factor', df['PoliticalFactors'].unique(), key='political_factor1')

# Filter data and calculate average flood probability
filtered_df1 = df[df['PoliticalFactors'] == selected_factor1]
average_probability1 = filtered_df1['FloodProbability'].mean()

st.write(f"Average Flood Probability for {selected_factor1}: {average_probability1:.2f}")

# Pie chart for selected political factor (with unique key)
labels = ['Flood Probability', 'No Flood Probability']
values = [average_probability1, 1 - average_probability1]
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_layout(title=f'Flood Probability Distribution for {selected_factor1}')
st.plotly_chart(fig, key='pie_chart1')


le_urbanization = LabelEncoder()
df['Urbanization_en'] = le_urbanization.fit_transform(df['Urbanization'])

# Create mapping of encoded values to original values
mapping = dict(zip(le_urbanization.classes_, le_urbanization.transform(le_urbanization.classes_)))

# Display the mapping in a dropdown
selected_encoding = st.selectbox('Select Urbanization Encoding:', options=list(mapping.keys()))
st.write(f"Encoded Value: {mapping[selected_encoding]}")

# Group the data by 'Urbanization' and calculate the average flood probability for each group
grouped_data = df.groupby('Urbanization_en')['FloodProbability'].mean().reset_index()

# Create a scatter plot to visualize the relationship
fig = px.scatter(grouped_data, x='Urbanization_en', y='FloodProbability', 
                 title='Average Flood Probability vs. Urbanization',
                 trendline="ols")  # Add a trendline
st.plotly_chart(fig)

ols_results = px.get_trendline_results(fig)
ols_params = ols_results.px_fit_results.iloc[0].params
slope = ols_params[1]
intercept = ols_params[0]
r_squared = ols_results.px_fit_results.iloc[0].rsquared

# Display the scatter plot
#st.plotly_chart(fig)

# Insights
st.subheader("Insights from the Scatter Plot:")

# Analyze the trendline
if slope > 0:
    st.write("- There seems to be a positive correlation between urbanization and flood probability.")
elif slope < 0:
    st.write("- There seems to be a negative correlation between urbanization and flood probability.")
else:
    st.write("- There seems to be little or no correlation between urbanization and flood probability.")

st.write(f"- The linear trendline suggests a relationship of y = {slope:.2f}x + {intercept:.2f}.")
st.write(f"- The R-squared value of {r_squared:.2f} indicates the strength of the linear fit.")


label_encoders = {}
for column in columns_to_encode:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  label_encoders[column] = le 

# Value Prediction Model
st.header("Flood Prediction Model")

features = ["MonsoonIntensity", "TopographyDrainage", "RiverManagement", "DrainageSystems", "Deforestation", "Urbanization", "InadequatePlanning", "ClimateChange", "CoastalVulnerability", "Landslides"]
X = df[features]
y = df["FloodProbability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write(f"Model R-squared: {r2_score(y_test, y_pred):.4f}")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.coef_
})
feature_importance = feature_importance.sort_values("Importance", ascending=False)

fig_importance = px.bar(feature_importance, x="Importance", y="Feature", orientation="h")
fig_importance.update_layout(title="Feature Importance for Flood Prediction")
st.plotly_chart(fig_importance)# Value Prediction Model
#st.header("Flood Prediction Model")

features = ["MonsoonIntensity", "TopographyDrainage", "RiverManagement", "DrainageSystems", "Deforestation", "Urbanization", "InadequatePlanning", "ClimateChange", "CoastalVulnerability", "Landslides"]
X = df[features]
y = df["FloodProbability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# Insights
st.subheader("Insights from the Model:")

# R-squared interpretation
r2 = r2_score(y_test, y_pred)
if r2 >= 0.8:
    st.write(f"- The model explains {r2:.2%} of the variance in flood probability, indicating a strong fit.")
elif r2 >= 0.5:
    st.write(f"- The model explains {r2:.2%} of the variance in flood probability, indicating a moderate fit.")
else:
    st.write(f"- The model explains {r2:.2%} of the variance in flood probability, indicating a weak fit. Further model exploration or feature selection might be necessary.")

# Most important features
top_features = feature_importance.head(3)["Feature"].tolist()
st.write(f"- The features with the highest impact on flood probability are: {', '.join(top_features)}.")


columns = ['Urbanization', 'InadequatePlanning', 'MonsoonIntensity',
           'TopographyDrainage', 'RiverManagement', 'DrainageSystems',
           'Deforestation', 'ClimateChange', 'CoastalVulnerability', 'Landslides']


# Scatter plots for Flood Probability vs. Contributing Factors
st.title("Relationship Between Flood Probability and Contributing Factors")
for col in columns[1:]:
    fig, ax = plt.subplots()
    sns.scatterplot(x='FloodProbability', y=col, data=df, ax=ax)
    plt.savefig(f"flood_probability_vs_{col}.png")
    st.pyplot(fig)


    # Insights based on the scatter plot
    if (2 > 1):
        correlation_coef = df['FloodProbability'].corr(df[col])
        if correlation_coef > 0.05:
            st.write(f"- **{col}:** There seems to be a strong positive correlation with flood probability.")
        elif correlation_coef < -0.05:
            st.write(f"- **{col}:** There seems to be a strong negative correlation with flood probability.")
        else:
            st.write(f"- **{col}:** The relationship with flood probability is weak or non-existent.")

        # You can add more specific insights based on your observations
        # of the scatter plot, such as identifying outliers or clusters.
    else:
        st.warning(f"Insufficient data points for {col}.")



# Store label encoders for each column
label_encoders = {}

# Select columns to encode (excluding 'FloodProbability')
columns_to_encode = df.columns[df.columns != 'FloodProbability']

# Apply label encoding to each selected column
for column in columns_to_encode:
  le = LabelEncoder()
  df[column] = le.fit_transform(df[column])
  label_encoders[column] = le  # Store the encoder


