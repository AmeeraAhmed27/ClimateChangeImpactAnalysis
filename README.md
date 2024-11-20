<div>
  <img src="/Images/cover.jpg" ></img>
</div>

# Welcome to Climate Change Impact Analysis Project
<div>
<img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="github"/>
<img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter" alt="jupyter" />

</div>
<em><p>Climate change is the long-term alteration of temperature and typical weather patterns in a place. Climate change could refer to a particular location or the planet as a whole. Climate change may cause weather patterns to be less predictable.</p></em>

## Project Info
<p>This project aims to analyze and visualize the impact of climate change using publicly available datasets. The interactive dashboard provides insights into temperature changes, CO2 emissions, sea level rise, and other climate-related variables over time.
</p>

## Features

- **Temperature Line Graph**: Visualizes temperature changes over time.
- **Precipitation Heat Map**: Illustrates regions affected by Precipitation.
- **CO2 Emissions Bar Chart**: Compares CO2 emissions across different countries.
- **Scatter Plot**: Displays the relationship between CO2 emissions and temperature.

## Tools Used

<div>
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" /> 
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" /> 
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python"/>
  <img src="https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter" alt="jupyter" />
  <img src="https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</div>

- **Python**: Programming language for data manipulation and analysis.
- **Pandas**: Library for data handling and analysis.
- **Plotly**: Library for creating interactive visualizations.
- **Dash**: Framework for building web applications with Python.
- **Jupyter Notebook**: For exploratory data analysis (EDA).
- **Streamlit**: For budling interactive user interface .

## Dataset Information
<p> The Dataset Chosen for this Project is from <a href="https://www.kaggle.com/datasets/goyaladi/climate-insights-dataset?resource=download"><b>kaggle</b></a>.

<br>This dataset provides valuable insights into the ongoing changes in our climate. It encompasses a comprehensive collection of temperature records, CO2 emissions data, and sea level rise measurements. With a focus on global trends, it enables researchers, scientists, and climate enthusiasts to analyze the impact of climate change on our planet.</p>
<p>The Climate Insights Dataset offers a wealth of historical climate data, enabling researchers, scientists, and climate enthusiasts to analyze and understand global climate trends. With a vast collection of records spanning multiple years, this dataset serves as a valuable resource for in-depth exploration and analysis.
  
The project uses a climate dataset with the following columns: 🌡️📈🌊

Temperature: Average temperature measurements in Celsius.

CO2 Emissions: Levels of carbon dioxide emissions in parts per million (ppm).

Sea Level Rise: Measured sea level rise in millimeters.

Precipitation: Rainfall amounts in millimeters.

Humidity: Relative humidity in percentage.

Wind Speed: Wind speed in kilometers per hour.
  
  
The raw data comes from the Kaggle.</p>
<p> Dataset Link: <a href="https://www.kaggle.com/datasets/goyaladi/climate-insights-dataset?resource=download">Climate Insights Dataset</a></p> 

  
## Installation of Tools and Libraries 

### Using PIP 

You would be needing multiple libraries such as Math, Numpy, Pandas and Seaborn for basic operations and many pre-processing and ML Model libraries.

```sh
pip install [Library_Name]
```
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/AmeeraAhmed27/ClimateChangeImpactAnalysis.git   

2. **Install the required packages**:
   ```sh
   pip install dash plotly pandas
   pip install streamlit


3. **Run the visualization**:
   ```sh
   python Visulization.py
	```

Open your web browser and navigate to `http://127.0.0.1:8050/` to view the visualization.


4. **Run the dashboard**:
   ```sh
   streamlit run Dashboard.py


Open your web browser and navigate to `http://192.168.8.12:8501/` to view the dashboard.


## Usage

After running the application, you can interact with the dashboard to explore:

- **Temperature trends** over different decades.
- **Heat maps** indicating areas most affected by climate-related events.
- **Comparative bar charts** of CO2 emissions from various countries.
- **Scatter plots** analyzing the correlation between CO2 emissions and temperature and any another features.



# Getting Started with the Project
## Analysis Project

### Importing the required Libraries 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Reading the CSV File from the local machine


```python
data=pd.read_csv('climate_change_data.csv')
```

### Basic Exploration on the Dataset 

This section of code helps us to comprehend the basic understanding of the dataset. We retrieved the dataset in a form of Pandas Data Frame.

```python
# Gives the Basic Structure of the Data Frame such as Data type and Col names.
data.info()

# Show the basic Statistical Values of the data such as Mean, Count, StD, Min and Quartiles
data.describe()


# Names of the Column 
print(data.columns.values)

# First 5 Records in the dataset
data.head()

# Shows the number of Null values in each column 
print(data.isna().sum())
```

### Manipulation in the Data Frame

Since, we are going to perform Time Series Analysis on the Dataset. So, now we will be changing the structure according to our ease and analysis techniques.

Transforming the Dt column to Date column ( Object datatype to Date Type)

```python
# Converting Dt into Date
data['Date'] = pd.to_datetime(data['Date'])
```

showing Temperature Variations Over Time


```python
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Date', y='Temperature')
plt.title('Temperature Variations Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

<div>
  <img src="/Images/TemperatureVariationsOverTimePlot.png" ></img>
</div>

Time Series Plot of Precipitation

```python
plt.figure(figsize=(14, 7))
sns.lineplot(data=data, x='Date', y='Precipitation')
plt.title('Precipitation Over Time')
plt.xlabel('Date')
plt.ylabel('Precipitation (mm)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

<div>
  <img src="/Images/PrecipitationOverTime.png" ></img>
</div>

```python
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

monthly_avg = data.groupby(['Year', 'Month'])['Precipitation'].mean().reset_index()

print(monthly_avg)

monthly_avg['Month-Year'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(DAY=1))

plt.figure(figsize=(14, 7))
sns.barplot(data=monthly_avg, x='Month-Year', y='Precipitation', color='lightblue')
plt.title('Average Monthly Precipitation')
plt.xlabel('Month-Year')
plt.ylabel('Average Precipitation (mm)')
plt.tight_layout()
plt.show()
```

<div>
  <img src="/Images/AverageMonthlyPrecipitation.png" ></img>
</div>

Box Plot of Precipitation by Month
```python
plt.figure(figsize=(14, 7))
sns.boxplot(data=data, x='Month', y='Precipitation', palette='Set3')
plt.title('Box Plot of Precipitation by Month')
plt.xlabel('Month')
plt.ylabel('Precipitation (mm)')
plt.xticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.show()
```

<div>
  <img src="/Images/BoxPlotofPrecipitationbyMonth.png" ></img>
</div>

Correlation Analysis

```python
correlation_matrix = data[['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Precipitation', 'Humidity', 'Wind Speed']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

<div>
  <img src="/Images/CorrelationMatrix.png" ></img>
</div>


Frequency of extreme weather events
```python
# Thresholds for extreme events
extreme_heat_threshold = 35
extreme_precipitation_threshold = 100

data['Extreme Heat'] = data['Temperature'] > extreme_heat_threshold
data['Extreme Precipitation'] = data['Precipitation'] > extreme_precipitation_threshold

extreme_heat_counts = data.groupby(data['Date'].dt.year)['Extreme Heat'].sum().reset_index()
print(extreme_heat_counts)

extreme_precipitation_counts = data.groupby(data['Date'].dt.year)['Extreme Precipitation'].sum().reset_index()

plt.figure(figsize=(14, 7))

# Plot extreme heat events
sns.lineplot(data=extreme_heat_counts, x='Date', y='Extreme Heat', label='Extreme Heat Events', color='red')

# Plot extreme precipitation events
sns.lineplot(data=extreme_precipitation_counts, x='Date', y='Extreme Precipitation', label='Extreme Precipitation Events', color='blue')

plt.title('Frequency of Extreme Weather Events Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Events')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

<div>
  <img src="/Images/WeatherEventsOverTime.png" ></img>
</div>

Analyze the relationships between CO2 levels and temperature changes over time
```python
correlation_matrix = data[['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Precipitation', 'Humidity', 'Wind Speed']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

<div>
  <img src="/Images/relationshipsbetweenCO2levelsandtemperature.png" ></img>
</div>

 Regression Analysis
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Select features and target variable
X = data[['CO2 Emissions']]
y = data['Temperature']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

r2 = r2_score(y_test, y_pred)
print(f'R2 Score: {r2:.2f}')

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['CO2 Emissions'], y=y_test, label='Actual', color='blue', alpha=0.6)
sns.scatterplot(x=X_test['CO2 Emissions'], y=y_pred, label='Predicted', color='red', alpha=0.6)
plt.title('Linear Regression: Actual vs Predicted Temperature')
plt.xlabel('CO2 Emissions (ppm)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()
```
<div>
  <img src="/Images/LinearRegression.png" ></img>
</div>

## Interactive Streamlit Dashboard 

The Climate Change Dashboard is an interactive web application designed to visualize and analyze climate change data. Built using Streamlit and Plotly, this dashboard allows users to explore various climate metrics and trends across different countries and time periods. Below are the key features and functionalities of the dashboard:

- **Interactive Filters**: Users can filter data by country and date range to focus on specific regions and time periods.
- **Visualizations**:
  - **Temperature Changes**: A line graph displays temperature trends over time for selected countries.
  - **CO2 Emissions**: A bar chart compares CO2 emissions by country.
  - **Precipitation Heat Map**: A density heat map visualizes precipitation levels across different locations and dates.
  - **Scatter Plots**: Users can create scatter plots to analyze relationships between different climate indicators (e.g., Temperature vs. CO2 Emissions).
  - **Time Series Graphs**: Interactive line graphs show trends for selected indicators over time.
- **Data Download**: Users can download the filtered dataset as an Excel file for offline analysis.
- **Download Plots**: Users can download the filtered dataset plot as png image.
## Dashboard Overview
<div>
  <img src="/Images/dashboard1.png" ></img>
</div>

The Climate Change Dashboard includes two main sections for data analysis and visualization:

### 1. Data Filtering and Visualization

In this section, users can filter data for the following visualizations:
- **Temperature Line Graph**
- **Heat Map for Precipitation**
- **CO2 Emissions Bar Chart**

Users can filter the data based on:
- **Country**
- **Date Range**

<div>
  <img src="/Images/dashboard2.png" ></img>
</div>
<div>
  <img src="/Images/dashboard4.png" ></img>
</div>
<div>
  <img src="/Images/dashboard5.png" ></img>
</div>
<div>
  <img src="/Images/dashboard6.png" ></img>
</div>
<div>
  <img src="/Images/dashboard7.png" ></img>
</div>
### 2. Scatter Plot Analysis

This section allows users to create scatter plots to analyze relationships between different climate indicators(e.g., Temperature vs. CO2 Emissions),country and date range

<div>
  <img src="/Images/dashboard3.png" ></img>
</div>
<div>
  <img src="/Images/dashboard8.png" ></img>
</div>
<div>
  <img src="/Images/dashboard9.png" ></img>
</div>
<div>
  <img src="/Images/dashboard10.png" ></img>
</div>
<div>
  <img src="/Images/dashboard11.png" ></img>
</div>
