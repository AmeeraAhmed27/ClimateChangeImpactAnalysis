import pandas as pd
import streamlit as st
import plotly.express as px
from io import BytesIO

# Load the dataset
df = pd.read_csv('climate_change_data.csv')
df['Date'] = pd.to_datetime(df['Date']) 

# Streamlit app title
st.title('Climate Change Dashboard')

# Sidebar for filters
st.sidebar.header('Filters')

# Location filter
locations = df['Country'].unique()
selected_locations = st.sidebar.multiselect('Select Country:', locations, default=[locations[0]])

# Date range filter
start_date = st.sidebar.date_input('Start Date', df['Date'].min().date())
end_date = st.sidebar.date_input('End Date', df['Date'].max().date())

# Filter the DataFrame based on user input
filtered_df = df[(df['Country'].isin(selected_locations)) & 
                 (df['Date'] >= pd.to_datetime(start_date)) & 
                 (df['Date'] <= pd.to_datetime(end_date))]

# Temperature Line Graph
temp_fig = px.line(filtered_df, x='Date', y='Temperature', 
                    title='Temperature Changes Over Time', 
                    color='Country', markers=True)
st.plotly_chart(temp_fig)


# CO2 Emissions Bar Chart
co2_fig = px.bar(filtered_df, x='Country', y='CO2 Emissions', 
                  title='CO2 Emissions by Country', 
                  color='CO2 Emissions', text='CO2 Emissions')
st.plotly_chart(co2_fig)


# Heat Map for Precipitation
heatmap_fig = px.density_heatmap(filtered_df, x='Location', y='Date', z='Precipitation',
                                  title='Heat Map of Precipitation',
                                  color_continuous_scale='Viridis')
st.plotly_chart(heatmap_fig)

st.dataframe(filtered_df)

# Download Button for Data
def download_data(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Climate Change Data')
    output.seek(0)
    return output

if st.button('Download Report'):
    excel_file = download_data(filtered_df)
    st.download_button(label='Download Excel File', data=excel_file, file_name='climate_change_report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# Dropdown for indicators
st.sidebar.subheader('Scatter Plot Settings')
available_indicators = ['Temperature', 'CO2 Emissions', 'Sea Level Rise', 'Precipitation', 'Humidity', 'Wind Speed']

# Country selection for Scatter Plot
selected_country = st.sidebar.selectbox('Select Country for Scatter Plot', ['All'] + list(locations))

xaxis_column = st.sidebar.selectbox('Select X-axis Indicator', available_indicators, index=0)
xaxis_type = st.sidebar.radio('X-axis Type', ['Linear', 'Log'])

yaxis_column = st.sidebar.selectbox('Select Y-axis Indicator', available_indicators, index=1)
yaxis_type = st.sidebar.radio('Y-axis Type', ['Linear', 'Log'])

# Date range filter
start_date_scttor = st.sidebar.date_input('Start Date for Scatter Plot', df['Date'].min().date())
end_date_scttor = st.sidebar.date_input('End Date for Scatter Plot', df['Date'].max().date())

# Filter DataFrame for Scatter Plot based on selections
scatter_filtered_df = df[(df['Date'] >= pd.to_datetime(start_date_scttor)) & 
                 (df['Date'] <= pd.to_datetime(end_date_scttor))]

if selected_country != 'All':
    scatter_filtered_df = scatter_filtered_df[scatter_filtered_df['Country'] == selected_country]
st.title("Scatter Plot")
# Main graph: Scatter plot
scatter_fig = px.scatter(
    scatter_filtered_df,
    x=xaxis_column,
    y=yaxis_column,
    hover_name='Country',
)

scatter_fig.update_xaxes(title=xaxis_column, type='linear' if xaxis_type == 'Linear' else 'log')
scatter_fig.update_yaxes(title=yaxis_column, type='linear' if yaxis_type == 'Linear' else 'log')
st.plotly_chart(scatter_fig)

# Time series graphs
def create_time_series(dff, y_column, title):
    time_series_fig = px.line(dff, x='Date', y=y_column, title=title,color='Country', markers=True)
    time_series_fig.update_xaxes(title='Date', showgrid=False)
    time_series_fig.update_yaxes(type='linear' if yaxis_type == 'Linear' else 'log')
    return time_series_fig

# X-axis time series   
if selected_country != 'All':
    # country_df = df[df['Country'] == selected_country]
     country_df = df[(df['Country'] == selected_country) & 
                 (df['Date'] >= pd.to_datetime(start_date_scttor)) & 
                 (df['Date'] <= pd.to_datetime(end_date_scttor))]
    
else:

    country_df = df [(df['Date'] >= pd.to_datetime(start_date_scttor)) & 
                 (df['Date'] <= pd.to_datetime(end_date_scttor))]

x_time_series_fig = create_time_series(country_df, xaxis_column, f'Time Series for {xaxis_column} in {selected_country}')
st.plotly_chart(x_time_series_fig)

# Y-axis time series
y_time_series_fig = create_time_series(country_df, yaxis_column, f'Time Series for {yaxis_column} in {selected_country}')
st.plotly_chart(y_time_series_fig)
st.dataframe(country_df)
# Download button for data
if st.button('Download Data'):
    excel_file = download_data(country_df)
    st.download_button(label='Download Excel File', data=excel_file, file_name='climate_change_report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')