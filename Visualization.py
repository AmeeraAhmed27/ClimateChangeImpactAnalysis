import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Load the data from CSV
df = pd.read_csv('climate_change_data.csv')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div(style={'padding': '20px'}, children=[
    html.H1(children='Climate Change Dashboard', style={'textAlign': 'center', 'color': '#333'}),
    
    # Line Graph for Temperature Changes
    dcc.Graph(
        id='temperature-line-graph',
        figure=px.line(df, x='Date', y='Temperature', title='Temperature Changes Over Time', 
                       template='plotly_dark', color='Country', markers=True)
    ),
    
    # Bar Chart for CO2 Emissions
    dcc.Graph(
        id='co2-bar-chart',
        figure=px.bar(df, x='Country', y='CO2 Emissions', title='CO2 Emissions by Country', 
                      template='plotly_dark', color='CO2 Emissions', text='CO2 Emissions')
    ),
    
    # Heat Map for CO2 Emissions
    dcc.Graph(
        id='heatmap',
        figure=px.density_heatmap(df, x='Location', y='Date', z='CO2 Emissions',
                                   title='Heat Map of CO2 Emissions', 
                                   template='plotly_dark', color_continuous_scale='Viridis')
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
