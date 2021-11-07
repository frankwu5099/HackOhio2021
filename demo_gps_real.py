import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import fetch_species_distributions
from sklearn.neighbors import KernelDensity
import geocoder
import googlemaps
import datetime as dt
import googlemaps
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
api_key = open('api').read().strip()
Longmin = -83.3
Longmax = -82.7
Latmin = 39.8
Latmax = 40.15
Longmid = (Longmin+Longmax)/2
Latmid = (Latmin+Latmax)/2
path_df = pd.read_csv("newPath.csv")

path_i = 0

def data_model(fn):
    root_df = pd.read_csv(fn)
    fil_columbus = (
        (Longmin<root_df.Longitude) &
        (root_df.Longitude <Longmax) &
        (Latmin<root_df.Latitude) &
        (root_df.Latitude<Latmax)
    )
    root_df= root_df[fil_columbus]
    T = pd.to_datetime(root_df.CrashReportedDateTime).dt
    root_df["normalized_time"] = ((T.hour)*60 + T.minute)/60*0.002
    root_df_expand = root_df.copy()
    root_df_expand["normalized_time"] =  root_df_expand["normalized_time"] +0.002*24
    root_df_expand = pd.concat((root_df,root_df_expand),ignore_index=True)
    kde_timespace = KernelDensity(
        bandwidth=0.002, metric="euclidean", kernel="gaussian", algorithm="ball_tree"
    )
    X = root_df_expand[['Longitude','Latitude',"normalized_time"]]
    kde_timespace.fit(X)
    return root_df, kde_timespace

def t_now():
    t_in_day = (dt.datetime.now().hour*60 + dt.datetime.now().minute)/60*0.002
    if t_in_day<24*0.002/2:
        return t_in_day + 24*0.002
    else:
        return t_in_day

def current_location_time():
    gmaps = googlemaps.Client(key=api_key)
    loc = gmaps.geolocate()
    dfgps = pd.DataFrame(columns = ["Long","Lat"])
    dfgps.Long = np.array([loc['location']['lng']])
    dfgps.Lat = np.array([loc['location']['lat']])
    dfgps["time"] = np.array([t_now()])
    return dfgps

root_df, kde_timespace = data_model("root_data.csv")
root_df["Scale"] = 1
root_df_sample = root_df.sample(5000)

dfgps = current_location_time()
dfgps['Scale'] = np.array([1])

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        #html.H4('Dangerous Place Detector'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-map'),
        dcc.Interval(
            id='interval-component',
            interval=800, # in milliseconds
            n_intervals=0
        )
    ])
)


@app.callback([Output('live-update-text', 'children'),Output('live-update-map', 'figure')],
              Input('interval-component', 'n_intervals'))
def update_risk(n):
    dfnow = current_location_time()
    dfnow['Scale'] = np.array([1])
    risk_score = kde_timespace.score_samples(dfnow[["Long","Lat","time"]])[0]
    print(dfnow[["Long","Lat","time"]])
    fig = px.scatter_mapbox(root_df_sample, lat='Latitude', lon='Longitude', size = 'Scale',opacity = 0.18, size_max = 25,
                            center=dict(lat=dfnow.Lat[0], lon=dfnow.Long[0]), zoom=14, color_discrete_sequence=['purple'],
                            mapbox_style="open-street-map",  width=1000, height=1200)
    color = "rgb(%d,%d,0)" %(int(255*(np.clip(risk_score,2.5,7)-2.5)/4.5),int(255*(7-np.clip(risk_score,2.5,7))/4.5))
    print(color)
    style = {'padding': '10px', 'fontSize': '36px', 'color':'#FFFFFF', 'background-color': color, 'text-align':'center', 'display':'block'}
    fig2 = px.scatter_mapbox(dfnow, lat="Lat", lon="Long",size_max = 18, size='Scale', color_discrete_sequence=[color], opacity=1)
    fig.add_trace(fig2.data[0])
    if risk_score<5.8:
        message = "Drive Safe!"
    else:
        message = "Be Careful!"

    return [
        html.Span('{:s}'.format(message), style=style),
    ],fig

if __name__ == '__main__':
    app.run_server(debug=True)
