# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:26:15 2022

@author: samuel bazaz
"""

# !!!!!!!!!!!!!!!!!!!!!!!!!A MODIFIER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mainpath = "../"
path_fig = "C:/Users/samud/Bureau/Python code/MC fig2/"
# !!!!!!!!!!!!!!!!!!!!!!!!!A MODIFIER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# _____________________________packages______________________________________________

import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.io as pio
from dash import Dash, dcc, html, Input, Output

pio.renderers.default = "browser"

#_______________________importation et preprocessing________________________________________

df3 = pd.read_csv( mainpath +"MC_data3.csv" )
unwanted = df3.columns[df3.columns.str.startswith('Unnamed')]
df3.drop(unwanted, axis=1, inplace=True)
df3 = df3.dropna()
df3["type"] = df3["type_estim"]+" "+df3["type_W"]

#________________calcul fig perf pour alpha donné___________________________________________

alph = 4

daux = pd.DataFrame()
daux["perf"] = -alph*np.log(df3.groupby(["type", "n", "N"]).var()["res"])  -np.log(df3.groupby(["type", "n", "N"]).mean()["tps"])
df3 = pd.merge(df3, daux,how='left', on=["type", "n", "N"])

# translation et normalisation
df3["perf"] -= min( min(df3["perf"]), 0 )
df3["perf"] /= max(df3["perf"])

# création et sauvegarde de la figure
fig11 = px.scatter_3d(df3, x='n', y='N', z='perf',
              color='type', opacity=0.5)

fig11.write_html( path_fig +"fig11.html")

#______________________________creation du server local_____________________________________

app = Dash(__name__)

app.layout = html.Div([
    html.Div(children = [
    dcc.Graph(id="graph", figure={
            'layout': {'height': 800,'width':1300 }}),
    ],style={'width': '95%' , 'flex': 1}),
    html.Div(children = [
    html.Br(),
    html.Label('Alpha'),
    dcc.Slider(
        id='alpha',
        min=0, max=10, step=1,
        value= 4,
        vertical=True,
    ),
    ], style={'width': '5%', 'flex': 1})
], style={'display': 'flex', 'flex-direction': 'row'})


@app.callback(
    Output("graph", "figure"), 
    Input("alpha", "value"))
def update_bar_chart(alpha):
    
    daux = pd.DataFrame()
    df3 = pd.read_csv('../MC_data3.csv')
    unwanted = df3.columns[df3.columns.str.startswith('Unnamed')]
    df3.drop(unwanted, axis=1, inplace=True)
    df3 = df3.dropna()
    df3["type"] = df3["type_estim"]+" "+df3["type_W"]
    daux["perf"] = -alpha*np.log(df3.groupby(["type", "n", "N"]).var()["res"])  -np.log(df3.groupby(["type", "n", "N"]).mean()["tps"])
    df3 = pd.merge(df3, daux,how='left', on=["type", "n", "N"])
    df3["perf"] -= min( min(df3["perf"]), 0 )
    df3["perf"] /= max(df3["perf"])
    fig = px.scatter_3d(df3, x='n', y='N', z='perf',
              color='type', opacity=0.5)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    
#http://127.0.0.1:8050/