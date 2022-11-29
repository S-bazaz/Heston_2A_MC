# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:27:32 2022

@author: samuel bazaz
"""

# !!!!!!!!!!!!!!!!!!!!!!!!!A MODIFIER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
mainpath = "../"
path_fig = "C:/Users/samud/Bureau/Python code/MC fig2/"
# !!!!!!!!!!!!!!!!!!!!!!!!!A MODIFIER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# _____________________________packages______________________________________________

import pandas as pd 
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"  # show dans un navigateur

#_____________________importation et preprocessing__________________________________

df3 =  pd.read_csv( mainpath +"MC_data3.csv" )
unwanted = df3.columns[df3.columns.str.startswith('Unnamed')]
df3.drop(unwanted, axis=1, inplace=True)
df3 = df3.dropna()
df3["type"] = df3["type_estim"]+" "+df3["type_W"]

df3_simple = df3[df3.type_W == "W_simple"]
df3_bridge = df3[df3.type_W == "W_bridge_dya"]
df3_QMC = df3[df3.type_W == "W_simple_QMC"]
df3_QMCHM = df3[df3.type_W == "W_simple_QMCHM"]

# ________________________figures_________________________________________________________

## 3D raw data
fig1 = px.scatter_3d(df3, x='n', y='N', z='tps',
              color= 'type', opacity=0.4)

fig2 = px.scatter_3d(df3, x='n', y='N', z='res',
              color='type', opacity=0.4)

## hist globales

fig3 = px.histogram( df3, x="res",  color="type",  opacity = 0.5, hover_data=df3.columns, marginal="box")
fig4 = px.histogram( df3, x="tps",  color="type",  opacity = 0.5, hover_data=df3.columns, marginal="violin")

## hist comparaison par types
fig5 = px.histogram( df3, x="res",  color="type_estim", opacity = 0.5, facet_col="type_W",  marginal="box", hover_data=df3.columns)
fig5.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig6 = px.histogram( df3, x="tps",  color="type_estim", opacity = 0.5, facet_col="type_W",  marginal="violin", hover_data=df3.columns)
fig6.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

## dynamique fine et influence n
fig7a = px.histogram( df3_simple, x="res", title = "W_simple" , color="n", opacity = 0.7, facet_col= "type_estim" , marginal = "box", hover_data=df3.columns,facet_col_spacing=0.01)
fig7a.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig7b = px.histogram( df3_bridge, x="res", title = "W_bridge_dya" , color="n", opacity = 0.7, facet_col= "type_estim" , marginal = "box", hover_data=df3.columns,facet_col_spacing=0.01)
fig7b.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig7c = px.histogram( df3_QMC, x="res", title = "W_simple_QMC" , color="n", opacity = 0.7, facet_col= "type_estim" , marginal = "box", hover_data=df3.columns,facet_col_spacing=0.01)
fig7c.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig7d = px.histogram( df3_QMCHM, x="res", title = "W_simple_QMCHM" , color="n", opacity = 0.7, facet_col= "type_estim" , marginal = "box", hover_data=df3.columns,facet_col_spacing=0.01)
fig7d.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig8a = px.histogram( df3_simple, x="tps", title = "W_simple" , color="n", opacity = 0.7, facet_col= "type_estim" , marginal = "box",hover_data=df3.columns, facet_col_spacing=0.01)
fig8a.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig8b = px.histogram( df3_bridge, x="tps", title = "W_bridge_dya" , color="n", opacity = 0.7, facet_col= "type_estim" , marginal = "box",hover_data=df3.columns, facet_col_spacing=0.01)
fig8b.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig8c = px.histogram( df3_QMC, x="tps", title = "W_simple_QMC" , color="n", opacity = 0.7, facet_col= "type_estim" , marginal = "box",hover_data=df3.columns, facet_col_spacing=0.01)
fig8c.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig8d = px.histogram( df3_QMCHM, x="tps", title = "W_simple_QMCHM" , color="n", opacity = 0.7, facet_col= "type_estim" , marginal = "box",hover_data=df3.columns, facet_col_spacing=0.01)
fig8d.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

## dynamique fine et influence N
fig9a = px.histogram( df3_simple, x="res", title = "W_simple" , color="N", opacity = 0.7, facet_col= "type_estim" , marginal = "box", hover_data=df3.columns,facet_col_spacing=0.01)
fig9a.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig9b = px.histogram( df3_bridge, x="res", title = "W_bridge_dya" , color="N", opacity = 0.7, facet_col= "type_estim" , marginal = "box", hover_data=df3.columns,facet_col_spacing=0.01)
fig9b.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig9c = px.histogram( df3_QMC, x="res", title = "W_simple_QMC" , color="N", opacity = 0.7, facet_col= "type_estim" , marginal = "box", hover_data=df3.columns,facet_col_spacing=0.01)
fig9c.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig9d = px.histogram( df3_QMCHM, x="res", title = "W_simple_QMCHM" , color="N", opacity = 0.7, facet_col= "type_estim" , marginal = "box", hover_data=df3.columns,facet_col_spacing=0.01)
fig9d.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))


fig10a = px.histogram( df3_simple, x="tps", title = "W_simple" , color="N", opacity = 0.7, facet_col= "type_estim" , marginal = "box",hover_data=df3.columns, facet_col_spacing=0.01)
fig10a.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig10b = px.histogram( df3_bridge, x="tps", title = "W_bridge_dya" , color="N", opacity = 0.7, facet_col= "type_estim" , marginal = "box",hover_data=df3.columns, facet_col_spacing=0.01)
fig10b.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig10c = px.histogram( df3_QMC, x="tps", title = "W_simple_QMC" , color="N", opacity = 0.7, facet_col= "type_estim" , marginal = "box",hover_data=df3.columns, facet_col_spacing=0.01)
fig10c.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

fig10d = px.histogram( df3_QMCHM, x="tps", title = "W_simple_QMCHM" , color="N", opacity = 0.7, facet_col= "type_estim" , marginal = "box",hover_data=df3.columns, facet_col_spacing=0.01)
fig10d.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))


#_____________________________________sauvegardes___________________________________________

fig1.write_html( path_fig +"fig1.html" )
fig2.write_html( path_fig +"fig2.html" )
fig3.write_html( path_fig +"fig3.html" )
fig4.write_html( path_fig +"fig4.html" )
fig5.write_html( path_fig +"fig5.html" )
fig6.write_html( path_fig +"fig6.html" )

fig7a.write_html( path_fig +"fig7a.html" )
fig7b.write_html( path_fig +"fig7b.html" )
fig7c.write_html( path_fig +"fig7c.html" )
fig7d.write_html( path_fig +"fig7d.html" )

fig8a.write_html( path_fig +"fig8a.html" )
fig8b.write_html( path_fig +"fig8b.html" )
fig8c.write_html( path_fig +"fig8c.html" )
fig8d.write_html( path_fig +"fig8d.html" )

fig9a.write_html( path_fig +"fig9a.html" )
fig9b.write_html( path_fig +"fig9b.html" )
fig9c.write_html( path_fig +"fig9c.html" )
fig9d.write_html( path_fig +"fig9d.html" )

fig10a.write_html( path_fig +"fig10a.html" )
fig10b.write_html( path_fig +"fig10b.html" )
fig10c.write_html( path_fig +"fig10c.html" )
fig10d.write_html( path_fig +"fig10d.html" )

#__________________________show_______________________________________________

#fig1.show()
#fig2.show()
#fig3.show()
#fig4.show()
#fig5.show()
#fig6.show()
