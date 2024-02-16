#import libraries
import streamlit as st
#fetech data from online
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


#title
app_name = 'Stock Market Forcasting App'
#st.write(app_name)
#on streamli app click on always run rerun
st.title(app_name)
#subheader
st.subheader('This app is created to forcast the stock market price of the selected company')
#add an image from online source
st.image("https://www.shutterstock.com/shutterstock/photos/1669591111/display_1500/stock-vector-stock-market-economic-graph-with-diagrams-business-and-financial-concepts-and-reports-abstract-1669591111.jpg")
#st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTekIofrr_BQLlLRu1k3yGTUM4uyMvNDGrrZw&usqp=CAU")

#take input from the user of app about the start and end date

#sidebar
st.sidebar.header(" Select the Parameters from below")

start_date = st.sidebar.date_input('Start date', date(2023,1,1))
end_date=st.sidebar.date_input('End date', date(2023,12,31))

#add ticker symbol list
ticker_list= ['AAPL','MSFT','GOOG','GOOGL','META','TSLA','NVDA','ADBE','PYPL','INTC','CMCSA','NFLX','PEP']
#dropdown list
ticker = st.sidebar.selectbox('Select the company',ticker_list)

#fetch data from user inputs using yfinance library

data = yf.download(ticker,start=start_date,end=end_date)
#since date displays as index we need to convert it into column
#add Date as a column to the dataframe
data.insert(0,'Date',data.index,True)
#reset the index
data.reset_index(drop=True,inplace=True)
#print the object or view data
st.write('Data from', start_date,'to',end_date)
st.write(data)

#plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select your specific date range on the sodebar, or zoom in on the plot and select your specific column")
fig = px.line(data, x='Date', y= data.columns, title='Closing price of the stock',width=1000, height=600)
st.plotly_chart(fig)


#add a select box to select column from data; except date select rest all the drop down functions so [1:]
column = st.selectbox('Select the column to be used for forecasting',data.columns[1:])

#subsetting the data
#in below line we update the data
data = data[['Date',column]]
st.write("Selected Data")
st.write(data)

#Adifular Test Check for Stationarity: If stationary will take ARIMA model
st.header('Is data Stationary?')
#st.write("**Note:** If p-value is less than 0.05, then data is stationary")
#st.write(adfuller(data[column])[1])
st.write(adfuller(data[column])[1] < 0.05)

#Lets Decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model ='additive', period=12)
st.write(decomposition.plot())
#make same plot in plotly
st.plotly_chart(px.line(x=data['Date'],y=decomposition.trend,title='Trend',width=1000, height=400,labels={'x':'Date', 'y':'Price'}).update_traces(line_color="Blue"))
st.plotly_chart(px.line(x=data['Date'],y=decomposition.seasonal,title='Seasonality',width=1000,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color="green"))
st.plotly_chart(px.line(x=data['Date'],y=decomposition.resid,title='Residuals',width=1000,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color="Red",line_dash='dot'))

#Lets run the model
#user input for 3 parameters of the model and seasonal order
p =st.slider('Select the value of p',0,5,2)
d =st.slider('Select the value of d',0,5,1)
q =st.slider('Select the value of q',0,5,2)
seasonal_order = st.number_input('Select the value of seasonal p',0,24,12)

model = sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

#print model summary
st.header('Model Summary')
st.write(model.summary())
st.write("---")
#predict the values with user input values
st.write("<p style='color:green; font-size: 50px, font-weight: bold;'>Forecasting the data</p>",unsafe_allow_html=True)
forcast_period = st.number_input("## Enter forcast period in days",value=10)

#predict all the values for the forcast period and the current dataset
predictions = model.get_prediction(start=len(data),end=len(data)+forcast_period-1)
predictions = predictions.predicted_mean
#st.write(len(predictions))
#add index to results dataframe as dates
predictions.index = pd.date_range(start=end_date, periods=len(predictions),freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0,'Date',predictions.index)
predictions.reset_index(drop=True,inplace=True)
st.write('## Predictions', predictions)
st.write('## Actual Data',data)
st.write("---")
#forcast_period = st.number_input('Select the number of days to forcast', 1,365,10)
#predict the future values

#lets plot the data
fig = go.Figure()
#add actual data to the plot
fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines',name='Actual',line=dict(color='blue')))
#add predicted data to the plot
fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['predicted_mean'], mode='lines',name='Predicted',line=dict(color='red')))
#set the title and axis labels
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price',width=1000, height=400)
#display the plot
st.plotly_chart(fig)

#Add buttouns to show and hide separate plots
show_plots=False
if st.button('Show Separate Plots'):
    if not show_plots:
        st.write(px.line(x=data['Date'], y=data[column],title='Actual',width=1200, height=400,labels={'x':'Date', 'y':'Price'}).update_traces(line_color="Blue"))
        st.write(px.line(x=predictions["Date"],y=predictions["predicted_mean"],title='Predicted',width=1200,height=400,labels={'x':'Date','y':'Price'}).update_traces(line_color="green"))
        show_plots = True
    else:
        show_plots = False
    #add hide plot button
    hide_plots = False
    if st.button("Hide Separated Plots"):
        if not hide_plots:
            hide_plots = True
        else:
            hide_plots =False
    st.write("---")
    
        