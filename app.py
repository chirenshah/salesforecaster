from flask import Flask,make_response,render_template,request
from numpy.testing._private.utils import _assert_no_gc_cycles_context
from pandas import get_dummies
from pandas_profiling import ProfileReport
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.preprocessing import RobustScaler
import keras
import tensorflow as tf
import matplotlib.dates as mdates
from flask import flash, redirect, url_for
from datetime import datetime, timedelta
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import plotly
import json

import plotly.graph_objs as go


UPLOAD_FOLDER = "C:/Users/shahc/OneDrive/Desktop/BE Project/uploads/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
#app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config)


Predictionprogress = "1"
progress = "False"
@app.route('/uploadajax', methods = ['POST' , 'GET'])
def uploadfile():
    try:
        if request.method == "POST":
            if request.files:
                file = request.files["myFile"]
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                global inputdf
                inputdf = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                #eda(file.filename)
                global filename 
                filename = file.filename
                global progress
                progress = "True"
        return "Success"
    except:
        progress = "True"

@app.route("/")
def home():
    return render_template("indexfinal1.html")

@app.route("/selection" , methods = ['POST' , 'GET'])
def selectionRender():
    return render_template('selection.html' , columns = inputdf.columns)

@app.route("/eda")
def eda(name = None):
    #Cleaning
    inputdf.fillna(0, inplace = True)
    #EDA
    report = ProfileReport(inputdf, title="EDA", html={'style': {'full_width':True}}, explorative=True, missing_diagrams={'bar': True})
    report.to_file("./templates/pandas_profiling_report.html")
    progress = "True"
    return render_template("pandas_profiling_report.html")

@app.route("/status" , methods = ['POST' , 'GET'])
def status():
    return progress

@app.route("/Predictionstatus" , methods = ['POST' , 'GET'])
def Predictionstatus():
    return Predictionprogress

@app.route("/edaDisplay" , methods = ['POST' , 'GET'])
def edaDisplay():
    return render_template("pandas_profiling_report.html")

# def parser(x):
#     return pd.datetime.strptime(x, dateFormat)

# def create_dataset(X, y, time_steps=1):
#     Xs, ys = [], []
#     for i in range(len(X) - time_steps):
#         v = X.iloc[i:(i + time_steps)].values
#         Xs.append(v)        
#         ys.append(y.iloc[i + time_steps])
#     return np.array(Xs), np.array(ys)

# @app.route("/timeseries" , methods = ['POST' , 'GET'])
# def lstm():
#     global Predictionprogress
#     df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "Sales.csv"), header=0, usecols=[time,target],parse_dates=[0], squeeze=True, date_parser=parser)
#     df['new' + time] = df[time]
#     df = df.set_index(time)
#     df = df.sort_index()
#     train_size = int(len(df) * 0.8)
#     test_size = len(df) - train_size
#     train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
#     global predictions_test
#     predictions_test =  test.copy()
#     f_columns = ['new' + time]
#     global f_transformer
#     global cnt_transformer

#     f_transformer = RobustScaler()
#     cnt_transformer = RobustScaler()

#     f_transformer = f_transformer.fit(train[f_columns].to_numpy())
#     cnt_transformer = cnt_transformer.fit(train[[target]])

#     train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
#     train[target] = cnt_transformer.transform(train[[target]])

#     test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
#     test[target] = cnt_transformer.transform(test[[target]])

#     time_steps = 10
#     # reshape to [samples, time_steps, n_features]
#     X_train, y_train = create_dataset(train, train[target], time_steps)
#     X_test, y_test = create_dataset(test, test[target], time_steps)
#     Predictionprogress = "2"
#     global model
#     model = keras.Sequential()
#     model.add(
#     keras.layers.Bidirectional(
#         keras.layers.LSTM(
#         units=128, 
#         input_shape=(X_train.shape[1], X_train.shape[2])
#         )
#     )
#     )
#     model.add(keras.layers.Dropout(rate=0.2))
#     model.add(keras.layers.Dense(units=1))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     global history
#     history = model.fit(
#         X_train, y_train, 
#         epochs=30, 
#         batch_size=32, 
#         validation_split=0.1,
#         shuffle=False
#     )
#     Predictionprogress = "3"
#     y_pred = model.predict(X_test)
#     y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))
#     y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
#     y_pred_inv = cnt_transformer.inverse_transform(y_pred)
#     fig, ax = plt.subplots(figsize=(6.4, 4.8))
#     # Add x-axis and y-axis
#     ax.scatter(df.index[0:len(y_train)], y_train_inv.flatten(),
#             color='purple',label='Previous Sales Data')
#     ax.scatter(df.index[len(y_train) :len(y_train) + len(y_test)], y_test_inv.flatten(),
#             color='green',label='Actual Sales Data')
#     ax.scatter(df.index[len(y_train): len(y_train) + len(y_test)], y_pred_inv.flatten(),
#             color='red',label='Predicted Data')
#     ax.xaxis.set_major_formatter(mdates.DateFormatter(dateFormat))
#     # Set title and labels for axes
#     ax.set(xlabel=time,
#         ylabel=target,
#         title="Sales Data")
#     plt.legend()
#     plt.savefig("./static/Graph.png")
#     Predictionprogress = "4"
#     return "hey"

def graphgenerator(model,test_generator,close_train,close_test,date_test,date_train):
    prediction = model.predict_generator(test_generator)
    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))

    trace1 = go.Scatter(
        x = date_train,
        y = close_train,
        mode = 'markers',
        name = 'History'
    )
    trace2 = go.Scatter(
        x = date_test,
        y = close_test,
        mode = 'markers',
        name = 'Actual Value'
    )

    trace3 = go.Scatter(
        x = date_test,
        y = prediction,
        mode='markers',
        name = 'Prediction'
    )
    fig = [trace1,trace2,trace3]
    # fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    return fig

def salespredictor(model,test_generator,close_train,close_test,date_test,prediction_dates,prediction_list):
    prediction = model.predict_generator(test_generator)
    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))

    trace2 = go.Scatter(
        x = date_test,
        y = close_test,
        mode = 'markers',
        name = 'Actual  Value'
    )

    trace3 = go.Scatter(
        x = date_test,
        y = prediction,
        mode='markers',
        name = 'Prediction'
    )
    
    trace1 = go.Scatter(
        x = prediction_dates,
        y = prediction_list,
        mode = 'markers',
        name = 'Forecast'
    )

    fig = [trace2,trace3,trace1]
    # fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    return fig

def predict(num_prediction, model):
    look_back = 15
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
    return prediction_list

def predict_dates(num_prediction,incrementdelta):
    last_date = df[time].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1,freq=incrementdelta).tolist()
    return prediction_dates

@app.route("/timeseries" , methods = ['POST' , 'GET'])
def lstm():
    global df
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename),usecols=[time,target])
    df[time] = pd.to_datetime(df[time],format=dateFormat)
    df.set_axis(df[time], inplace=True)
    df = df.sort_index(axis=0)
    global close_data
    global Predictionprogress
    close_data = df[target].values
    close_data = close_data.reshape((-1,1))
    
    split_percent = 0.8
    split = int(split_percent*len(close_data))
    global close_train,close_test,date_train,date_test
    close_train = close_data[:split]
    close_test = close_data[split:]
    date_train = df[time][:split]
    date_test = df[time][split:]
    global timedeltacalc
    timedeltacalc = date_test.copy() 
    look_back = 1
    global train_generator,test_generator
    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)
    global model
    model = Sequential()
    model.add(
        LSTM(128,
        activation = "relu",
        input_shape=(look_back,1)),
    )
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    num_epochs = 25
    Predictionprogress = '2'
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1 ,shuffle=False)
    predictiongraph = graphgenerator(model,test_generator,close_train,close_test,date_test,date_train)
    Predictionprogress = '3'
    global graphJSON
    graphJSON = json.dumps(predictiongraph, cls=plotly.utils.PlotlyJSONEncoder)
    Predictionprogress = "4"
    return "hey"

@app.route("/getAttributes" , methods = ['POST' , 'GET'])
def get_attributes():
    global time,target,dateFormat
    if request.method == "POST":
            target = request.form.get("target")
            time = request.form.get("time")
            ts_or_lr = request.form.get("flag")
            dateFormat = request.form.get("format")
            if(ts_or_lr == "Time Series"):
                lstm()
            else:
                prediction()
    return '4'

@app.route("/prediction" , methods = ['POST' , 'GET'])
def prediction():
    try:
        global Predictionprogress
        Predictionprogress = "1"
        inputdf[time] = pd.to_datetime(inputdf[time],dateFormat)
        inputdf[time] = inputdf[time].map(datetime.toordinal)
        data1 = pd.get_dummies(inputdf)
        X=data1.drop(target,1)
        y=data1[[target]]
        X_train, X_test, y_train , y_test = train_test_split(X,y,test_size=0.05,random_state=0)
        unnormalised_X_time = X_test.copy()
        unnormalised_X_target = y_test.copy()
        print(unnormalised_X_time)
        print(unnormalised_X_target)
        global sc
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        regressor1 = HistGradientBoostingRegressor()
        #RMSE: 1.629386

        
        regressor2 = LGBMRegressor(random_state=0)
        #RMSE: 1.673168

        
        regressor3 = CatBoostRegressor(iterations=2000, random_state = 0, verbose = 200)

        
        regressor4 = XGBRegressor()

        regressor6 = RandomForestRegressor(n_estimators = 1000, random_state = 0)

        from sklearn.ensemble import GradientBoostingRegressor
        regressor7 = GradientBoostingRegressor(random_state=0)

        from sklearn import linear_model
        regressor8 = linear_model.BayesianRidge()

        from sklearn.svm import SVR
        regressor9 = SVR(kernel = 'rbf')

        from sklearn.neural_network import MLPRegressor
        regressor10 = MLPRegressor(random_state=0, max_iter=1000)

        from sklearn.ensemble import ExtraTreesRegressor
        regressor11 = ExtraTreesRegressor(n_estimators=1000, random_state=0)

        from sklearn.tree import DecisionTreeRegressor
        regressor12 = DecisionTreeRegressor(random_state = 0)
        #estimators = [('hist', regressor1), ('lgbm', regressor2), ('cb', regressor3), ('xgb', regressor4), ('nimbus', regressor5), ('gbr', regressor7), ('br', regressor8), ('svr', regressor9)]
        estimator = [('hist', regressor1), ('lgbm', regressor2), ('cb', regressor3), 
                    ('xgb', regressor4), ('rfr', regressor6), 
                    ('gbr', regressor7), ('br', regressor8), ('svr', regressor9)]
        weight = [4,3,1,1,1,1,1,1]
        global regressor
        regressor = VotingRegressor(estimators = estimator, weights = weight)
        Predictionprogress = "2"
        trace1 = go.Scatter(
            x = unnormalised_X_time[time],
            y = unnormalised_X_target[target],
            mode ="markers" ,
            name = "Actual Value"
        )
        #regressor = StackingRegressor(estimators=estimators,final_estimator=vregressor)
        regressor.fit(X_train, y_train)
        Predictionprogress = "3"
        y_pred = regressor.predict(X_test)
        # trace1 = go.Scatter(
        #     x = unnormalised_X_time[time],
        #     y = unnormalised_X_target[target],
        #     mode ="markers" ,
        #     name = "Actual Value"
        # )

        trace2 = go.Scatter(
            x = unnormalised_X_time[time],
            y = y_pred,
            mode ="markers" ,
            name = "Prediction"
        )
        global r , m
        Predictionprogress = "4"
        r = r2_score(y_test, y_pred)
        r = r*100
        m = mean_squared_error(y_test, y_pred)
        data = [trace1,trace2]
        global regressionJson 
        regressionJson = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        return "yes"
    except Exception as e:
        print(e)
        Predictionprogress = "5"
        return e

@app.route("/forecast_regression" ,  methods = ['POST' , 'GET'])
def forecast_regression():
    try:
        if request.method == "POST":
            dict1 = {}
            dataset = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "Sales.csv"))
            for element in request.form:
                if(dataset[element].dtype == np.int64):
                    value = np.int64(request.form[element])
                    temp = {
                        element:value
                    }
                elif(dataset[element].dtype == np.float64):
                    value = np.float64(request.form[element])
                    temp = {
                        element:value
                    }
                else:
                    value = request.form[element]
                    temp = {
                        element:value
                    }
                dict1.update(temp)
            dataset_withoutTarget = dataset.drop(target,1)
            newdf = dataset_withoutTarget.append(dict1, ignore_index="TRUE")
            newdf_dummyvalue = pd.get_dummies(newdf)
            df_finalValue = sc.transform(newdf_dummyvalue.tail(1))
            predict = regressor.predict(df_finalValue)
            return str(predict[0])
    except Exception as e:
        print(e)
        return "oopsie"

# @app.route("/forecast_time_series" ,  methods = ['POST' , 'GET'])
# def forecast_time_series():
#     try:
#         # time = time
#         if request.method == "POST":
#             time_value = request.form.get(time)
#             copy_test = pd.Dataframe()
#             copy_test = copy_test.append(predictions_test[-10:])
#             name = 'new'+time
#             #add time at the end of the copy of test 
#             copy_test = copy_test.append({
#                 name:datetime.strptime(time_value, dateFormat),
#                 target:1
#             },ignore_index=True)
#             f_columns = [name]
#             #normalising the copy of test
#             copy_test.loc[:, f_columns] = f_transformer.transform(copy_test[f_columns].to_numpy())
#             copy_test[target] = cnt_transformer.transform(copy_test[[target]])
#             print(copy_test.shape)
#             #reshaping the copy of test into [samples,time_steps,features]
#             predx,predy = create_dataset(copy_test,copy_test[target],10)
#             #predicting the answer for all values in test along with the new value
#             b = model.predict(predx)
#             print(b)
#             #denormalising the predicted value
#             b_inv = cnt_transformer.inverse_transform(b)
#             return str(b_inv[-1][0])
#     except Exception as e:
#         print(e)
#         return "oopsie"

@app.route("/forecast_time_series" ,  methods = ['POST' , 'GET'])
def forecast_time_series():
    if request.method == "POST":
        time_value = request.form.get(time)
    time_value = int(time_value)
    forecast = predict(time_value, model)
    forecast_dates = predict_dates(time_value,incrementdelta)
    forecastgraph = salespredictor(model,test_generator,close_train,close_test,date_test,forecast_dates,forecast)
    forecastJSON = json.dumps(forecastgraph, cls=plotly.utils.PlotlyJSONEncoder)
    return forecastJSON

@app.route("/resultsRegression" , methods = ['POST' , 'GET'])
def results_regression():
    inputdf = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], "Sales.csv"))
    return render_template("prediction_regression.html",r2_score = r, mse = m , columns = inputdf.columns.drop(target) , Target = target , plot = regressionJson)

@app.route("/resultsTimeSeries" , methods = ['POST' , 'GET'])
def results_time_series():
    num_prediction = int(len(df)*0.001) + 1
    result = timedelta(days=0)
    i = 0
    length = len(timedeltacalc)
    global ans
    global incrementdelta
    try:
        while result < timedelta(days=1):
            result = timedeltacalc[length -1] - timedeltacalc[length - 1 -i]
            i = i+1
    except Exception as e:
        print(e)
    
    if(result/360 > timedelta(days=1)):
        ans = 'Years'
        incrementdelta = '1Y'
    elif(result/30 > timedelta(days=1)):
        ans = 'Months'
        incrementdelta = '1M'
    else:
        incrementdelta = '1D'
        ans = 'Days'
    value = str(num_prediction) + " " + ans
    return render_template("prediction_TimeSeries.html", columns = time , Target = target , plot = graphJSON, value = value)
    
# @app.route('/csv/')  
# def download_csv():  
#     #csv = 'foo,bar,baz\nhai,bai,crai\n'  
#     response = make_response("Train.csv")
#     cd = 'attachment; filename=mycsv.csv'
#     response.headers['Content-Disposition'] = cd 
#     response.mimetype='text/csv'

#     return response