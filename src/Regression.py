import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime as dt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
from sklearn.preprocessing import PolynomialFeatures

def read_dataset():
    #Reading values from csv file 
    df = pd.read_csv("CurrencyPredictionData.csv") 
    input_values = np.array(df.iloc[:,0]); 
    input_values = input_values.reshape(-1,1)
    output_values = np.array(df.iloc[:,1]); 
    output_values = output_values.reshape(-1,1)
    return df

def data_visualisation(df):
    new_col_names = ['Time Series', 'inr', 'pkr', 'cny', 'kwd', 'aed', 'lkr', 'chf', 'eur', 'all', 'dzd',
                     'aoa', 'xcd', 'ars', 'amd', 'awg', 'shp', 'aud', 'azn', 'bsd', 'bhd', 'bdt', 'bbd',
                     'byn', 'bzd', 'xof', 'bmd', 'btn', 'bob', 'bam', 'bwp', 'brl', 'bnd', 'bgn', 'bif', 'cve']
    df.columns = new_col_names
    df.columns = map(str.upper, df.columns)
    df.rename(columns=lambda x:x+'_USD', inplace=True)
    df.rename(columns={'TIME SERIES_USD':'Time Series'}, inplace=True)
    print(df[['Time Series','INR_USD']])
    df['Time Series'] = pd.to_datetime(df['Time Series'])
    df['month'] = df['Time Series'].dt.month
    df['year'] = df['Time Series'].dt.year
    df['month_year'] = df['Time Series'].dt.to_period('M')
    df_groupby_currency = df.groupby('month_year').INR_USD.mean().reset_index()
    x = df_groupby_currency['month_year'].astype(str)
    y = df_groupby_currency['INR_USD']

    plt.figure(figsize=(8,4))
    plt.plot(x, y)
    plt.title("Exchange Rate: INR/USD")
    plt.xlabel("Month")
    plt.ylabel("Exchange Rate")
    plt.show()
        
    df['Time Series'] = pd.to_datetime(df['Time Series'])
    df['Time Series']=df['Time Series'].map(dt.datetime.toordinal)
    
    input_values = np.array(df.iloc[:,0]); 
    input_values = input_values.reshape(-1,1)
    
    output_values = np.array(df.iloc[:,1]); 
    output_values = output_values.reshape(-1,1)
    
    return input_values, output_values

def plot_predictions(input_feature_X, output_Y, X_test, Y_predicted, title):
    print("Plotting the graph ....")
    plt.figure(figsize=(20,8))
    
    x_date_value = []
    for numeric_value in input_feature_X:
        x_date_value.append(dt.datetime.fromordinal(numeric_value))
    
    plt.scatter(x_date_value, output_Y, c="red", label="Training Data")
    
    x_test_date_value = []
    for numeric_value in X_test:
        x_test_date_value.append(dt.datetime.fromordinal(numeric_value))
    plt.plot(x_test_date_value, Y_predicted, linewidth=4, color="blue", label="Predictions")
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
    
    # Modify year intervals
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=730))
    
    # Puts x-axis labels on an angle
    plt.gca().xaxis.set_tick_params(rotation = 30)  

    # Changes x-axis range
    left = dt.date(2010, 12, 1)
    right = dt.date(2018, 6, 1)

    plt.gca().set_xbound(left, right)
    
    plt.xlabel('Year') 
    plt.ylabel('Exchange Rate') 
    plt.title(title)
    plt.legend()    
    plt.show()
    print(title)
    print("------------------------------")


def perform_linear_regression(input_X, output_Y):

    x_train, x_test, y_train, y_test = train_test_split(input_X, output_Y, train_size=0.8, test_size=0.2, random_state=1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    ts_cross_val = TimeSeriesSplit(n_splits=5)
    cv = cross_val_score(model, x_train, y_train, cv = ts_cross_val, scoring= "neg_mean_squared_error")
    print("Cross validation - ",cv)
    print("R-squared of training data is: " + str(model.score(x_train, y_train)))
    print("R-squared of testing data is: " + str(model.score(x_test, y_test)))
        
    plt.figure(figsize=(20,8))
    y_pred = model.predict(x_test)
    r2 = metrics.r2_score(y_test, y_pred)
    print("R2 score : " + str(r2))
    print("Mean absolute error - " +str(metrics.mean_absolute_error(y_test, y_pred)))
    print("Mean squared error - " +str(metrics.mean_squared_error(y_test, y_pred)))    
    plt.plot(y_pred, "y", label="prediction", linewidth=2.0)
    plt.plot(y_test, "g", label="real_values", linewidth=2.0)
    plt.legend(loc="best")
    plt.title("Linear regression")
    plt.xlabel("Time")
    plt.ylabel("Exchange Rate")
    plt.show()
    plot_predictions(input_X, output_Y, x_test, y_pred, 'Linear Regression') 
    print('----------------- Linear Regression -----------------')
    
def perform_polynomial_linear_regression(input_X, output_Y):

    Xpoly = PolynomialFeatures(5).fit_transform(input_X)
    x_train, x_test, y_train, y_test = train_test_split(Xpoly, output_Y, train_size=0.8, test_size=0.2, random_state=1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    ts_cross_val = TimeSeriesSplit(n_splits=5)
    cv = cross_val_score(model, x_train, y_train, cv = ts_cross_val, scoring= "neg_mean_squared_error")
    print("Cross validation - ",cv)
    print("R-squared of training data is: " + str(model.score(x_train, y_train)))
    print("R-squared of testing data is: " + str(model.score(x_test, y_test)))
        
    plt.figure(figsize=(20,8))
    y_pred = model.predict(x_test)
    r2 = metrics.r2_score(y_test, y_pred)
    print("R2 score : " + str(r2))
    print("Mean absolute error - " +str(metrics.mean_absolute_error(y_test, y_pred)))
    print("Mean squared error - " +str(metrics.mean_squared_error(y_test, y_pred)))    
    plt.plot(y_pred, "y", label="prediction", linewidth=2.0)
    plt.plot(y_test, "g", label="real_values", linewidth=2.0)
    plt.legend(loc="best")
    plt.title("Linear regression")
    plt.xlabel("Time")
    plt.ylabel("Exchange Rate")
    plt.show() 
    print('----------------- Polynomial Linear Regression -----------------')
    
def perform_lasso_ridge_regression(input_feature_X, output_Y, Ci_range, model_name):
    for Ci in Ci_range:
        if model_name == "Lasso":
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=1/(2*Ci))
        else:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1/(2*Ci))
        
        x_train, x_test, y_train, y_test = train_test_split(input_feature_X, output_Y, train_size=0.8, test_size=0.2, random_state=1)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        ts_cross_val = TimeSeriesSplit(n_splits=5)
        cv = cross_val_score(model, x_train, y_train, cv = ts_cross_val, scoring= "neg_mean_squared_error")
        print("Cross validation - ",cv)
        print("R-squared of training data is: " + str(model.score(x_train, y_train)))
        print("R-squared of testing data is: " + str(model.score(x_test, y_test)))
        r2 = metrics.r2_score(y_test, y_pred)
        print("R2 score : " + str(r2))
        print("Mean absolute error - " +str(metrics.mean_absolute_error(y_test, y_pred)))
        print("Mean squared error - " +str(metrics.mean_squared_error(y_test, y_pred)))
        title = model_name+" regression with C = "+str(Ci)
        print(title)
        print("Coefficient values - ", model.coef_)
        print("----")
        print("Intercept value - ", model.intercept_)
        plot_predictions(input_feature_X, output_Y, x_test, y_pred, title)
        
def perform_kfold_regression_varying_c(input_feature_X, output_Y, kfold, Ci_range, model_name):
    mean_error=[]; std_error=[]
    temp=[]
    from sklearn.model_selection import KFold
    for Ci in Ci_range:
        kf = KFold(n_splits=kfold)
        for train, test in kf.split(input_feature_X):
            if model_name == "Lasso":
                from sklearn.linear_model import Lasso
                model = Lasso(alpha=1/(2*Ci))
            else:
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1/(2*Ci))
            model.fit(input_feature_X[train], output_Y[train])
            ypred = model.predict(input_feature_X[test])
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(output_Y[test],ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
              
    plt.title(model_name+" regression with k = "+str(kfold)+" folds"+" and C ")    
    plt.errorbar(Ci_range,mean_error,yerr=std_error)
    plt.xlabel('Ci-range')
    plt.ylabel('Mean square error')
    plt.legend('STD')
    plt.show()

def perform_kfold_regression(input_feature_X, output_Y, Ci, kf_range, model_name):
    mean_error=[]; std_error=[]
    temp=[]
    from sklearn.model_selection import KFold
    for i in kf_range:
        kf = KFold(n_splits=i)
        for train, test in kf.split(input_feature_X):
            if model_name == "Lasso":
                from sklearn.linear_model import Lasso
                model = Lasso(alpha=1/(2*Ci))
            else:
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1/(2*Ci))
            
            model.fit(input_feature_X[train], output_Y[train])
            ypred = model.predict(input_feature_X[test])
            from sklearn.metrics import mean_squared_error
            temp.append(mean_squared_error(output_Y[test],ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        if(i==5):
            print('Mean & variance of 5 estimates for - '+model_name)
            print('Mean - ', mean_error[1])
            print('SD - ', std_error[1])
    plt.title(model_name+" regression with varying K-folds and C = " +str(Ci))            
    plt.errorbar(kf_range,mean_error,yerr=std_error)
    plt.xlabel('K-split')
    plt.ylabel('Mean square error')
    plt.xlim((0,120))
    plt.legend('STD')
    plt.show()
    
def compare_models(input_feature_X, output_Y):
    Xpoly = PolynomialFeatures(5).fit_transform(input_feature_X)
    x_train, x_test, y_train, y_test = train_test_split(Xpoly, output_Y, train_size=0.8, test_size=0.2, random_state=1)
    model_poly = LinearRegression()
    model_poly.fit(x_train, y_train)
    y_pred_poly = model_poly.predict(x_test) 

    from sklearn.linear_model import Lasso
    model = Lasso(alpha=1/(2*0.001))
    x_train, x_test, y_train, y_test = train_test_split(input_feature_X, output_Y, train_size=0.8, test_size=0.2, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    ts_cross_val = TimeSeriesSplit(n_splits=5)
    print("----- Lasso ------")    
    cv = cross_val_score(model, x_train, y_train, cv = ts_cross_val, scoring= "neg_mean_squared_error")
    print("Cross validation - ",cv)
    print("R-squared of training data is: " + str(model.score(x_train, y_train)))
    print("R-squared of testing data is: " + str(model.score(x_test, y_test)))
    r2 = metrics.r2_score(y_test, y_pred)
    print("R2 score : " + str(r2))
    print("Mean absolute error - " +str(metrics.mean_absolute_error(y_test, y_pred)))
    print("Mean squared error - " +str(metrics.mean_squared_error(y_test, y_pred)))
    print("----- Lasso ------")
    
    from sklearn.linear_model import Ridge
    model_ridge = Ridge(alpha=1/(2*0.001))
    x_train, x_test, y_train, y_test = train_test_split(input_feature_X, output_Y, train_size=0.8, test_size=0.2, random_state=1)
    model_ridge.fit(x_train, y_train)
    y_pred_ridge = model_ridge.predict(x_test)
    ts_cross_val = TimeSeriesSplit(n_splits=5)
    cv = cross_val_score(model, x_train, y_train, cv = ts_cross_val, scoring= "neg_mean_squared_error")
    print("----- Ridge ------")
    print("Cross validation - ",cv)
    print("R-squared of training data is: " + str(model_ridge.score(x_train, y_train)))
    print("R-squared of testing data is: " + str(model_ridge.score(x_test, y_test)))
    r2 = metrics.r2_score(y_test, y_pred)
    print("R2 score : " + str(r2))
    print("Mean absolute error - " +str(metrics.mean_absolute_error(y_test, y_pred_ridge)))
    print("Mean squared error - " +str(metrics.mean_squared_error(y_test, y_pred_ridge)))
    print("----- Ridge ------")
    
    print("-----Dummy Regressor------")    
    from sklearn.dummy import DummyRegressor
    x_train, x_test, y_train, y_test = train_test_split(input_feature_X, output_Y, train_size=0.8, test_size=0.2, random_state=1)
    model_dummy = DummyRegressor()
    model_dummy.fit(x_train, y_train)
    y_pred_dummy = model_dummy.predict(x_test)
    cv = cross_val_score(model_dummy, x_train, y_train, cv = ts_cross_val, scoring= "neg_mean_squared_error")
    print("Cross validation - ",cv)
    print("R-squared of training data is: " + str(model_dummy.score(x_train, y_train)))
    print("R-squared of testing data is: " + str(model_dummy.score(x_test, y_test)))
    r2 = metrics.r2_score(y_test, y_pred)
    print("R2 score : " + str(r2))
    print("Mean absolute error - " +str(metrics.mean_absolute_error(y_test, y_pred_dummy)))
    print("Mean squared error - " +str(metrics.mean_squared_error(y_test, y_pred_dummy)))
    print("-----Dummy Regressor------")
    x_train, x_test, y_train, y_test = train_test_split(input_feature_X, output_Y, train_size=0.8, test_size=0.2, random_state=1)
    model_linear = LinearRegression()
    model_linear.fit(x_train, y_train)
    y_pred_linear = model_linear.predict(x_test)
    
    
    plt.figure(figsize=(20,8))
    x_test_date_value = []
    for numeric_value in x_test:
        x_test_date_value.append(dt.datetime.fromordinal(numeric_value))
    plt.plot(x_test_date_value, y_pred, linewidth=2, color="blue", label="Lasso")
    plt.plot(x_test_date_value, y_pred_ridge, linewidth=4, color="red", label="Ridge")
    plt.plot(x_test_date_value, y_pred_linear, linewidth=2, color="green", label="Linear")
    plt.plot(x_test_date_value, y_pred_dummy, linewidth=2, color="black", label="Dummy")
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
    
    # Modify year intervals
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=730))
    
    # Puts x-axis labels on an angle
    plt.gca().xaxis.set_tick_params(rotation = 30)  

    # Changes x-axis range
    left = dt.date(2010, 12, 1)
    right = dt.date(2018, 6, 1)

    plt.gca().set_xbound(left, right)
    
    plt.xlabel('Year') 
    plt.ylabel('Exchange Rate') 
    plt.title('Comparison of all models')
    plt.legend(bbox_to_anchor=(1.05, 1))    
    plt.show()
    print("------------------------------")
    
    
def main():
    #Read dataset from csv file and store it in variables
    dataframe = read_dataset()
    input_values, output_values = data_visualisation(dataframe)
    perform_linear_regression(input_values, output_values)
    perform_polynomial_linear_regression(input_values, output_values)
    
    Ci_range = [0.0001, 0.0005, 0.001, 0.005, 1, 5, 10]
    model_name= "Lasso"
    perform_lasso_ridge_regression(input_values, output_values, Ci_range, model_name)
    kf_range = [2, 5, 10, 25, 50, 100]
    perform_kfold_regression(input_values,output_values, 0.001, kf_range, model_name)
    perform_kfold_regression_varying_c(input_values,output_values, 50, Ci_range, model_name)
    
    Ci_range = [0.0001, 0.0005, 0.001, 0.005, 1, 5, 10]
    Ci_range = [0.0001, 0.1, 0.5, 1, 5, 10, 50, 100]
    model_name= "Ridge"
    perform_lasso_ridge_regression(input_values, output_values, Ci_range, model_name)
    kf_range = [2, 5, 10, 25, 50, 100]
    perform_kfold_regression(input_values,output_values, 100, kf_range, model_name)
    perform_kfold_regression_varying_c(input_values,output_values, 50, Ci_range, model_name)
    
    compare_models(input_values, output_values)    

if __name__ == "__main__":
    main()
