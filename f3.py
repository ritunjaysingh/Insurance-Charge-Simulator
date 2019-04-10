from flask import Flask, flash, render_template, session, redirect, url_for, session, request
from flask_wtf import FlaskForm
from wtforms import (StringField, BooleanField, DateTimeField,
                     RadioField,SelectField,TextField,
                     TextAreaField,SubmitField)
from werkzeug import secure_filename
from wtforms.validators import DataRequired
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from wtforms.validators import DataRequired
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class InfoForm(FlaskForm):
    age = StringField("Please Enter Your Age : ",validators=[DataRequired()])
    sex = RadioField("Please Choose your Gender : ",choices = [('male','Male'),('female','Female')])
    bmi = StringField("Please Enter your bmi : ",validators=[DataRequired()])
    children = StringField("Please Enter Your no. of children : ",validators=[DataRequired()])
    smoker = RadioField("Are you a smoker : ",choices=[('yes','Yes'),('no','No')])
    ep = RadioField("Do you have any existing plan : ",choices=[('True','True'),('False','False')])
    income = RadioField("Please Enter Your Income Range : ",choices=[('low','Low'),('medium','Medium'),('high','High')])
    algo = SelectField("Please Choose An Algorithm to train your model : ",choices=[('ran','RandomForestRegressor'),('svr','SupportVectorRegressor'),('dt','DecisionTreeRegressor')] )
    submit = SubmitField("submit")

@app.route('/', methods=['GET', 'POST'])
def index():

    form = InfoForm()
    if form.validate_on_submit():
        session['age'] = form.age.data
        session['sex'] = form.sex.data
        session['bmi'] = form.bmi.data
        session['children'] = form.children.data
        session['smoker'] = form.smoker.data
        session['ep'] = form.ep.data
        session['income'] = form.income.data
        session['algo'] = form.algo.data

        if session['ep'] == 'True' :
            session['ep'] = 1
        else :
            session['ep'] = 0
        
        df2 = pd.DataFrame({'age':[session['age']],'sex':[session['sex']], 'bmi':[session['bmi']],'children':[session['children']],
        'smoker':[session['smoker']],'existing_plan':[session['ep']],'income':[session['income']]})
        
        df2.to_csv('C:/Users/ritunjay.singh/Desktop/Pyt.csv', sep=',')
        return redirect(url_for("predict"))

    return render_template('index.html', form=form)

@app.route('/predict')
def predict(): 
    df1 = pd.read_csv("C:/Users/ritunjay.singh/Desktop/Pyt.csv")
    df1 = df1[['age','sex','bmi','children','smoker','existing_plan','income']]
    df = pd.read_excel("C:/Users/ritunjay.singh/Desktop/in.xlsx")

    X = df.iloc[:,0:7].values
    y = df.iloc[:,7:8].values
    X_test = df1.iloc[:,0:7].values
    y_test = df1.iloc[:,7:8].values
    lr_X = LabelEncoder()
    X[:,1]=lr_X.fit_transform(X[:,1])
    X[:,4]=lr_X.fit_transform(X[:,4])
    X[:,6]=lr_X.fit_transform(X[:,6])

    lr_X_test = LabelEncoder()
    X_test[:,1]=lr_X_test.fit_transform(X_test[:,1])
    X_test[:,4]=lr_X_test.fit_transform(X_test[:,4])
    X_test[:,6]=lr_X_test.fit_transform(X_test[:,6])

    oh = OneHotEncoder(categorical_features=[6])
    X = oh.fit_transform(X).toarray()
    X_test = oh.transform(X_test).toarray()

    sr = StandardScaler()
    X = sr.fit_transform(X)
    X_test = sr.transform(X_test)

    if session['algo'] == 'ran' :
        rf = RandomForestRegressor(n_estimators=100,random_state=0)
        rf.fit(X,y.ravel())
        y_pred =rf.predict(X_test)
    elif session['algo'] == 'svr' :
        svr = SVR(kernel = 'rbf')
        svr.fit(X,y.ravel())
        y_pred =svr.predict(X_test)
    else:
        dr = DecisionTreeRegressor()
        dr.fit(X,y.ravel())
        y_pred =dr.predict(X_test)

    return render_template('mm.html', prediction=round(y_pred[0],2))


if __name__ == '__main__':
    app.run(debug=True)
