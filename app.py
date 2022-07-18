from msilib.schema import tables
from tkinter import CENTER, LEFT
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import csv
model =pickle.load(open('model.pkl','rb'))
app= Flask(__name__)    

@app.route("/")
def hello():
    return render_template("index.html") 
@app.route("/", methods=['POST'])
def predict_fraud():
    step=int(request.form.get('step'))
    type=request.form.get('type') 
    amount=float(request.form.get('amount'))
    oldbalanceOrg=float(request.form.get('oldbalanceOrg'))
    newbalanceOrig=float(request.form.get('newbalanceOrig'))    
    newbalanceDest=float(request.form.get('newbalanceDest'))
    oldbalanceDest=float(request.form.get('oldbalanceDest'))  
    errorBalanceOrig=newbalanceOrig+amount-oldbalanceOrg
    errorBalanceDest=oldbalanceDest+amount-newbalanceDest
    if(type=='CASH_IN' or type=='PAYMENT' or type== 'DEBIT'):
        return "TRANSACTION IS NOT FRAUDULENT"
    if(type=="CASH_OUT"):
        type=1
    else:
        type=0
    result=model.predict(np.array([step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,\
        newbalanceDest,errorBalanceOrig, errorBalanceDest]).reshape(1,9))
    
    if(result==1):
        mk="TRANSACTION IS FRAUDULENT"
    else:
        mk="TRANSACTION IS NOT FRAUDULENT"
    return render_template("index.html", predict_fraud=mk)

@app.route('/predict_csv', methods=['POST','GET'])  
def predict_csv():
    if request.method=='POST':
        # f=request.files['csvFile']
        # file=f.read()
        # csv_file=csv.reader(file)
        # df=pd.read_csv(csv_file)
        df = pd.read_csv(request.files.get('csvFile'))
        df['isFraud']= [0]*len(df)
        df['Probability of Fraud']=[0.0]*len(df)
        df.loc[df.type=='TRANSFER ','type']=0
        df.loc[df.type=='CASH_OUT','type']=1 
        df.type.astype(int) 
        for i in range(0,len(df)):
            if model.predict(np.array([df['step'][i],df['type'][i],df['amount'][i],df['oldbalanceOrg'][i],
            df['newbalanceOrig'][i],df['oldbalanceDest'][i],df['newbalanceDest'][i],df['newbalanceOrig'][i]
            -df['oldbalanceOrg'][i]+df['amount'][i], df['oldbalanceDest'][i]-df['newbalanceDest'][i]+df['amount'][i]]).reshape(1,9))==1:
                df['isFraud'][i]=1
            df['Probability of Fraud'][i]=model.predict_proba(np.array([df['step'][i],df['type'][i],df['amount'][i],df['oldbalanceOrg'][i],
            df['newbalanceOrig'][i],df['oldbalanceDest'][i],df['newbalanceDest'][i],df['newbalanceOrig'][i]
            -df['oldbalanceOrg'][i]+df['amount'][i], df['oldbalanceDest'][i]-df['newbalanceDest'][i]+df['amount'][i]]).reshape(1,9))[0][1]
        pf=df
        pf.loc[pf.type==0,'type']='TRANSFER'
        pf.loc[pf.type==1,'type']='CASH_OUT' 
        pf.type=pf.type.astype(str) 
    return render_template('data.html', tables=[pf.to_html(col_space=130,header=True,justify=LEFT,bold_rows=True, border=10)],titles=[''])
        
if __name__=="__main__":    
    app.run(debug=True)     