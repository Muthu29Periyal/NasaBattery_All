from flask import Flask,render_template,request
import pickle as pkl
import numpy as np

app=Flask(__name__)
pol_reg=pkl.load(open('model/state_of_health_poly_reg.pkl','rb'))
mx1=pkl.load(open('model/state_of_health_minmax.pkl','rb'))
model_soh=pkl.load(open('model/state_of_health.pkl','rb'))
model_rul=pkl.load(open('model/remaining_useful_cycle.pkl','rb'))
mx2=pkl.load(open('model/remaining_useful_cycle_minmax.pkl','rb'))


@app.route('/')
def main():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/info')
def description():
    return render_template('info.html')

@app.route('/estimator')
def predict():
    return render_template('estimator.html')

@app.route('/result',methods=['POST','GET'])
def result():
    if request.method=='POST':
        a=request.form.get("vthreshold")
        c=request.form.get("dmax_time")
        b=request.form.get("dmax_vtime")
        x1=[[a,b]]
        input1=pol_reg.transform(mx1.transform(x1))
        SOH= float(model_soh.predict(input1))
        x2=[[a,b,c,SOH]]
        input2=mx2.transform(x2)
        RUL= float(model_rul.predict(input2))
        return render_template('result.html',SOH='{:2.4}'.format(SOH*100),RUL=round(RUL))
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
