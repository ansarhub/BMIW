
...# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 20:37:35 2022

@author: junaid ansar
"""

from flask import Flask, render_template, request,session

import pickle
import numpy as np

model = pickle.load(open('bmi.pkl', 'rb'))
model1 = pickle.load(open('svc.pkl', 'rb'))

app = Flask(__name__)
app.secret_key="abs"



@app.route('/home')
def man():
    return render_template('home.html')


@app.route('/home.html', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['d']
    
    
    #from sklearn.preprocessing import StandardScaler
    #sc=StandardScaler()
    
    #sc.fit_transform([[data2,data1]])
    #arr =np.array([[data2,data1]])
    
    #arr=arr.append(arr,data1)
    #H=sc.fit_transform(arr)
    
    
    
    
    '''from sklearn.model_selection import cross_val_score    
    from sklearn import metrics
    cross_val_score(model, data2, data1, cv=5, scoring='f1_macro')  
    arr =np.array([[data2,data1]])
    '''
    
    arr =np.array([[data2,data1]])
    pred = model.predict(arr)
    pre=int(pred)
    session['res']=str(pre)
    session['da1']=str(data1)
    session['da2']=str(data2)
    session['dat3']=str(data3)
    #arr1=np.array([[data1,data2,pred]])
    #pred1=model1.predict(arr1)
    #pred1=str(pred)
    
    if  pre > int(data3):
        W="Compare your Actual weight and predict weight you are in under weight"
    elif pre < int(data3):
        W="Compare your Actual weight and your predicted weight you  are over weight"
    else:
        W="Compare your Actual weight  and yuor predicted weight your weight is normal"
    

    
    
    #score = model.score(arr,pred)
    #print("Test score: {0:.2f} %".format(100 * score))
    
    return render_template('home.html', data=pred,w1=W)
    #return render_template('bmi.html',dat=pred1)



@app.route('/bmi/')
def bmi():
    re=session.get('res')
    
    da1 =session.get('da1')
    da2 =session.get('da2')
    dat3 =session.get('dat3')
    #data2 = request.form['b']
    #data3 = data
    #da2=data2
    #datas3=float('da3')
    arr1= np.array([[da2, da1,dat3]])
    #pred = model.predict(arr)
    pred1=model1.predict(arr1)
    print(re,da1,da2)
    return render_template('bmi.html',dat=pred1)


@app.route('/bmr')
def bmr1():
    return render_template('BMR.html')

@app.route('/BMR.html',methods=['POST'])
def bmr():
    W=session.get('res')
    H =session.get('da1')
    g=session.get('da2')
    H1=int(H)
    W1=int(W)
    a=request.form['age']
    a1=int(a)
    AL = request.form['al']
    session['al1']=str(AL)
    #g= request.form['b']
    #bmr=0
    if g== '1':
        bmr=(10* W1) +(6.25*H1)-(5*a1)+5
        session['bm']=bmr 
    elif g == '0':
        bmr=(10*W1 )+ ( 6.25*H1)-(5*a1 )-161
        session['bm']=bmr 
        
    
    #print( "Your basel metabolic rate is:"+ str( bmr)+".")
    return render_template('BMR.html',result=bmr)




@app.route('/dnc/')
def dnc():
    al =session.get('al1')
    bmr =session.get('bm')
    bmr1=int(bmr)
    ALI=0
   
    if al=='1':
        ALI=1.2
        # Activity Level Index ALI=1.2 
    elif al =='2':
        ALI=1.375
    elif al=='3':
        ALI=1.46
    elif al=='4':
        ALI=1.725
    elif al=='5' :
        ALI=1.9
 # daily calories needed
    DCN= int (bmr1 * ALI)
    return render_template('dnc.html',res=DCN)






if __name__ == "__main__":
    app.run(debug=True)



