import streamlit as st
import joblib

svm_clf = joblib.load("svm_clf_model.joblib")
svm_clf = joblib.load("knn_clf_model.joblib")
svm_clf = joblib.load("rf_clf_model.joblib")


st.title("Iris' spice predict")

clf = st.sidebar.selectbox("#### choose classifier",["KNN","SVM","Random Forest"])

s1 = st.slider("花萼length",3.0,8.0,5.8) # 5.8是預設值
s2 = st.slider("花萼width",2.0,5.0,3.5) # 3.5是預設值
s3 = st.slider("花瓣length",1.0,7.0,4.5) # 4.5是預設值
s4 = st.slider("花瓣width",0.1,2.6,1.2) # 5.8是預設值

labels = ['setosa', 'versicolor', 'virginica']

if clf == "KNN":
    clf_model = knn_clf
elif clf == "SVM" : 
    clf_model = svm_clf
else : 
    clf_model = rf_clf


if st.button("do forcast"):
    X = [[s1,s2,s3,s4]]
    y = clf_model.predict(X)
    st.write("### forcast result",labels[y[0]])

