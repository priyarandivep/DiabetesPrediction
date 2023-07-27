import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings("ignore")

# load the classifier and scaler
with open('model.pkl', "rb")as pkl:
     classifier=pickle.load(pkl)
    
with open("scaling.pkl", "rb") as sc:
     scaler= pickle.load(sc)


def main():
     st.header("Diabetes Prediction")
     left, right= st.columns(2)
     Pregnancies=left.number_input("Enter Pregnancies as whole number", step =1 , value=0)
     Glucose=right.number_input("Enter Glucose as whole number", step =1, value =0)
     BloodPressure=left.number_input("Enter Blood Pressure as whole number", step =1 , value=0)
     SkinThickness=right.number_input("Enter Skin Thickness  as whole number", step =1, value =0)
     Insulin =left.number_input("Enter Insulin  as whole number", step =1 , value=0)
     BMI=right.number_input("Enter BMI  as whole number", step =1, value=0)
     DiabetesPedigreeFunction =left.number_input("Enter Diabetes Pedigree Function  as decimal number", step =0.001 , value=0.00)
     Age=right.number_input("Enter Age  as whole number", step =1, value=0)

     Predict_Button= st.button("Am I Diabetic??")
     if Predict_Button:
          data= pd.DataFrame({"Pregnancies": Pregnancies, "Glucose":Glucose, "BloodPressure": BloodPressure, "SkinThickness": SkinThickness, 
                             "Insulin": Insulin , "BMI": BMI, "DiabetesPedigreeFunction": DiabetesPedigreeFunction, "Age": Age},  index=[0])
          
          scaled_data= scaler.transform(data)
          result= classifier.predict(scaled_data)

          if result[0]==0:
               st.success("You are not Diabetic")
          else:
               st.success("You are Diabetic")

if __name__=="__main__":
    main()