import pickle as pkl
import numpy as np
import streamlit as st
st.title("Salary Prediction Model")

filepath = "./salary_model.sav"
model = pkl.load(open(filepath, "rb"))

def pred(x): 
    x=np.array(x).reshape(1,-1)
    result = model.predict(x) #print(result)
    return result[0]

def main():
    age = st.number_input("Age: ")
    new_data = [age] 
    if st.button("Predict"):
        st.write("Estimated Salary: ",pred (new_data))

if __name__ =="__main__":
    main()
