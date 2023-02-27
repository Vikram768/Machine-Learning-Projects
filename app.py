import streamlit as st
import pickle
import numpy as np
pipe = pickle.load(open("laptop.pkl","rb"))
df = pickle.load(open("df.pkl","rb"))

st.title("Laptop Preditor")

Company = st.selectbox("company brand",df["Company"].unique())
Type = st.selectbox("Type",df["TypeName"].unique())
Ram = st.selectbox("RAM(in GB)",[2,4,6,8,12,16,24,32,64])
Gpu = st.selectbox("Gpu",df["Gpu"].unique())
OpSys = st.selectbox("OpSys",df["OpSys"].unique())
Weight = st.number_input("weight of the laptop")
Touchscreen = st.selectbox("Touchscreen",["No","Yes"])
IPS = st.selectbox("IPS",["No","Yes"])
Screen_size = st.number_input("screen size")
resolution = st.selectbox("Screen Resolution",["1920x1080","1366x768",
"1600x900","3840x2160","3200x1800","2880x1800","2560x1600","2560x1440","2304x1440"])
Cpu = st.selectbox("Cpu",df["Cpu Brand"].unique())
SSD= st.selectbox("SSD(in GB)",[0,8,128,256,512,1024])
HDD = st.selectbox("HDD(in GB)",[0,128,256,512,1024,2048])

if st.button("Predict Price"):
    ppi = None
    if Touchscreen == "Yes":
        Touchscreen = 1
    else :
        Touchscreen = 0
    if IPS == "yes":
        IPS = 1
    else:
        IPS = 0
    x_res = resolution.split("x")[0]
    y_res = resolution.split("x")[1]

    ppi = ((x_res**2 + x_res**2)**0.5)/Screen_size
    query = np.array([Company,Type,Ram,Gpu,OpSys,Weight,Touchscreen,IPS,ppi,Cpu,SSD,HDD])
    query = query.reshape(1,12)
    st.title(pipe.predict(query))
