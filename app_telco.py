import streamlit as st 
import pickle
import pandas as pd
import numpy as np


st.sidebar.title("Churn Probability of a Single Customer")
html_temp = """
<div style="background-color:green;padding:10px">
<h1 style="color:white;text-align:center;">Churn Prediction App </h1>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

tenure = st.sidebar.slider("Tenure : Months of customer stayed",1,72,33,step=1)
MonthlyCharges = st.sidebar.slider("Charge : Amount charged to customer",18.25,118.75,50.00,step = 0.05)
Contract=st.sidebar.radio("Contract term", ('Month-to-month', 'One year', 'Two year'))
OnlineSecurity=st.sidebar.radio("Customer`s online security", ('No', 'Yes', 'No internet service'))
InternetService=st.sidebar.radio("Customer’s internet service provider", ('DSL', 'Fiber optic', 'No'))
TechSupport=st.sidebar.radio("Customer has tech support?", ('No', 'Yes', 'No internet service'))

my_dict = { 'tenure' : tenure,
'MonthlyCharges' : MonthlyCharges,
'Contract' : Contract,
'OnlineSecurity' : OnlineSecurity,
'InternetService' : InternetService,
'TechSupport' : TechSupport}
df = pd.DataFrame.from_dict([my_dict])
columns = ['tenure','MonthlyCharges','OnlineSecurity_No','OnlineSecurity_No internet service',
'OnlineSecurity_Yes','InternetService_DSL','InternetService_Fiber optic',
'InternetService_No','Contract_Month-to-month','Contract_One year',
'Contract_Two year','TechSupport_No', 'TechSupport_No internet service','TechSupport_Yes']
df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
#st.title("ssss")
#st.sidebar.table(df.T)

with open('my_model_out.pkl', 'rb') as file:  
    model = pickle.load(file)

run = st.sidebar.button("RUN")
if run:
	prediction = model.predict(df)[0]
	if prediction == 1:
		prediction = "Churn Yes !"
		st.sidebar.warning(prediction)
	else:
		prediction = "Churn No"
		st.sidebar.success(prediction)

st.cache()
dfshow = pd.read_csv("df_out.csv",index_col=0)

if st.checkbox('Select Random Customers'):
	num = st.slider("Number of random Customers to be shown",1,dfshow.shape[0],1,step=1)
	selection = dfshow.iloc[np.random.randint(dfshow.shape[0], size=num)]
	st.success(f"Churn probability of {num} randomly selected customer")
	st.table(selection)

#if st.checkbox('click see top 5'):
	#st.table(dfshow.head())


if st.checkbox('Top Customers to Churn'):
	num_top = st.slider("Number of top customer to Churn",1,dfshow.shape[0],1,step=1)
	selection_top = dfshow.iloc[dfshow['Churn Probability'].sort_values(ascending=False)[:num_top].index.values]
	st.warning(f"Top {num_top} Customers to Churn")
	st.table(selection_top)

if st.checkbox('Top Customers to Loyal'):
	num_top = st.slider("Number of top customer to Loyal",1,dfshow.shape[0],1,step=1)
	selection_top = dfshow.iloc[dfshow['Churn Probability'].sort_values(ascending=True)[:num_top].index.values]
	st.success(f"Top {num_top} Customers to Loyal")
	st.table(selection_top)
#"Top Customers to Churn"
#"Top Customers to Loyal"


#st.table(dfshow.head())



