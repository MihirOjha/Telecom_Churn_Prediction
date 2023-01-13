import pickle
import streamlit as st
import pandas as pd
from PIL import Image

model_file = 'model.pkl'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
def main():
    image = Image.open('image1.jpeg')
    st.image(image, width=650)
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "From file"))
    st.sidebar.info("This app is created to predict Customer Churn")
    st.sidebar.markdown(
        "For the features: 'The customer has international plan :' and 'The customer has voice mail plan :'")
    st.sidebar.code("0 : No (False) \n1 : Yes(True)")
    if add_selectbox == 'Online':
        account_length = st.number_input('Number of months the customer has been with the current telco provider :', min_value=0, max_value=240, value=0)
        area_code = st.selectbox('"area_code_AAA" where AAA = 3 digit area code :', ['408', '415', '510'])
        international_plan = st.selectbox('The customer has international plan :', ['0', '1'])
        voice_mail_plan = st.selectbox('The customer has voice mail plan :', ['0', '1'])
        number_vmail_messages = st.slider('Number of voice-mail messages. :', min_value=0, max_value=60, value=0)
        total_day_minutes = st.slider('Total minutes of day calls :', min_value=0, max_value=360, value=100)
        total_day_calls = st.slider('Total day calls :', min_value=0, max_value=200, value=50)
        total_eve_minutes = st.slider('Total minutes of evening calls :', min_value=0, max_value=400, value=200)
        total_eve_calls = st.slider('Total number of evening calls :', min_value=0, max_value=200, value=100)
        total_night_minutes = st.slider('Total minutes of night calls :', min_value=0, max_value=400, value=200)
        total_night_calls = st.slider('Total number of night calls :', min_value=0, max_value=200, value=100)
        total_intl_minutes = st.slider('Total minutes of international calls :', min_value=0, max_value=60, value=0)
        total_intl_calls = st.slider('Total number of international calls :', min_value=0, max_value=20, value=0)
        number_customer_service_calls = st.slider('Number of calls to customer service :', min_value=0, max_value=10, value=0)
        output = ""
        output_prob = ""
        input_dict={'Account length':account_length,'Area code': area_code,'International plan':international_plan,'Voice mail plan':voice_mail_plan\
            ,'Number vmail messages':number_vmail_messages,'Total day minutes':total_day_minutes,'Total day calls':total_day_calls\
            ,'Total eve minutes':total_eve_minutes,'Total eve calls':total_eve_calls,'Total night minutes':total_night_minutes\
            ,'Total night calls':total_night_calls,'Total intl minutes':total_intl_minutes,'Total intl calls':total_intl_calls\
            ,'Customer service calls':number_customer_service_calls}

        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            X = input_df
            y_pred = model.predict_proba(X)[0][1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
            st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))


    if add_selectbox == 'From file':
        uploaded_file = st.file_uploader("Upload a CSV file containing customer information", type=["csv"])
        if uploaded_file is not None:
            # Specify the data types of the columns
            dtypes = {'Account length': float,
                      'Area code': str,
                      'International plan': str,
                      'Voice mail plan': str,
                      'Number vmail messages': float,
                      'Total day minutes': float,
                      'Total day calls': float,
                      'Total eve minutes': float,
                      'Total eve calls': float,
                      'Total night minutes': float,
                      'Total night calls': float,
                      'Total intl minutes': float,
                      'Total intl calls': float,
                      'Customer service calls': float,
                      'Churn': str}

            input_df = pd.read_csv(uploaded_file, dtype=dtypes)

            # Drop the 'state' and 'area code' columns
            input_df = input_df.drop(columns=['state', 'area code'])

            # Convert 'yes'/'no' in 'International plan' and 'Voice mail plan' columns to 1/0
            input_df['International plan'] = input_df['International plan'].map({'yes': 1, 'no': 0})
            input_df['Voice mail plan'] = input_df['Voice mail plan'].map({'yes': 1, 'no': 0})

            # Convert 'True'/'False' in 'Churn' column to 1/0
            input_df['Churn'] = input_df['Churn'].map({'True': 1, 'False': 0})

            if st.button("Predict"):
                X = input_df.drop(columns=['Churn'])
                y_pred = model.predict_proba(X)[0][1]
                churn = y_pred >= 0.5
                output_prob = float(y_pred)
                output = bool(churn)
                st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))



if __name__ == '__main__':
    main()