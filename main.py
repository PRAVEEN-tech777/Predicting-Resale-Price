import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd
from geopy.distance import geodesic
import statistics
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# -------------------------Reading the data on Lat and Long of all the MRT Stations in Singapore------------------------
try:
    data = pd.read_csv('mrt.csv')
    mrt_location = pd.DataFrame(data)
except FileNotFoundError:
    st.error("MRT stations data file not found.")

headers = {"Authorization": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjNjU2NGQxYjc4NTUwZDY1MzU2NTU5M2RjMjQ3ZTM3ZSIsImlzcyI6Imh0dHA6Ly9pbnRlcm5hbC1hbGItb20tcHJkZXppdC1pdC0xMjIzNjk4OTkyLmFwLXNvdXRoZWFzdC0xLmVsYi5hbWF6b25hd3MuY29tL2FwaS92Mi91c2VyL3Bhc3N3b3JkIiwiaWF0IjoxNzIyMTc0NDIyLCJleHAiOjE3MjI0MzM2MjIsIm5iZiI6MTcyMjE3NDQyMiwianRpIjoiaGFXYlBzU3pCU3BuSlU5QyIsInVzZXJfaWQiOjQxNzMsImZvcmV2ZXIiOmZhbHNlfQ.ONNTqRz09iBTJ0ZIwtjncnc6WWkYc69KJ3_pOOtZZFQ"}

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    page_icon="🏨",
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )

# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :blue[Domain :] Real Estate")

# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 87%)]")

    try:
        with st.form("form1"):
            street_name = st.text_input("Street Name")
            block = st.text_input("Block Number")
            floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
            lease_commence_date = st.number_input('Lease Commence Date', min_value=1960, max_value=2023, step=1)
            storey_range = st.text_input("Storey Range (Format: 'Value1 TO Value2')")

            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button:
                try:
                    with open(r"model.pkl", 'rb') as file:
                        loaded_model = pickle.load(file)
                    with open(r'scaler.pkl', 'rb') as f:
                        scaler_loaded = pickle.load(f)
                except FileNotFoundError:
                    st.error("Model or scaler file not found.")
                    st.stop()

                lease_remain_years = 99 - (2023 - lease_commence_date)

                try:
                    split_list = storey_range.split(' TO ')
                    float_list = [float(i) for i in split_list]
                    storey_median = statistics.median(float_list)
                except ValueError:
                    st.error("Invalid storey range format. Please use 'Value1 TO Value2'.")
                    st.stop()

                address = block + " " + street_name
                query_address = address
                query_string = f'https://www.onemap.gov.sg/api/common/elastic/search?searchVal={query_address}&returnGeom=Y&getAddrDetails=Y&pageNum=1'
                resp = requests.request("GET", query_string, headers=headers)

                origin = []
                data_geo_location = json.loads(resp.content)
                if data_geo_location['found'] != 0:
                    latitude = data_geo_location['results'][0]['LATITUDE']
                    longitude = data_geo_location['results'][0]['LONGITUDE']
                    origin.append((latitude, longitude))
                else:
                    st.error("Address not found.")
                    st.stop()

                mrt_lat = mrt_location['latitude']
                mrt_long = mrt_location['longitude']
                list_of_mrt_coordinates = [(lat, long) for lat, long in zip(mrt_lat, mrt_long)]

                list_of_dist_mrt = [geodesic(origin, dest).meters for dest in list_of_mrt_coordinates]
                min_dist_mrt = min(list_of_dist_mrt)

                cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

                new_sample = np.array(
                    [[cbd_dist, min_dist_mrt, np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
                new_sample = scaler_loaded.transform(new_sample[:, :5])
                new_pred = loaded_model.predict(new_sample)[0]
                st.write('## :green[Predicted resale price:] ', np.exp(new_pred))

    except Exception as e:
        st.error(f"An error occurred: {e}")
