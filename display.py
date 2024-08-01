import pandas as pd
import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Singapore Flat Resale",
    page_icon="🇸🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add a header with a cool image
st.markdown(
    """
    <style>
        .header {
            background-color: #000000;
            padding: 5px;
            text-align: center;
            border-radius: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header img {
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 100px;
            object-fit: cover;
        }
        .title-text {
            font-size: 2em; /* Large font size */
            color: #FFFFFF; /* Primary color */
            divider: "--";
            font-weight: bold;
        }
        .scroll-container {
            overflow-x: auto;
            padding: 10px;
        }
        
    </style>
    <div class="header">
        <h1 class="title-text">Singapore Flat Resale Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True,)

a = st.container(height=300)
with a:
    st.image("/Users/shanthakumark/Downloads/wp2449403.jpg",clamp=True,use_column_width=True)
    


tab1, tab2 = st.tabs(['Prediction Model', 'Model Architecture'])

@st.cache_resource
def load_all_data():
    with open("/Users/shanthakumark/Desktop/Sharing/pickle_files/flat_type_u.pkl", 'rb') as flat_type_p:
        flat_type_o = pickle.load(flat_type_p)

    with open("/Users/shanthakumark/Desktop/Sharing/pickle_files/town_unique.pkl", 'rb') as town_unique_p:
        town_u = pickle.load(town_unique_p)

    with open("/Users/shanthakumark/Desktop/Sharing/pickle_files/lease_date.pkl", 'rb') as lease_com:
        lease = pickle.load(lease_com)

    with open("/Users/shanthakumark/Desktop/Sharing/pickle_files/street_name_u.pkl", 'rb') as street_name:
        street_nn = pickle.load(street_name)

    with open("/Users/shanthakumark/Desktop/Sharing/pickle_files/block_u.pkl", 'rb') as block_i:
        block_uniques = pickle.load(block_i)
        
    with open("/Users/shanthakumark/Desktop/Sharing/pickle_files/year_unique.pkl", 'rb') as year_u_p:
        year_uniques = pickle.load(year_u_p)

    with open("/Users/shanthakumark/Desktop/Sharing/pickle_files/model_rf_1.pkl", 'rb') as model_pred:
        model_ = pickle.load(model_pred)

    with open("/Users/shanthakumark/Desktop/Sharing/pickle_files/flat_type_transformed.pkl", 'rb') as flat_trans:
        flat_to_trans = pickle.load(flat_trans)
    
    with open("/Users/shanthakumark/Desktop/Sharing/pickle_files/street_name_transformed.pkl", 'rb') as street_trans:
        street_to_trans = pickle.load(street_trans)
    
    with open("/Users/shanthakumark/Desktop/Sharing/pickle_files/town_transformed.pkl", 'rb') as town_trans:
        town_to_trans = pickle.load(town_trans)
    
    return flat_type_o, town_u, lease, street_nn, block_uniques, year_uniques, model_, flat_to_trans, street_to_trans, town_to_trans

# Load all data once
flat_type, town_type, lease_u, street_nn, block_uniques, year_uniques, model_, flat_to_trans, street_to_trans, town_to_trans = load_all_data()

sorted_lease = pd.Series(lease_u).sort_values().to_list()
sorted_year = pd.Series(year_uniques).sort_values().to_list()

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.container(height=420):
            st.markdown('<h2 style="color: #3028aa;">Details</h2>', unsafe_allow_html=True)
            year_ = st.selectbox(label="Select The Year:", options=sorted_year)
            floor_area_sqm = st.number_input("Enter Floor Area sqm:", min_value=28.0, max_value=366.7)
            flat_type_selected = st.selectbox(label="Select the Flat Type", options=flat_type)
    
    with col2:
        with st.container(height=420):
            st.markdown('<h2 style="color: #3028aa;">Details</h2>', unsafe_allow_html=True)
            town = st.selectbox(label="Select the Town", options=town_type)
            lease_date = st.selectbox(label="Select the Lease Year", options=sorted_lease)
            street_name = st.selectbox(label="Select the Street Name:", options=street_nn)
            block_name = st.selectbox(label="Select the Block:", options=block_uniques)
    
    

    conp1 = st.container(height=260)
    with conp1:
        result_placeholder = st.empty()  # Create an empty container
        if st.button(label="Predict", key="predict_button"):
            # Collect inputs
            flat_type_en = flat_to_trans.get(flat_type_selected, flat_type_selected)  # Use default if not found
            street_en = street_to_trans.get(street_name, street_name)
            town_en = town_to_trans.get(town, town)

            x = [year_, floor_area_sqm, flat_type_en, town_en, lease_date, street_en, block_name]
            pred_y = np.array(x).reshape(1, -1)
            
            # Prediction
            predicted = model_.predict(pred_y)
            result_placeholder.write(f"<h3 style='color: #90EE90;'>The Resale Price : SGD {predicted[0]:,.2f}</h3>", unsafe_allow_html=True)
            df = pd.DataFrame({'year_':year_,'floor_area_sqm':floor_area_sqm,'flat_type_selected':flat_type_selected,'town':town,'lease_date':lease_date,'street_name':street_name,'block_name':block_name,'resale_predicted':predicted})
            with st.expander(label="See your data and price in table 📊"):
                st.dataframe(df)
        else:
            # Show default message when button is not clicked
            result_placeholder.write("<h3 style='color: #dc3545;'>Hey, enter values to predict</h3>", unsafe_allow_html=True)
    


    compp = st.container(height = 300)
    with compp:
        with st.container(height=420):
            st.image("https://wallpaperaccess.com/full/8642963.gif",channels='BGR')

# Add some CSS styling for better visual appeal
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .stContainer {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

with tab2:
    colt1,colt2,colt3 = st.columns([0.2,0.2,0.2])
    with colt1:
        with st.expander("COLLECTING DATA"):
            coo1,coo2 = st.columns([1,1.3])
            with coo1:
                    st.image("https://cdn-icons-png.flaticon.com/128/3270/3270865.png")
            with coo2:
                st.markdown('<h2 style="color: #FFFFFF;">COLLECTING DATA</h2>', unsafe_allow_html=True)
                st.write("Collected data from link: https://beta.data.gov.sg/collections/189/view")

    with colt2:
        with st.expander("Feature Engineering"):
            cop1,cop2 = st.columns([1,1.3])
            with cop1:
                st.image("/Users/shanthakumark/Downloads/Machine.png")
            with cop2:
                st.markdown('<h2 style="color: #FFFFFF;">Feature Engineering</h2>', unsafe_allow_html=True)
                st.write("""\n
                        Treated Outlier: columns with outlier treated with IQR Method 
                        \n
                        Removed Feature: Features with no use has been Removed 
                        \n
                        Created new feature: Created new features of year and month""")
                
    with colt3:
        with st.expander("DATA EXPLORATION"):
            tp1,tp2 = st.columns([1,1.3])
            with tp1:
                    st.image("/Users/shanthakumark/Downloads/Data Analytics.png")
            with tp2:
                st.markdown('<h2 style="color: #FFFFFF;">DATA EXPLORATION</h2>', unsafe_allow_html=True)
                st.write("""\n
                        Plotted Charts Like 📊: 
                        * boxplot 
                        * Bar plot for categorical variables
                        * Pie Chart
                        * Distplot for distribution """)
                

    cols1,cols2,cols3 = st.columns([1,1,1])
    with cols1:
        with st.expander("FEATURE IMPORTANCES AND TRAIN TEST SPLIT"):
            coo1,coo2 = st.columns([1,1.3])
            with coo1:
                    st.image("https://cdn.gogeticon.net/files/2193640/c63760792e72951834770eae95e7e190.png")
            with coo2:
                st.markdown('<h2 style="color: #FFFFFF;">FEATURE IMPORTANCES AND TRAIN TEST SPLIT</h2>', unsafe_allow_html=True)
                st.write("""\n 
                        * Used SelectKBest and f_regression for selecting feature importance
                        * Used Decision Tree regressor feature importances 
                        * Used Extra tree regressor feature importances
                        * Splitted X and Y
                        * Splitted train and test data""")
    with cols2:
        with st.expander("MODEL BUILDING"):
            s1,s2 = st.columns([1,1.3])
            with s1:
                    st.image("/Users/shanthakumark/Downloads/Regression.png")
            with s2:
                st.markdown('<h2 style="color: #FFFFFF;">MODEL BUILDING</h2>', unsafe_allow_html=True)
                st.write("""\n 
                        * Build Tree Model 
                            - Decision Tree Regressor
                            - Random Forest Regressor
                            - Extra Tree Regressor 
                        """)
    with cols3:
        with st.expander(label = "METRICS"):
            j1,j2 = st.columns([1,1.3])
            with j1:
                    st.image("https://cdn.sanity.io/images/3ogo9b9g/production/d38bf20971bacbcb3c91a6f4c7f1fd120281beb4-1006x1006.png?w=3840&q=75&fit=clip&auto=format")
            with j2:
                st.markdown('<h2 style="color: #FFFFFF;">METRICS</h2>', unsafe_allow_html=True)
                st.write("""\n 
                        * Metrics 
                            - Mean Squared Error
                            - Mean Absolute Error
                            - R2_score
                            - Absolute R2 Score 
                        """)
    st.divider()
    colp1,colp2,colp3 = st.columns([0.2,0.2,0.2])


    with colp2:
        with st.expander("FINAL MODEL FOR PREDICTION"):
            j1,j2 = st.columns([1,1.3])
            with j1:
                st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjcEclBbXyjilnVzUPDKPCDBN_mEbAM7VLFOdkf3vm2BwEzULP2blP6gzOjIoVL0JI_L19yPQjhjrp54fiugq0rPA3_3HWGGoAq5Xir7vpWR7taPNUOY52XunyBca_EYIup4tRxmehjjc-M/s1600-rw/Random+Forest.png")
            with j2:
                st.markdown('<h3 style="color: #FFFFFF;">RANDOM FOREST REGRESSOR</h3>', unsafe_allow_html=True)
                st.markdown('<h3 style="color: #90EE90;">r2_score: 97.5 %</h3>', unsafe_allow_html=True)       
        
