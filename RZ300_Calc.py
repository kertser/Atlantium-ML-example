import streamlit as st
from pickle import load
from tensorflow import keras
import plotly.express as px
import pandas as pd
import numpy as np

# Putting title
st.title("RZB-300 Calculator, based on ML/ANN:")
st.caption('calculated values for the following settings:')

valTitle = '<p style="font-family:sans-serif; color:Green; font-size: 25px;"><u>Input the key parameters:</u></p>'
st.sidebar.markdown(valTitle, unsafe_allow_html=True)

# Slider for the Power
Power = st.sidebar.slider("Power per module [kW]:", min_value=0.5, max_value=3.2, value=3.2, step=0.1, help="Power [kW]")

# Slider for the Flow Rate
Flow = st.sidebar.slider("Flow Rate [m^3/h]:", min_value=20, max_value=500, value=100, step=1, help="Flow Rate [m^3/h]")

# Sliders for the UVT
UVT254 = st.sidebar.slider("UVT at 254nm [%-1cm]:", min_value=40, max_value=96, value=92, step=1, help="UV-Transmittance at 254nm [%-1cm]")
UVT215 = st.sidebar.slider("UVT at 215nm [%-1cm]:", min_value=40, max_value=96, value=92, step=1, help="UV-Transmittance at 215nm [%-1cm]")

# Select the number of Lamps
NLamps = st.sidebar.selectbox("Select the system configuration: ", ['RZB-300x1', 'RZB-300x2', 'RZB-300x3', 'RZB-300x4'])
NLamps = int(NLamps[-1])

# Fetching and Printing:
data = pd.DataFrame({
    "Power [kW]": [Power],
    "Flow Rate [m^3/h]": [Flow],
    "UVT254 [%-1cm]": [UVT254],
    "UVT215 [%-1cm]": [UVT215],
    'Lamps [#]': [NLamps]
})

st.table(data.assign(hack='').set_index('hack').style.format('{:.2f}'))

# --- ANN part --- #

# Load the model
model = keras.models.load_model('SavedModel_RED/REDmodel.h5')
scaler = load(open('SavedModel_RED/scaler.pkl', 'rb'))

# Single testing point:
#88.6,63.33,257.611562,23.18,6.008833333,2,70.98111298

UVT254 = np.exp(-UVT254/100) # [%-1cm]
UVT215 = np.exp(-UVT215/100) # [%-1cm]
Flow = Flow # [m^3/hr]
UVS = 23.2 # [mJ/cm^2] - UV sensitivity aka D1Log
Power = Power # [kW]
N_Lamps = int(NLamps)

X_vector = scaler.transform(pd.DataFrame([UVT254,UVT215,Flow,UVS,Power,N_Lamps,1]).T)[:,:6]

# Dummy values added before the inverse transform due to the scaler preferences
RED = scaler.inverse_transform([1,1,1,1,1,1,float(model.predict(X_vector))])[6]

st.markdown('**RED = {} [mJ/cm^2]**'.format(round(float(RED),1)))

UVT = []
X_vector = []

### --- RED vs. UVT chart --- ###

for uvt_value in range(40, 96): # Drawing chart for %T=254nm
    X_vector.append([np.exp(-uvt_value/100), UVT215, Flow, UVS, Power, N_Lamps, 1])
    UVT.append(uvt_value)

X_vector = scaler.transform(pd.DataFrame(X_vector))[:,:6]
rows, cols = (len(X_vector), 6)
arr = pd.DataFrame([[0]*cols]*rows)
arr['y'] = pd.DataFrame(model.predict(X_vector))
REDs = scaler.inverse_transform(arr)[:,6]

UVTplotData = pd.DataFrame({'UVT':UVT,'RED':REDs})

fig = px.scatter(UVTplotData, trendline='lowess', x='UVT', y='RED', title='RED vs UVT254 at fixed flow rate')
fig.update_traces(marker_size=3)
fig.data[1].line.color = 'red'
fig.data[1].line.width = 0.7
fig.data[1].line.dash = 'dash'
st.plotly_chart(fig)

### --- RED vs. Flow chart --- ###
Flows = []
X_vector = []

for flow_value in range(100, 400, 10): # Drawing chart for %T=254nm
    X_vector.append([UVT254, UVT215, flow_value, UVS, Power, N_Lamps, 1])
    Flows.append(flow_value)

X_vector = scaler.transform(pd.DataFrame(X_vector))[:,:6]
rows, cols = (len(X_vector), 6)
arr = pd.DataFrame([[0]*cols]*rows)
arr['y'] = pd.DataFrame(model.predict(X_vector))
REDs = scaler.inverse_transform(arr)[:,6]

FlowPlotData = pd.DataFrame({'Flow':Flows,'RED':REDs})

fig2 = px.scatter(FlowPlotData, trendline='lowess', x='Flow', y='RED', title='RED vs Flow at fixed UVT')
fig2.update_traces(marker_size=3)
fig2.data[1].line.color = 'red'
fig2.data[1].line.width = 0.7
fig2.data[1].line.dash = 'dash'
st.plotly_chart(fig2)

#col1, col2 = st.columns(2)
#col1.plotly_chart(fig)
#col2.plotly_chart(fig2)