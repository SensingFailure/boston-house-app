import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**
""")
st.write('---')

# Loads the Boston House Price Dataset
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('Per Capita Crime Rate (CRIM)', X.CRIM.min(), X.CRIM.max(), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('Proportion of Residential lots over 25K sq. ft. (ZN)', X.ZN.min(), X.ZN.max(), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('Non-retail Business Acres (INDUS)', X.INDUS.min(), X.INDUS.max(), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('Charles River Tract (CHAS)', X.CHAS.min(), X.CHAS.max(), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('Nitrogen Oxides Concentration (pp 10M) (NOX)', X.NOX.min(), X.NOX.max(), float(X.NOX.mean()))
    RM = st.sidebar.slider('Avg. Number of Rooms (RM)', X.RM.min(), X.RM.max(), float(X.RM.mean()))
    AGE = st.sidebar.slider('Owner-occupied units built prior to 1940 (AGE)', X.AGE.min(), X.AGE.max(), float(X.AGE.mean()))
    DIS = st.sidebar.slider('WM of Distance to 5 Boston Employment Centers (DIS)', X.DIS.min(), X.DIS.max(), float(X.DIS.mean()))
    RAD = st.sidebar.slider('Accessibility to Highways (RAD)', X.RAD.min(), X.RAD.max(), float(X.RAD.mean()))
    TAX = st.sidebar.slider('Property Taxt Rate per $10,000 (TAX)', X.TAX.min(), X.TAX.max(), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('Pupil-Teacher Ratio (PTRATIO)', X.PTRATIO.min(), X.PTRATIO.max(), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('Porportion of Black Population (B)', X.B.min(), X.B.max(), float(X.B.mean()))
    LSTAT = st.sidebar.slider('Lower Status of Population (LSTAT)', X.LSTAT.min(), X.LSTAT.max(), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
fig=shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
fig1=shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
