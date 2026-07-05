import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Enterprise Valuator", page_icon="💎", layout="centered")
st.title("💎 Automated Real Estate Valuation Interface")
st.markdown("Query your optimized operational pipelines using the vector matrix controls below.")
st.markdown("---")

@st.cache_resource
def load_cached_production_binaries():
    pipeline = joblib.load('house_model_pipeline.pkl')
    neighborhoods = joblib.load('neighborhoods.pkl')
    conditions = joblib.load('sale_conditions.pkl')
    return pipeline, neighborhoods, conditions

try:
    model, NEIGHBORHOODS, CONDITIONS = load_cached_production_binaries()
except FileNotFoundError:
    st.error("❌ Critical Failure: Serialization binaries missing. Run `train_and_plot.py` first to initiate files.")
    st.stop()

with st.form("production_user_input_gate"):
    st.markdown("### 📋 Property Parameter Vector Matrix Elements")
    col1, col2 = st.columns(2)
    
    with col1:
        overall_qual = st.slider("Material/Finish Quality Level (1-10)", 1, 10, 6)
        gr_liv_area   = st.number_input("Living Area Size Above Ground (sq ft)", value=1500, step=50)
        total_bsmt_sf = st.number_input("Basement Construction Size Footprint (sq ft)", value=800, step=50)
        lot_area     = st.number_input("Total Property Lot Size Dimensions (sq ft)", value=8000, step=500)
        
        # Raw dropdown options displayed naturally to user
        raw_neighborhood = st.selectbox("Geographic Location Neighborhood", sorted([n.upper() for n in NEIGHBORHOODS]))
        
    with col2:
        year_built   = st.slider("Original Foundation Placement Year", 1880, 2010, 1990)
        year_remod   = st.slider("Remodeling / Addition Year", 1950, 2010, 1990)
        yr_sold      = st.slider("Target Sale Assessment Transaction Year", 2006, 2010, 2008)
        
        garage_cars   = st.selectbox("Garage Boundary Vehicle Capacity Size", [0, 1, 2, 3, 4], index=2)
        full_bath    = st.selectbox("Full Bathrooms Count Units", [0, 1, 2, 3, 4], index=2)
        half_bath    = st.selectbox("Half Bathrooms Count Units", [0, 1, 2], index=0)
        bsmt_full_bath = st.selectbox("Basement Full Bathrooms Count", [0, 1, 2], index=0)
        fireplaces   = st.selectbox("Functional Fireplaces Distribution Density", [0, 1, 2, 3], index=0)
        
        raw_sale_cond = st.selectbox("Market Transaction Conditions Setting", sorted([c.upper() for c in CONDITIONS]))
        
    submit = st.form_submit_button("🔮 Compute Pipeline Valuation Forecast")

if submit:
    # Build single feature mapping row dictionaries
    payload = {
        'GrLivArea': gr_liv_area, 'TotalBsmtSF': total_bsmt_sf, 'LotArea': lot_area,
        'YearBuilt': year_built, 'OverallQual': overall_qual, 'GarageCars': garage_cars,
        'FullBath': full_bath, 'HalfBath': half_bath, 'BsmtFullBath': bsmt_full_bath,
        'Fireplaces': fireplaces, 'Neighborhood': raw_neighborhood.lower().strip(), # Pipeline expects clean lowercase tokens
        'SaleCondition': raw_sale_cond.lower().strip(), 'YrSold': yr_sold, 'YearRemodAdd': year_remod
    }
    
    input_df = pd.DataFrame([payload])
    
    # Calculate runtime engineered features on live inputs array matching training conditions
    input_df['TotalSF'] = input_df['GrLivArea'] + input_df['TotalBsmtSF']
    input_df['TotalBath'] = input_df['FullBath'] + (0.5 * input_df['HalfBath']) + input_df['BsmtFullBath']
    input_df['HouseAge'] = input_df['YrSold'] - input_df['YearBuilt']
    input_df['IsRemodeled'] = (input_df['YearRemodAdd'] != input_df['YearBuilt']).astype(int)
    
    # Forward processed payload maps vector through the serialized model weights pipeline object
    raw_log_prediction = model.predict(input_df)[0]
    final_inferred_valuation = np.expm1(raw_log_prediction)
    
    st.success("### Pipeline Inference Complete!")
    st.metric(label="Calculated Asset Market Valuation Forecast", value=f"${final_inferred_valuation:,.2f}")