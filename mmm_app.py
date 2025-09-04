import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose # Import seasonal_decompose
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.optimize import curve_fit, nnls, minimize, Bounds, LinearConstraint # Import for optimization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io # Import the io module
import warnings
warnings.filterwarnings('ignore')

# Import for tick formatting in plots
import matplotlib.ticker as mticker

# Import display and HTML for rendering the media report table (though st.write(..., unsafe_allow_html=True) is preferred)
from IPython.display import display, HTML


# --- Custom Functions (Copy these from your notebook or import them if in a separate file) ---

# You MUST include the definitions for these functions in your final .py file,
# either by copying them directly above this line or importing them from another local file.
# Examples of functions you'll need:
# - convert_indian_number
# - assign_fiscal_year
# - enhanced_adstock
# - weibull_saturation
# - apply_enhanced_transformations
# - ConstrainedLinearRegression
# - optimize_transformation_params
# - enhanced_mmm_analysis
# - calculate_roi_by_fiscal_year
# - calculate_media_effectiveness
# - plot_avp_with_holdout (Adapt to return Plotly figs)
# - plot_contribution_by_bucket (Adapt to return Plotly figs)
# - plot_fy_comparison (Adapt to return Plotly figs)
# - plot_fy_pie_charts (Adapt to return Plotly figs)
# - plot_media_response_curves (Adapt to return Plotly figs)
# - run_scenario
# - compare_scenarios
# - objective_function
# - run_optimization


# Placeholder for your custom functions (replace with actual code or imports)
def convert_indian_number(value):
    # Add your implementation here
    if isinstance(value, str):
        cleaned_value = value.replace(',', '').strip()
        if cleaned_value in ['-', ''] or cleaned_value.isspace():
            return np.nan
        try:
            return float(cleaned_value)
        except ValueError:
            return np.nan
    return value

# Assume other necessary functions are defined or imported (e.g., enhanced_adstock, weibull_saturation, enhanced_mmm_analysis, etc.)
# For the Streamlit app to run, you need to ensure all functions called below are defined or imported correctly.


# --- Streamlit App Structure ---

st.set_page_config(layout="wide")

st.title("Marketing Mix Modeling (MMM) App")

# --- Sidebar for Inputs ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'df_engineered' not in st.session_state:
    st.session_state['df_engineered'] = None
if 'seasonal_period' not in st.session_state:
    st.session_state['seasonal_period'] = 52 # Default seasonal period
if 'model_results' not in st.session_state:
    st.session_state['model_results'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'modeling_buckets' not in st.session_state:
     st.session_state['modeling_buckets'] = None # Store modeling buckets for later use


# --- Main Content Area ---
if uploaded_file is not None:
    # Load and preprocess data from the uploaded file
    try:
        # Simple loading with pandas, assuming similar preprocessing as your function
        data = pd.read_csv(uploaded_file)

        # Assuming 'Week_Ending' is the date column and 'Sales' is the target
        date_column = 'Week_Ending'
        target_variable = 'Sales'

        # Apply Indian number conversion to relevant columns if needed
        all_columns = data.columns.tolist()
        if date_column in all_columns:
            all_columns.remove(date_column)

        for col in all_columns:
             if data[col].dtype == 'object':
                 try:
                      data[col] = data[col].apply(convert_indian_number)
                 except Exception as e:
                      st.warning(f"Could not apply number conversion to column '{col}': {e}")


        # Convert date column and set index
        if date_column in data.columns:
            data[date_column] = pd.to_datetime(data[date_column], format='%d-%m-%Y %H:%M', errors='coerce', infer_datetime_format=True, cache=True)
            data = data.dropna(subset=[date_column]) # Drop rows where date conversion failed
            data = data.sort_values(date_column).set_index(date_column)
            data.index.name = 'Week_Ending' # Ensure index name is set
            st.sidebar.success("Data loaded and date index set successfully!")
            st.subheader("Raw Data Preview (with Date Index)")
            st.write(data.head())

            # Store loaded data in session state
            st.session_state['data'] = data.copy() # Store a copy
            st.session_state['target_variable'] = target_variable # Store target variable

        else:
            st.error(f"Date column '{date_column}' not found. Cannot proceed with analysis.")
            st.stop() # Stop the app if date column is missing


        # --- EDA Section ---
        st.sidebar.header("Exploratory Data Analysis (EDA)")
        if st.sidebar.button("Run EDA"):
            if st.session_state['data'] is not None:
                 # Assuming perform_streamlit_eda is defined (copied/imported) above
                 perform_streamlit_eda(st.session_state['data'].copy(), target_var=st.session_state['target_variable']) # Use data from session state
            else:
                 st.warning("Please upload data first.")


        # --- Feature Engineering Section ---
        st.sidebar.header("Feature Engineering")
        # Add controls for feature engineering options
        st.session_state['seasonal_period'] = st.sidebar.selectbox(
            "Seasonal Decomposition Period",
            [4, 13, 26, 52],
            index=[4, 13, 26, 52].index(st.session_state['seasonal_period']), # Set default based on session state
            key='seasonal_period_selectbox' # Add a unique key
        )

        # Placeholder for other FE controls (dummies, splits, supers) - Add Streamlit widgets for these
        st.sidebar.info("Add controls for Custom Dummies, Variable Splitting, Super Campaigns here.")
        # Example for dummy dates:
        # dummy_dates_input = st.sidebar.text_input("Custom Dummy Dates (YYYY-MM-DD, comma-separated):")


        if st.sidebar.button("Run Feature Engineering"):
             if st.session_state['data'] is not None:
                 with st.spinner("Running Feature Engineering..."):
                     # Assuming perform_streamlit_feature_engineering is defined (copied/imported) above
                     # You'll need to pass other FE options (dummies, splits, supers) to this function
                     st.session_state['df_engineered'] = perform_streamlit_feature_engineering(
                         st.session_state['data'].copy(), # Use data from session state
                         st.session_state['seasonal_period'],
                         target_var=st.session_state['target_variable']
                         # Pass other FE parameters here
                     )
                     # The perform_streamlit_feature_engineering function should also return modeling_buckets
                     # For now, we'll just assume df_engineered is returned.
                     # You might need to adapt your function to return both.
                     # Example adaptation: return df_engineered, modeling_buckets

                 if st.session_state['df_engineered'] is not None:
                     st.subheader("Engineered Features Preview:")
                     st.write(st.session_state['df_engineered'].head())
                     st.success("Feature Engineering complete!")
                     # Display Media Performance Report if perform_streamlit_feature_engineering generates it

             else:
                 st.warning("Please upload data first.")


        # --- Model Training Section ---
        st.sidebar.header("Model Training")
        # Add controls for Model Training
        model_type = st.sidebar.selectbox("Select Model", ["Ridge", "Lasso", "ElasticNet"], key='model_type_selectbox')
        holdout_weeks = st.sidebar.number_input("Holdout Weeks", min_value=1, value=12, key='holdout_weeks_input')
        enforce_positive = st.sidebar.checkbox("Enforce Positive Media Coefficients", value=True, key='enforce_positive_checkbox')

        if st.sidebar.button("Train Model"):
           if st.session_state['df_engineered'] is not None:
               with st.spinner("Training Model..."):
                   # Prepare model_choice dictionary
                   model_choice = {
                       'model': model_type.lower(),
                       'grid': {}, # Define default grids or add widgets for hyperparameter tuning
                       'reasons': [], # Add widgets for reasons if needed
                       'notes': '',
                       'timestamp': pd.Timestamp.utcnow().isoformat() + 'Z',
                       'ready_to_train': True
                   }

                   # Assuming enhanced_mmm_analysis is defined (copied/imported) above
                   # enhanced_mmm_analysis function expects df_features (which is df_engineered here)
                   # and MODEL_CHOICE, HOLDOUT_WEEKS, enforce_positive
                   # It should return results dictionary and scaler
                   # It should also return modeling_buckets or ensure it's set in global scope if needed

                   # Need to get modeling_buckets here if enhanced_mmm_analysis doesn't return it
                   # Or adapt enhanced_mmm_analysis to take modeling_buckets as input

                   # For now, let's assume modeling_buckets is generated/available after FE or passed
                   # If your FE function returns modeling_buckets, update the FE section call.
                   # Example: st.session_state['df_engineered'], st.session_state['modeling_buckets'] = perform_streamlit_feature_engineering(...)

                   # If modeling_buckets is not returned by FE, you might need to reconstruct it
                   # or ensure your enhanced_mmm_analysis function can handle it.

                   # Temporary placeholder for modeling_buckets if not from FE:
                   if st.session_state['modeling_buckets'] is None:
                        st.warning("Modeling buckets not found in session state. Attempting to reconstruct basic buckets.")
                        # Basic reconstruction (may not match exact buckets from notebook FE)
                        numeric_cols = st.session_state['df_engineered'].select_dtypes(include=[np.number]).columns.tolist()
                        target_var = st.session_state['target_variable']
                        if target_var in numeric_cols:
                             numeric_cols.remove(target_var)

                        # Simple heuristic for buckets (needs refinement based on your actual data/FE)
                        base_vars = [col for col in numeric_cols if col not in ['Discount1', 'Discount2'] and 'impression' not in col.lower() and 'click' not in col.lower() and '_Spend' not in col]
                        promo_vars = [col for col in numeric_cols if 'Discount' in col]
                        media_vars = [col for col in numeric_cols if ('impression' in col.lower() or 'click' in col.lower()) and '_Spend' not in col]

                        st.session_state['modeling_buckets'] = {
                           'base_vars': base_vars,
                           'promo_vars': promo_vars,
                           'media_vars': media_vars,
                           'target_var': target_var
                       }
                        st.write("Reconstructed Basic Modeling Buckets:", st.session_state['modeling_buckets'])


                   if st.session_state['modeling_buckets'] is not None:
                       # Pass modeling_buckets to enhanced_mmm_analysis if needed
                       results, scaler = enhanced_mmm_analysis(
                           st.session_state['df_engineered'].copy(), # Use engineered data
                           model_choice,
                           holdout_weeks,
                           enforce_positive
                           # Pass modeling_buckets here if your function expects it
                       )

                       if results:
                           st.session_state['model_results'] = results
                           st.session_state['scaler'] = scaler # Store scaler

                           st.subheader("Model Training & Analysis Results")
                           st.write("Model Performance Metrics:", results['metrics'])

                           # Display visualizations (assuming Plotly figures are returned)
                           if results['visualizations'].get('avp_chart'):
                               st.plotly_chart(results['visualizations']['avp_chart'])
                           if results['visualizations'].get('contribution_timeline'):
                               st.plotly_chart(results['visualizations']['contribution_timeline'])
                           if results['visualizations'].get('fy_contribution_bars'):
                               st.plotly_chart(results['visualizations']['fy_contribution_bars'])
                           if results['visualizations'].get('fy_contribution_pies'):
                                # Check if pie charts were generated (can be None)
                                if results['visualizations']['fy_contribution_pies']:
                                    st.plotly_chart(results['visualizations']['fy_contribution_pies'])
                                else:
                                    st.info("Fiscal Year Pie Charts not generated (likely insufficient data).")
                           if results['visualizations'].get('response_curves'):
                                # Check if response curves were generated (can be None)
                                if results['visualizations']['response_curves']:
                                    st.plotly_chart(results['visualizations']['response_curves'])
                                else:
                                     st.info("Media Response Curves not generated (likely no paid media).")


                           # Display analysis tables
                           if 'fy_roi' in results['analyses'] and not results['analyses']['fy_roi'].empty:
                                st.write("ROI by Fiscal Year:")
                                # Display formatted HTML table (assuming your function returns HTML or format here)
                                # If fy_roi is a DataFrame, display it:
                                st.dataframe(results['analyses']['fy_roi'])
                           else:
                                st.info("No Fiscal Year ROI data to display.")

                           if 'media_effectiveness' in results['analyses'] and not results['analyses']['media_effectiveness'].empty:
                                st.write("Media Effectiveness Metrics:")
                                st.dataframe(results['analyses']['media_effectiveness'])
                           else:
                                st.info("No Media Effectiveness data to display.")

                           if 'fy_contributions' in results['analyses'] and not results['analyses']['fy_contributions'].empty:
                                st.write("Contributions by Fiscal Year:")
                                st.dataframe(results['analyses']['fy_contributions'])
                           else:
                                st.info("No Fiscal Year Contribution data to display.")


                           st.success("Model Training complete!")

                       else:
                           st.error("Model training failed.")

           else:
              st.warning("Please run Feature Engineering first.")


        # --- Scenario Analysis Section ---
        st.sidebar.header("Scenario Analysis")
        # Add controls for Scenario Analysis
        st.sidebar.info("Scenario analysis controls will be here. Requires a trained model.")

        # Example Scenario Controls (will need to be integrated):
        # if st.session_state.get('model_results') is not None and st.session_state.get('df_engineered') is not None:
        #     st.sidebar.subheader("Define Scenario")
        #     media_channels_for_scenario = st.session_state['modeling_buckets'].get('media_vars', [])
        #     if media_channels_for_scenario:
        #         scenario_channel = st.sidebar.selectbox("Select channel to change:", [''] + media_channels_for_scenario)
        #         if scenario_channel:
        #             change_type = st.sidebar.radio("Change type:", ["Percentage change", "Fixed spend value"])
        #             if change_type == "Percentage change":
        #                 spend_change_pct = st.sidebar.number_input("Percentage change (+/-):", value=0.0) / 100.0
        #                 fixed_spend_value = None
        #             else:
        #                 fixed_spend_value = st.sidebar.number_input("Fixed spend value:")
        #                 spend_change_pct = 0.0
        #
        #             if st.sidebar.button("Run Scenario"):
        #                 with st.spinner(f"Running Scenario: {scenario_channel} change..."):
        #                     # Assuming run_scenario function is defined (copied/imported)
        #                     # run_scenario expects base_df, scenario_name, media_channel_to_change, spend_change_pct, fixed_spend_value
        #                     # It also needs access to the trained model, scaler, and transformation parameters (via global scope or passed)
        #                     # You'll need to ensure these are accessible to the run_scenario function in your .py file
        #
        #                     # Need to ensure adstock/weibull params are available to run_scenario
        #                     # If your enhanced_mmm_analysis returns them in results['params'], you can store them
        #                     # st.session_state['adstock_params_optimized'] = results['params']['theta']
        #                     # st.session_state['weibull_params_optimized'] = results['params']['weibull']
        #
        #                     scenario_result = run_scenario(
        #                         st.session_state['df_engineered'].copy(), # Base data for scenario
        #                         scenario_name=f"{scenario_channel} Change",
        #                         media_channel_to_change=scenario_channel,
        #                         spend_change_pct=spend_change_pct,
        #                         fixed_spend_value=fixed_spend_value
        #                         # Pass model, scaler, adstock/weibull params if run_scenario expects them
        #                     )
        #
        #                     if scenario_result:
        #                         st.subheader(f"Scenario: {scenario_result['scenario_name']} Results")
        #                         st.write(f"Total Predicted Sales: {scenario_result['predicted_sales'].sum():,.0f}")
        #                         st.write(f"Total Incremental Sales (vs Base): {scenario_result['incremental_sales'].sum():,.0f}")
        #                         st.write(f"Total Scenario Spend: {scenario_result['total_spend']:,.0f}")
        #                         st.write(f"Total Incremental Spend (vs Base): {scenario_result['incremental_spend']:,.0f}")
        #                         # Format and display ROI
        #                         roi_formatted = f"{scenario_result['scenario_roi']:.2%}" if not np.isinf(scenario_result['scenario_roi']) and not np.isnan(scenario_result['scenario_roi']) else str(scenario_result['scenario_roi'])
        #                         st.write(f"Scenario ROI: {roi_formatted}")
        #                     else:
        #                         st.error("Scenario analysis failed.")
        #             else:
        #                 st.info("Select a channel to define a scenario.")
        #     else:
        #         st.info("No media channels identified for scenario analysis.")
        # else:
        #    st.info("Train the model first to enable scenario analysis.")


        # --- Optimization Section ---
        st.sidebar.header("Optimization")
        # Add controls for Optimization
        st.sidebar.info("Optimization controls will be here. Requires a trained model.")

        # Example Optimization Controls (will need to be integrated):
        # if st.session_state.get('model_results') is not None and st.session_state.get('df_engineered') is not None:
        #      st.sidebar.subheader("Run Optimization")
        #      total_volume_budget = st.sidebar.number_input(
        #          "Total Volume Budget",
        #          value=st.session_state['df_engineered'][st.session_state['modeling_buckets'].get('media_vars', [])].sum().sum() if st.session_state.get('modeling_buckets') else 0.0
        #      )
        #      # Add controls for min/max percentage constraints per channel
        #      # min_pct = st.sidebar.number_input("Min % of historical volume:", value=0.0)
        #      # max_pct = st.sidebar.number_input("Max % of historical volume:", value=1.0)
        #      st.sidebar.info("Add detailed channel constraints here.")
        #
        #      if st.sidebar.button("Run Optimization"):
        #          with st.spinner("Running Optimization..."):
        #              # Assuming run_optimization function is defined (copied/imported)
        #              # run_optimization expects total_budget, spend_min_pct, spend_max_pct, channel_min_spend, channel_max_spend
        #              # It also needs access to the trained model, scaler, adstock/weibull params, original_df (via global or passed)
        #              # Ensure original_df (st.session_state['data']) and df_engineered (st.session_state['df_engineered']) are accessible

        #              # Need to pass necessary data and parameters to run_optimization
        #              # You might need to adapt your run_optimization function's signature
        #
        #              optimization_results = run_optimization(
        #                  total_budget=total_volume_budget,
        #                  spend_min_pct=0.0, # Replace with widget values
        #                  spend_max_pct=float('inf') # Replace with widget values
        #                  # Pass model, scaler, adstock/weibull params, original_df if run_optimization expects them
        #              )
        #
        #              if optimization_results and optimization_results['success']:
        #                  st.subheader("Optimization Results")
        #                  st.write("Optimal Media Volume Allocation:")
        #                  st.dataframe(optimization_results['optimal_allocation_df'])
        #                  st.write(f"Predicted Total Sales at Optimal Allocation: {optimization_results['predicted_sales']:,.0f}")
        #              elif optimization_results:
        #                  st.error(f"Optimization failed: {optimization_results.get('message', 'Unknown error')}")
        #              else:
        #                  st.error("Optimization did not return results.")
        # else:
        #      st.info("Train the model first to enable optimization.")


    except Exception as e:
        st.error(f"An error occurred during data loading or initial processing: {e}")

else:
    st.info("Upload a CSV file to get started with the MMM analysis.")
