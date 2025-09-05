# mmm_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Marketing Mix Modeling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
    }
    .section-header {
        font-size: 2rem;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to convert Indian number format to float
def convert_indian_number(value):
    """Convert Indian number format string to float"""
    if isinstance(value, str):
        cleaned_value = value.replace(',', '').strip()
        if cleaned_value in ['-', ''] or cleaned_value.isspace():
            return np.nan
        try:
            return float(cleaned_value)
        except ValueError:
            return np.nan
    return value

# Fixed assign_fiscal_year function
def assign_fiscal_year(date_series, start_month=4):
    """
    Assign fiscal year based on date.
    Assumes date_series is a pandas Series or DatetimeIndex.
    """
    fiscal_years = []
    for date in date_series:
        if date.month >= start_month:
            fiscal_years.append(f"FY{date.year % 100:02d}/{(date.year + 1) % 100:02d}")
        else:
            fiscal_years.append(f"FY{(date.year - 1) % 100:02d}/{date.year % 100:02d}")
    return pd.Series(fiscal_years, index=date_series.index if hasattr(date_series, 'index') else None)

# Load and preprocess data
def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the marketing mix data"""
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # First, identify all columns that might contain Indian number format
    all_columns = data.columns.tolist()
    date_column = 'Week_Ending'
    
    if date_column in all_columns:
        all_columns.remove(date_column)
    
    # Convert all numeric columns (including Sales) - Optimized conversion
    for col in all_columns:
        if data[col].dtype == 'object':
            try:
                # Attempt to convert a sample to detect if it's numeric strings
                data[col].sample(min(10, len(data))).astype(str).str.replace(',', '').astype(float)
                # If the sample conversion works, apply to the whole column
                data[col] = data[col].apply(convert_indian_number)
            except (ValueError, AttributeError):
                pass
    
    # Handle missing values in Paid Search Impressions
    if 'Paid Search Impressions' in data.columns:
        missing_count = data['Paid Search Impressions'].isna().sum()
        if missing_count > 0:
            data['Paid Search Impressions'] = data['Paid Search Impressions'].fillna(0)
    
    # Convert date column
    if 'Week_Ending' in data.columns:
        data['Week_Ending'] = pd.to_datetime(data['Week_Ending'], format='%d-%m-%Y %H:%M', errors='coerce', infer_datetime_format=True)
        data = data.sort_values('Week_Ending').reset_index(drop=True)
    
    return data

# Perform comprehensive EDA
def perform_comprehensive_eda(data, target_var='Sales'):
    """Perform comprehensive exploratory data analysis"""
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # 1. Basic Information
    st.markdown('<h3 class="sub-header">1. Basic Dataset Information</h3>', unsafe_allow_html=True)
    st.write(f"Shape: {data.shape}")
    st.write(f"Columns: {list(data.columns)}")
    if 'Week_Ending' in data.columns:
        st.write(f"Date Range: {data['Week_Ending'].min()} to {data['Week_Ending'].max()}")
    st.write(f"Missing Values: {data.isnull().sum().sum()}")
    
    # Check if target variable is numeric
    if data[target_var].dtype == 'object':
        st.warning(f"Target variable '{target_var}' is not numeric after preprocessing.")
    
    # 2. Summary Statistics
    st.markdown('<h3 class="sub-header">2. Summary Statistics</h3>', unsafe_allow_html=True)
    numeric_df = data.select_dtypes(include=[np.number])
    st.write("Numeric Variables Summary:")
    st.write(numeric_df.describe())
    
    # Add skewness and kurtosis
    skewness = numeric_df.skew().to_frame('Skewness')
    kurtosis = numeric_df.kurtosis().to_frame('Kurtosis')
    stats_df = pd.concat([skewness, kurtosis], axis=1)
    st.write("Skewness and Kurtosis:")
    st.write(stats_df)
    
    # 3. Univariate Analysis
    st.markdown('<h3 class="sub-header">3. Univariate Analysis</h3>', unsafe_allow_html=True)
    numeric_cols = numeric_df.columns.tolist()
    
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=data, x=col, kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_ylabel("Frequency")
        
        # Add vertical lines for mean and median
        mean_val = data[col].mean()
        median_val = data[col].median()
        
        # Formatting based on column type
        if 'impressions' in col.lower() or 'clicks' in col.lower() or col == target_var:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
        elif 'discount' in col.lower():
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:,.2f}')
        ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:,.2f}')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    
    # 4. Bivariate Analysis
    st.markdown('<h3 class="sub-header">4. Bivariate Analysis</h3>', unsafe_allow_html=True)
    if target_var in numeric_cols:
        numeric_cols_for_scatter = numeric_cols.copy()
        numeric_cols_for_scatter.remove(target_var)
        
        for col in numeric_cols_for_scatter:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=data, x=col, y=target_var, ax=ax)
            ax.set_title(f"{target_var} vs {col}")
            
            # Formatting based on column type
            if 'impressions' in col.lower() or 'clicks' in col.lower():
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
            elif 'discount' in col.lower():
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
            
            if target_var in numeric_df.columns:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
            
            # Add correlation coefficient
            correlation = data[col].corr(data[target_var])
            ax.text(data[col].min() + (data[col].max() - data[col].min()) * 0.05,
                    data[target_var].max() - (data[target_var].max() - data[target_var].min()) * 0.05,
                    f"r = {correlation:.3f}",
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            st.pyplot(fig)
            plt.close(fig)
    
    # 5. Time Series Analysis
    st.markdown('<h3 class="sub-header">5. Time Series Analysis</h3>', unsafe_allow_html=True)
    if 'Week_Ending' in data.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=data, x='Week_Ending', y=target_var, ax=ax)
        ax.set_title(f"{target_var} Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{target_var}")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)
    
    # 6. Correlation Analysis
    st.markdown('<h3 class="sub-header">6. Correlation Analysis</h3>', unsafe_allow_html=True)
    st.write("Full Correlation Matrix:")
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=".5", ax=ax)
    ax.set_title("Correlation Matrix - All Variables")
    st.pyplot(fig)
    plt.close(fig)
    
    # 7. Outlier Analysis
    st.markdown('<h3 class="sub-header">7. Outlier Analysis</h3>', unsafe_allow_html=True)
    if target_var in numeric_df.columns:
        Q1 = data[target_var].quantile(0.25)
        Q3 = data[target_var].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[target_var] < lower_bound) | (data[target_var] > upper_bound)]
        normal_data = data[~((data[target_var] < lower_bound) | (data[target_var] > upper_bound))]
        st.write(f"Number of potential outliers in {target_var}: {len(outliers)}")
        
        if len(outliers) > 0:
            st.write("Outlier values:")
            st.write(outliers[['Week_Ending', target_var]])
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=normal_data, x='Week_Ending', y=target_var, label='Normal Values', color='darkgreen', ax=ax)
            sns.scatterplot(data=outliers, x='Week_Ending', y=target_var, color='red', label='Outliers', s=100, ax=ax)
            ax.axhline(upper_bound, color='red', linestyle='dashed', linewidth=1, label='Upper Bound')
            ax.axhline(lower_bound, color='red', linestyle='dashed', linewidth=1, label='Lower Bound')
            ax.set_title(f"Outlier Detection in {target_var}")
            ax.set_xlabel("Date")
            ax.set_ylabel(f"{target_var}")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.warning(f"Target variable '{target_var}' is not numeric. Cannot perform outlier analysis.")

# Feature engineering function
def perform_feature_engineering(data):
    """Perform feature engineering on the data"""
    st.markdown('<h2 class="section-header">Feature Engineering</h2>', unsafe_allow_html=True)
    
    df = data.copy()
    
    # Add fiscal year
    if 'Week_Ending' in df.columns:
        df['Fiscal_Year'] = assign_fiscal_year(df['Week_Ending'])
        st.success("Added Fiscal Year column")
    
    # Seasonal decomposition
    st.markdown('<h3 class="sub-header">Seasonal Decomposition</h3>', unsafe_allow_html=True)
    if 'Week_Ending' in df.columns and 'Sales' in df.columns:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            period = st.selectbox("Select seasonal period", [4, 13, 26, 52], index=3)
            
            if st.button("Perform Seasonal Decomposition"):
                temp_df = df.set_index('Week_Ending').sort_index()
                decomposition = seasonal_decompose(temp_df['Sales'], period=period, model='additive', extrapolate_trend='freq')
                
                df['SIndex'] = decomposition.seasonal.values
                st.success(f"Seasonal Index (SIndex) created with period {period}")
                
                # Show decomposition plot
                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                axes[0].plot(decomposition.observed)
                axes[0].set_ylabel("Observed")
                axes[0].set_title("Seasonal Decomposition")
                
                axes[1].plot(decomposition.trend)
                axes[1].set_ylabel("Trend")
                
                axes[2].plot(decomposition.seasonal)
                axes[2].set_ylabel("Seasonal")
                
                axes[3].plot(decomposition.resid)
                axes[3].set_ylabel("Residual")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        except ImportError:
            st.warning("statsmodels is not installed. Please install it to use seasonal decomposition.")
        except Exception as e:
            st.error(f"Error in seasonal decomposition: {str(e)}")
    
    # Custom dummy variables
    st.markdown('<h3 class="sub-header">Custom Dummy Variables</h3>', unsafe_allow_html=True)
    if 'Week_Ending' in df.columns:
        dummy_dates = st.text_input("Enter comma-separated dates for dummy variables (YYYY-MM-DD):")
        
        if dummy_dates and st.button("Create Dummy Variables"):
            try:
                date_strings = [d.strip() for d in dummy_dates.split(',') if d.strip()]
                date_objects = pd.to_datetime(date_strings, format='%Y-%m-%d', errors='coerce')
                valid_dates = date_objects[date_objects.notna()]
                
                if len(valid_dates) > 0:
                    for i, date_obj in enumerate(valid_dates, 1):
                        dummy_name = f"dummy{i}_{date_obj.strftime('%Y-%m-%d')}"
                        df[dummy_name] = (df['Week_Ending'] == date_obj).astype(int)
                        st.success(f"Created dummy variable: {dummy_name}")
                else:
                    st.warning("No valid dates entered.")
            except Exception as e:
                st.error(f"Error creating dummy variables: {str(e)}")
    
    # Split variables
    st.markdown('<h3 class="sub-header">Split Variables</h3>', unsafe_allow_html=True)
    if 'Week_Ending' in df.columns:
        split_vars = [col for col in df.columns if col not in ['Week_Ending', 'Sales', 'SIndex', 'Fiscal_Year'] and not col.startswith('dummy')]
        
        if split_vars:
            split_var = st.selectbox("Select variable to split", split_vars)
            split_date = st.date_input("Select split date", value=df['Week_Ending'].min())
            
            if st.button("Split Variable"):
                try:
                    split_dt = pd.to_datetime(split_date)
                    pre_mask = df['Week_Ending'] <= split_dt
                    
                    df[f"{split_var}_pre"] = df[split_var].where(pre_mask, 0)
                    df[f"{split_var}_post"] = df[split_var].where(~pre_mask, 0)
                    
                    # Drop the original variable
                    df.drop(columns=[split_var], inplace=True)
                    
                    st.success(f"Split {split_var} at {split_dt.date()}")
                except Exception as e:
                    st.error(f"Error splitting variable: {str(e)}")
    
    # Super campaigns
    st.markdown('<h3 class="sub-header">Super Campaigns</h3>', unsafe_allow_html=True)
    media_vars = [col for col in df.columns if any(kw in col.lower() for kw in ['impressions', 'clicks', 'social', 'search', 'video', 'tv', 'display', 'email']) and '_Spend' not in col]
    
    if media_vars:
        selected_vars = st.multiselect("Select variables to combine into super campaign", media_vars)
        campaign_name = st.text_input("Enter name for super campaign", value="Super_Campaign")
        
        if selected_vars and campaign_name and st.button("Create Super Campaign"):
            try:
                # Create super campaign volume
                df[f"{campaign_name}_Volume"] = df[selected_vars].sum(axis=1)
                
                # Drop the original variables
                df.drop(columns=selected_vars, inplace=True)
                
                st.success(f"Created super campaign: {campaign_name}")
            except Exception as e:
                st.error(f"Error creating super campaign: {str(e)}")
    
    return df

# Model training function
def train_model(data, target_var='Sales'):
    """Train marketing mix model"""
    st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)
    
    # Select features
    feature_options = [col for col in data.columns if col != target_var and col != 'Week_Ending']
    selected_features = st.multiselect("Select features for model", feature_options, default=feature_options)
    
    if not selected_features:
        st.warning("Please select at least one feature")
        return None, None, None
    
    # Select model type
    model_type = st.selectbox("Select model type", ["Ridge", "Lasso", "ElasticNet", "Linear Regression"])
    
    # Set up hyperparameters
    if model_type == "Ridge":
        alpha = st.slider("Alpha (regularization strength)", 0.01, 10.0, 1.0, 0.01)
    elif model_type == "Lasso":
        alpha = st.slider("Alpha (regularization strength)", 0.0001, 1.0, 0.1, 0.0001)
    elif model_type == "ElasticNet":
        alpha = st.slider("Alpha (regularization strength)", 0.0001, 1.0, 0.1, 0.0001)
        l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01)
    
    # Train/test split
    test_size = st.slider("Test set size (%)", 10, 40, 20)
    
    if st.button("Train Model"):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data
            X = data[selected_features].fillna(0)
            y = data[target_var].fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_type == "Ridge":
                model = Ridge(alpha=alpha)
            elif model_type == "Lasso":
                model = Lasso(alpha=alpha)
            elif model_type == "ElasticNet":
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            else:
                model = LinearRegression()
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Show results
            st.success("Model trained successfully!")
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{mse:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")
            col3.metric("R²", f"{r2:.4f}")
            
            # Feature importance
            if hasattr(model, 'coef_'):
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': model.coef_
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(importance_df)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance_df['Feature'], importance_df['Importance'])
                ax.set_title("Feature Importance")
                st.pyplot(fig)
                plt.close(fig)
            
            # Actual vs Predicted plot
            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted Values")
            st.pyplot(fig)
            plt.close(fig)
            
            return model, scaler, selected_features
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None, None
    
    return None, None, None

# Main app
def main():
    st.markdown('<h1 class="main-header">Marketing Mix Modeling</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    def main():
    st.markdown('<h1 class="main-header">Marketing Mix Modeling</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessed' not in st.session_state:
        st.session_state.preprocessed = False
    if 'engineered' not in st.session_state:
        st.session_state.engineered = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None # Corrected line
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'features' not in st.session_state:
        st.session_state.features = None

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_section = st.sidebar.radio(
        "Go to",
        ["Data Upload", "Data Preprocessing", "EDA", "Feature Engineering", "Model Training", "Results"]
    )
    
    # Data Upload Section
    if app_section == "Data Upload":
        st.markdown('<h2 class="section-header">Data Upload</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                data = load_and_preprocess_data(uploaded_file)
                st.session_state.data = data
                st.session_state.preprocessed = True
                
                st.success("Data loaded successfully!")
                st.write("Data Preview:")
                st.dataframe(data.head())
                
                st.write("Data Summary:")
                st.write(f"Shape: {data.shape}")
                st.write(f"Columns: {list(data.columns)}")
                if 'Week_Ending' in data.columns:
                    st.write(f"Date Range: {data['Week_Ending'].min()} to {data['Week_Ending'].max()}")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    # Data Preprocessing Section
    elif app_section == "Data Preprocessing" and st.session_state.preprocessed:
        st.markdown('<h2 class="section-header">Data Preprocessing</h2>', unsafe_allow_html=True)
        
        data = st.session_state.data
        
        # Show missing values
        st.subheader("Missing Values")
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            st.write(missing_values[missing_values > 0])
            
            # Option to fill missing values
            if st.button("Fill Missing Values with 0"):
                data = data.fillna(0)
                st.session_state.data = data
                st.success("Missing values filled with 0")
                st.experimental_rerun()
        else:
            st.success("No missing values found!")
        
        # Show data types
        st.subheader("Data Types")
        st.write(data.dtypes)
        
        # Option to convert data types
        st.subheader("Convert Data Types")
        numeric_cols = data.select_dtypes(include=['object']).columns.tolist()
        if numeric_cols:
            col_to_convert = st.selectbox("Select column to convert to numeric", numeric_cols)
            if st.button("Convert to Numeric"):
                data[col_to_convert] = pd.to_numeric(data[col_to_convert], errors='coerce')
                st.session_state.data = data
                st.success(f"Converted {col_to_convert} to numeric")
                st.experimental_rerun()
        else:
            st.info("No object columns to convert")
    
    # EDA Section
    elif app_section == "EDA" and st.session_state.preprocessed:
        perform_comprehensive_eda(st.session_state.data)
    
    # Feature Engineering Section
    elif app_section == "Feature Engineering" and st.session_state.preprocessed:
        engineered_data = perform_feature_engineering(st.session_state.data)
        st.session_state.data = engineered_data
        st.session_state.engineered = True
        
        st.subheader("Engineered Data Preview")
        st.dataframe(engineered_data.head())
    
    # Model Training Section
    elif app_section == "Model Training" and st.session_state.engineered:
        model, scaler, features = train_model(st.session_state.data)
        
        if model is not None:
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.features = features
            st.session_state.model_trained = True
    
    # Results Section
    elif app_section == "Results" and st.session_state.model_trained:
        st.markdown('<h2 class="section-header">Model Results</h2>', unsafe_allow_html=True)
        
        data = st.session_state.data
        
        # Prepare data for plotting
        X = data[st.session_state.features].fillna(0)
        y = data['Sales'].fillna(0)
        X_scaled = st.session_state.scaler.transform(X)
        y_pred = st.session_state.model.predict(X_scaled)
        
        # Residual plot
        st.subheader("Residual Plot")
        residuals = y - y_pred
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residual Plot")
        st.pyplot(fig)
        plt.close(fig)
        
        # Time series of actual vs predicted
        if 'Week_Ending' in data.columns:
            st.subheader("Time Series: Actual vs Predicted")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data['Week_Ending'], y, label='Actual')
            ax.plot(data['Week_Ending'], y_pred, label='Predicted', alpha=0.7)
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.set_title("Actual vs Predicted Sales Over Time")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)
        
        # Download results
        st.subheader("Download Results")
        results_df = pd.DataFrame({
            'Date': data['Week_Ending'] if 'Week_Ending' in data.columns else range(len(y)),
            'Actual': y,
            'Predicted': y_pred,
            'Residuals': residuals
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="mmm_results.csv",
            mime="text/csv"
        )
    
    # Handle cases where preprocessed data is not available
    elif not st.session_state.preprocessed and app_section != "Data Upload":
        st.warning("Please upload and preprocess data first")
    
    # Handle cases where model is not trained
    elif app_section == "Results" and not st.session_state.model_trained:
        st.warning("Please train a model first")

if __name__ == "__main__":
    main()