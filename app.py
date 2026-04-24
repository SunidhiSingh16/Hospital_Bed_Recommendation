import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_hospital_data, prepare_ml_features, generate_synthetic_data
from src.poisson_model import PoissonPredictor
from src.ml_model import HospitalAdmissionPredictor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Hospital Bed Demand Forecasting", layout="wide")

# Add custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
        border: none;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Hospital Bed Demand Forecasting")
st.write("""
This application predicts hospital bed demand using a combination of statistical methods (Poisson Distribution) 
and Machine Learning. It helps hospitals better prepare for patient admissions by analyzing patterns and predicting future demand.
""")

@st.cache_data(show_spinner=False)
def load_default_data():
    return load_hospital_data(scenario="normal")


# Initialize session state for deployment-safe cold starts
if 'df' not in st.session_state:
    st.session_state.df = None

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Try deterministic bootstrap so the app does not depend on UI interaction order
if st.session_state.df is None:
    try:
        st.session_state.df = load_default_data()
        st.session_state.data_loaded = True
    except Exception:
        st.session_state.data_loaded = False

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Configuration", "📈 Analysis", "🤖 Predictions", "📋 Report"])

with tab1:
    st.header("Settings & Parameters")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Data Parameters")
        with st.expander("Customize Data Generation", expanded=True):
            # Add start date selection
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().replace(month=1, day=1),
                help="Select the start date for data generation"
            )
            
            base_admissions = st.slider(
                "Base Daily Admissions",
                min_value=10,
                max_value=200,
                value=50,
                help="Average number of daily admissions"
            )
            
            weekday_impact = st.slider(
                "Weekday Effect",
                min_value=0,
                max_value=50,
                value=10,
                help="How much weekdays increase admissions"
            )
            
            seasonal_impact = st.slider(
                "Seasonal Effect",
                min_value=0,
                max_value=50,
                value=15,
                help="Impact of seasonal variations"
            )
            
            special_events_count = st.slider(
                "Number of Special Events",
                min_value=0,
                max_value=20,
                value=5,
                help="Number of unusual spikes in admissions"
            )
            
            special_events_magnitude = st.slider(
                "Special Event Impact",
                min_value=10,
                max_value=100,
                value=30,
                help="How much special events increase admissions"
            )
    
    with col2:
        st.subheader("Scenario & Model Selection")
        # Predefined Scenarios
        scenario = st.selectbox(
            "Select Scenario",
            ["custom", "normal", "high_variation", "pandemic", "summer_surge"],
            help="Choose between custom parameters or predefined scenarios"
        )

        scenario_descriptions = {
            "custom": "Using custom parameters defined above",
            "normal": "Regular hospital operations with typical seasonal and weekly patterns",
            "high_variation": "More volatile admissions with larger differences between peak and low periods",
            "pandemic": "Simulates a pandemic scenario with higher baseline admissions and more frequent spikes",
            "summer_surge": "Unusual summer increase in admissions, contrary to typical seasonal patterns"
        }

        st.info(scenario_descriptions[scenario])

        # Model Selection
        model_type = st.selectbox(
            "Select Model Type",
            ["Poisson Only", "Poisson + ML (GBM)", "Poisson + ML (XGBoost)"],
            help="Choose between statistical model only or enhanced with machine learning"
        )

        # Load Data Button
        st.write("")  # Add some spacing
        st.write("")  # Add some spacing
        if st.button("🔄 Load Data", key="load_data"):
            with st.spinner('Loading and processing data...'):
                try:
                    if scenario == "custom":
                        custom_params = {
                            "start_date": start_date.strftime("%Y-%m-%d"),
                            "base_admissions": base_admissions,
                            "weekday_impact": weekday_impact,
                            "seasonal_impact": seasonal_impact,
                            "special_events_count": special_events_count,
                            "special_events_magnitude": special_events_magnitude
                        }
                        df = load_hospital_data(scenario="custom", custom_params=custom_params)
                    else:
                        df = load_hospital_data(scenario=scenario)
                    
                    # Store the data in session state
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    
                    # Show success message
                    st.success("✅ Data loaded successfully!")
                    time.sleep(1)  # Give time to see the success message
                    st.rerun()  # Updated to use st.rerun() instead of st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    st.stop()

# Only show other tabs if data is loaded
if not st.session_state.data_loaded or st.session_state.df is None:
    st.warning("👆 Please load the data first using the button in the Configuration tab.")
    st.stop()

# Get data from session state
df = st.session_state.df

with tab2:
    st.header("Data Analysis")
    
    # Interactive Time Series Plot
    st.subheader("Interactive Admission Trends")
    fig = px.line(df, x='date', y='daily_admissions', 
                  title='Hospital Admissions Over Time')
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekly Pattern Analysis
        weekly_avg = df.groupby('day_of_week')['daily_admissions'].mean()
        fig_weekly = px.bar(weekly_avg, 
                           title='Average Admissions by Day of Week',
                           labels={'day_of_week': 'Day of Week (0=Monday)', 
                                  'value': 'Average Admissions'})
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with col2:
        # Monthly Pattern Analysis
        monthly_avg = df.groupby('month')['daily_admissions'].mean()
        fig_monthly = px.bar(monthly_avg, 
                            title='Average Admissions by Month',
                            labels={'month': 'Month', 
                                   'value': 'Average Admissions'})
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Statistics Cards
    st.subheader("Key Metrics")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Daily Admissions", 
                 f"{df['daily_admissions'].mean():.1f}",
                 f"{df['daily_admissions'].std():.1f} σ")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Weekday vs Weekend Difference",
                 f"{(df[df['is_weekend'] == 0]['daily_admissions'].mean() - df[df['is_weekend'] == 1]['daily_admissions'].mean()):.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Seasonal Variation",
                 f"{(df[df['month'].isin([12,1,2])]['daily_admissions'].mean() - df[df['month'].isin([6,7,8])]['daily_admissions'].mean()):.1f}")
        st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.header("Predictions & Model Performance")
    
    # Poisson Distribution Analysis
    poisson_model = PoissonPredictor()
    recent_data = df.tail(7)['daily_admissions'].values
    poisson_model.fit(recent_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Poisson Distribution")
        fig = poisson_model.plot_distribution()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Prediction Range")
        lower, upper = poisson_model.get_most_likely_range(confidence=0.80)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.write("80% Confidence Interval:")
        st.markdown(f'<p class="big-font">{lower:.0f} - {upper:.0f} admissions</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Machine Learning Analysis
    if model_type != "Poisson Only":
        st.header("Machine Learning Insights")
        
        X, y = prepare_ml_features(df)
        train_size = int(0.8 * len(df))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        ml_model = HospitalAdmissionPredictor(
            model_type='xgboost' if 'XGBoost' in model_type else 'gbm'
        )
        ml_model.fit(X_train, y_train)
        metrics = ml_model.evaluate(X_test, y_test)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Model Performance")
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
            st.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.subheader("Feature Importance")
            importance = ml_model.get_feature_importance(X.columns)
            importance_df = pd.DataFrame(
                {'Feature': list(importance.keys()), 
                 'Importance': list(importance.values())}
            ).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h',
                        title='Feature Importance Analysis')
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction Comparison
        st.subheader("Model Predictions Comparison")
        latest_features = X.iloc[-1:].copy()
        ml_prediction = ml_model.predict(latest_features)[0]
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("ML Model Prediction", f"{ml_prediction:.1f}")
            st.metric("Poisson Prediction", f"{poisson_model.lambda_:.1f}",
                     f"{ml_prediction - poisson_model.lambda_:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col6:
            # Prediction confidence visualization
            fig = go.Figure()
            fig.add_shape(type="rect",
                x0=lower, x1=upper,
                y0=0, y1=1,
                fillcolor="rgba(0,176,246,0.2)",
                line=dict(width=0),
                layer="below")
            fig.add_vline(x=ml_prediction, line_dash="dash", 
                         line_color="red", annotation_text="ML Prediction")
            fig.add_vline(x=poisson_model.lambda_, line_dash="dash",
                         line_color="green", annotation_text="Poisson Prediction")
            fig.update_layout(title="Prediction Comparison with Confidence Interval",
                            xaxis_title="Number of Admissions",
                            showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Summary Report")
    
    # Generate comprehensive report
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    report_md = f"""
    ### Hospital Bed Demand Forecast Report
    Generated on: {report_time}
    
    #### Configuration
    - Scenario: {scenario}
    - Model Type: {model_type}
    
    #### Key Statistics
    - Average Daily Admissions: {df['daily_admissions'].mean():.1f}
    - Standard Deviation: {df['daily_admissions'].std():.1f}
    - Weekday/Weekend Difference: {(df[df['is_weekend'] == 0]['daily_admissions'].mean() - df[df['is_weekend'] == 1]['daily_admissions'].mean()):.1f}
    - Seasonal Variation: {(df[df['month'].isin([12,1,2])]['daily_admissions'].mean() - df[df['month'].isin([6,7,8])]['daily_admissions'].mean()):.1f}
    
    #### Predictions
    - Poisson Expected Value: {poisson_model.lambda_:.1f}
    - 80% Confidence Range: {lower:.0f} - {upper:.0f}
    """
    
    if model_type != "Poisson Only":
        report_md += f"""
    #### Machine Learning Insights
    - ML Prediction: {ml_prediction:.1f}
    - Mean Absolute Error: {metrics['mae']:.2f}
    - Root Mean Squared Error: {metrics['rmse']:.2f}
    
    #### Top Predictive Features
    """
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            report_md += f"- {feat}: {imp:.3f}\n"
    
    st.markdown(report_md)
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Report (Text)",
            report_md,
            file_name=f"hospital_forecast_report_{scenario}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )
    
    with col2:
        # Create CSV with predictions
        predictions_df = pd.DataFrame({
            'date': [datetime.now() + timedelta(days=1)],
            'poisson_prediction': [poisson_model.lambda_],
            'confidence_lower': [lower],
            'confidence_upper': [upper]
        })
        if model_type != "Poisson Only":
            predictions_df['ml_prediction'] = ml_prediction
        
        st.download_button(
            "Download Predictions (CSV)",
            predictions_df.to_csv(index=False),
            file_name=f"predictions_{scenario}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        ) 