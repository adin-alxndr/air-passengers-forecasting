import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import warnings
import matplotlib.pyplot as plt
import io
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Air Passengers Forecasting",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 2rem;
        font-weight: 700;
        color: #A23B72;
        padding: 12px 20px;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-left: 6px solid #2E86AB;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div.stButton > button {
        border-radius: 8px;
        height: 45px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #2E86AB;
        color: white;
    }
    .stat-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #2E86AB;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">✈️ Air Passengers Forecasting Dashboard</p>', unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
            Airline Passenger Forecasting using SARIMA and Prophet Models
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('air_passengers_data.csv', index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        # If file not found, create sample data
        url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
        df = pd.read_csv(url)
        df.columns = ['Month', 'Passengers']
        df['Month'] = pd.to_datetime(df['Month'])
        df = df.set_index('Month')
        return df

# Load models function
@st.cache_resource
def load_models():
    try:
        sarima_model = SARIMAXResults.load('sarima_model.pkl')
        with open('prophet_model.pkl', 'rb') as f:
            prophet_model = pickle.load(f)
        with open('model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        return sarima_model, prophet_model, metrics
    except FileNotFoundError:
        return None, None, None

# Perform ADF test
def adf_test(series, title=''):
    result = adfuller(series.dropna())
    
    output = {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }
    
    return output

# Load data
df = load_data()
sarima_model, prophet_model, metrics = load_models()

# Sidebar Navigator Panel
st.sidebar.markdown("# ⚙️ Navigation Panel")

# Navigation buttons
if "page" not in st.session_state:
    st.session_state.page = "overview"

if st.sidebar.button("📊 Overview", width='stretch'):
    st.session_state.page = "overview"

if st.sidebar.button("📈 Time Series Analysis", width='stretch'):
    st.session_state.page = "analysis"

if st.sidebar.button("🔮 Forecasting", width='stretch'):
    st.session_state.page = "forecast"

if st.sidebar.button("📉 Model Comparison", width='stretch'):
    st.session_state.page = "comparison"

page = st.session_state.page

st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 About This Dashboard")

st.sidebar.info("""
**Dataset**  
Air Passengers (1949–1960)

**Forecasting Models**
- SARIMA (1,1,1)(1,1,1,12)
- Prophet (Meta/Facebook)

**Evaluation Metrics**
- MAE — Mean Absolute Error  
- RMSE — Root Mean Squared Error  
- MAPE — Mean Absolute Percentage Error  
- R² — Coefficient of Determination
""")

# Page 1: Overview
if page == "overview":
    st.markdown('<p class="sub-header">📊 Dataset Overview</p>', unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Observations",
            value=f"{len(df)}",
            delta="Months"
        )
    
    with col2:
        st.metric(
            label="Date Range",
            value=f"{df.index.year.min()}-{df.index.year.max()}",
            delta="12 Years"
        )
    
    with col3:
        st.metric(
            label="Average Passengers",
            value=f"{df['Passengers'].mean():.0f}K",
            delta=f"+{df['Passengers'].pct_change().mean()*100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="Max Passengers",
            value=f"{df['Passengers'].max():.0f}K",
            delta=df['Passengers'].idxmax().strftime('%b %Y')
        )
    
    # Interactive time series plot
    st.markdown("### 📈 Passenger Trend Over Time")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Passengers'],
        mode='lines+markers',
        name='Passengers',
        line=dict(color='#2E86AB', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Date</b>: %{x|%Y-%m}<br><b>Passengers</b>: %{y:.0f}K<extra></extra>'
    ))
    
    fig.update_layout(
        title='Air Passengers Time Series (1949-1960)',
        xaxis_title='Year',
        yaxis_title='Number of Passengers (thousands)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Insights
    st.markdown("### 💡 Key Insights")
    
    st.info("""
        **📈 Upward Trend**
        
        Passenger numbers show consistent growth from 1949 to 1960, 
        indicating increasing air travel demand with approximately 
        **{:.1f}% average monthly growth**.
    """.format(df['Passengers'].pct_change().mean()*100))

    st.success("""
        **🔄 Seasonal Pattern**
        
        Clear yearly seasonality with peaks during summer months 
        (vacation season). Seasonality period of **12 months** detected.
    """)

    growth_rate = ((df['Passengers'].iloc[-1] / df['Passengers'].iloc[0]) - 1) * 100
    st.warning("""
        **📊 Overall Growth**
        
        From {} to {}, passenger count increased by **{:.0f}%**, 
        from {}K to {}K passengers.
    """.format(
        df.index[0].year,
        df.index[-1].year,
        growth_rate,
        df['Passengers'].iloc[0],
        df['Passengers'].iloc[-1]
        ))

# Page 2: Time Series Analysis
elif page == "analysis":
    st.markdown('<p class="sub-header">📈 Time Series Decomposition</p>', unsafe_allow_html=True)
    
    # Short explanation
    st.markdown("""
        Time series decomposition helps identify the **trend**, **seasonality**, and **random variations** 
        in airline passenger data over time.
    """)

    # How to read the chart
    st.info("""
        **How to read this chart**

        • **Observed** – the original passenger data  
        • **Trend** – the long-term growth pattern  
        • **Seasonal** – repeating yearly patterns  
        • **Residual** – random fluctuations not explained by the model
    """)
    
    # Perform decomposition
    decomposition = seasonal_decompose(df['Passengers'], model='multiplicative', period=12)
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.15
    )
    
    # Observed
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Passengers'], name='Observed', 
                   line=dict(color='#2E86AB')),
        row=1, col=1
    )
    
    # Trend
    fig.add_trace(
        go.Scatter(x=df.index, y=decomposition.trend, name='Trend',
                   line=dict(color='#C73E1D')),
        row=2, col=1
    )
    
    # Seasonal
    fig.add_trace(
        go.Scatter(x=df.index, y=decomposition.seasonal, name='Seasonal',
                   line=dict(color='#F18F01')),
        row=3, col=1
    )
    
    # Residual
    fig.add_trace(
        go.Scatter(x=df.index, y=decomposition.resid, name='Residual',
                   line=dict(color='#6A994E')),
        row=4, col=1
    )
    
    fig.update_layout(
        height=900,
        showlegend=False,
        template='plotly_white',
        title_text="Time Series Decomposition"
    )
    
    fig.update_xaxes(title_text="Month", row=4, col=1)
    
    st.plotly_chart(fig, width='stretch')
    
    # Analysis insights
    st.markdown("### 🔍 Decomposition Insights")
    
    st.info("""
        **Trend Component**

        - Shows consistent and nearly linear growth over time  
        - No clear signs of decline or saturation  
        - Indicates stable expansion in the airline industry
    """)

    st.success("""
        **Seasonal Component**

        - Displays a clear and stable seasonal pattern  
        - Follows a 12-month (annual) cycle  
        - Passenger numbers typically peak during mid-year months  
        - Lower demand occurs during the early and late months of the year
    """)

# Page 3: Forecasting
elif page == "forecast":
    st.markdown('<p class="sub-header">🔮 Future Forecasting</p>', unsafe_allow_html=True)
    
    if sarima_model is None or prophet_model is None:
        st.warning("⚠️ Models not found. Please run the Jupyter notebook first to train the models.")
    else:
        # Forecast controls
        st.markdown("### ⚙️ Forecast Settings")
        
        forecast_periods = st.slider(
            "Forecast Horizon (months)",
            min_value=3,
            max_value=24,
            value=12,
            step=3
        )

        model_choice = st.selectbox(
            "Select Model",
            ["Both", "SARIMA", "Prophet"]
        )

        st.caption("Select a model and forecast horizon to generate future passenger predictions.")
                
        # Generate forecasts
        if st.button("🚀 Generate Forecast", type="primary", width='stretch'):
            with st.spinner("Generating forecasts..."):
                # SARIMA forecast
                sarima_forecast = sarima_model.forecast(steps=forecast_periods)
                
                # Prophet forecast
                future = prophet_model.make_future_dataframe(periods=forecast_periods, freq='MS')
                prophet_forecast_df = prophet_model.predict(future)
                prophet_pred = prophet_forecast_df.iloc[-forecast_periods:]['yhat'].values
                
                # Create future dates
                last_date = df.index[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.offsets.MonthBegin(1),
                    periods=forecast_periods,
                    freq='MS'
                )
                
                # Plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Passengers'],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='#2E86AB', width=2),
                    marker=dict(size=4)
                ))
                
                # SARIMA forecast
                if model_choice in ["Both", "SARIMA"]:
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=sarima_forecast,
                        mode='lines+markers',
                        name='SARIMA Forecast',
                        line=dict(color='#C73E1D', width=2, dash='dash'),
                        marker=dict(size=6, symbol='diamond')
                    ))
                
                # Prophet forecast
                if model_choice in ["Both", "Prophet"]:
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=prophet_pred,
                        mode='lines+markers',
                        name='Prophet Forecast',
                        line=dict(color='#F18F01', width=2, dash='dot'),
                        marker=dict(size=6, symbol='square')
                    ))
                    
                    # Add confidence interval for Prophet
                    prophet_lower = prophet_forecast_df.iloc[-forecast_periods:]['yhat_lower'].values
                    prophet_upper = prophet_forecast_df.iloc[-forecast_periods:]['yhat_upper'].values
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates.tolist() + future_dates.tolist()[::-1],
                        y=prophet_upper.tolist() + prophet_lower.tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(241, 143, 1, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Prophet 95% CI',
                        showlegend=True
                    ))
                
                # Add vertical line at forecast start
                fig.add_vline(
                    x=last_date.timestamp() * 1000,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Forecast Start",
                    annotation_position="top"
                )
                
                fig.update_layout(
                    title=f'{forecast_periods}-Month Future Forecast',
                    xaxis_title='Date',
                    yaxis_title='Passengers (thousands)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=600
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Forecast table
                st.markdown("### 📋 Forecast Values")
                
                forecast_table = pd.DataFrame({
                    'Month': future_dates.strftime('%Y-%m'),
                    'SARIMA Forecast': sarima_forecast.values.round(2),
                    'Prophet Forecast': prophet_pred.round(2),
                    'Difference': (prophet_pred - sarima_forecast.values).round(2)
                })
                
                st.dataframe(forecast_table, hide_index=True, width='stretch')
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Average SARIMA Forecast",
                        f"{sarima_forecast.mean():.2f}K"
                    )
                
                with col2:
                    st.metric(
                        "Average Prophet Forecast",
                        f"{prophet_pred.mean():.2f}K"
                    )
                
                with col3:
                    avg_diff = abs(prophet_pred - sarima_forecast.values).mean()
                    st.metric(
                        "Average Difference",
                        f"{avg_diff:.2f}K"
                    )
                
                # Download button
                csv = forecast_table.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv,
                    file_name=f"forecast_{forecast_periods}months.csv",
                    mime="text/csv",
                    width='stretch'
                )

# Page 4: Model Comparison
elif page == "comparison":
    st.markdown('<p class="sub-header">📉 Model Performance Comparison</p>', unsafe_allow_html=True)
    
    if metrics is None:
        st.warning("⚠️ Metrics not found. Using default values from notebook.")
        
        # Use values from your notebook output
        metrics = {
            'sarima': {
                'mae': 25.27,
                'rmse': 31.79,
                'mape': 5.43,
                'r2': 0.8345
            },
            'prophet': {
                'mae': 18.32,
                'rmse': 22.59,
                'mape': 4.06,
                'r2': 0.9164
            },
            'best_model': 'Prophet'
        }
    
    # Display metrics
    st.markdown("### 🎯 Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 SARIMA Model")
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            st.metric("MAE", f"{metrics['sarima']['mae']:.2f}")
            st.metric("MAPE", f"{metrics['sarima']['mape']:.2f}%")
        
        with subcol2:
            st.metric("RMSE", f"{metrics['sarima']['rmse']:.2f}")
            st.metric("R² Score", f"{metrics['sarima']['r2']:.4f}")
        
        # Performance indicator
        if metrics['sarima']['r2'] > 0.8:
            st.success("✅ Excellent Performance")
        elif metrics['sarima']['r2'] > 0.6:
            st.info("✓ Good Performance")
        else:
            st.warning("⚠ Moderate Performance")
    
    with col2:
        st.markdown("#### 🔮 Prophet Model")
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            st.metric("MAE", f"{metrics['prophet']['mae']:.2f}")
            st.metric("MAPE", f"{metrics['prophet']['mape']:.2f}%")
        
        with subcol2:
            st.metric("RMSE", f"{metrics['prophet']['rmse']:.2f}")
            st.metric("R² Score", f"{metrics['prophet']['r2']:.4f}")
        
        # Performance indicator
        if metrics['prophet']['r2'] > 0.8:
            st.success("✅ Excellent Performance")
        elif metrics['prophet']['r2'] > 0.6:
            st.info("✓ Good Performance")
        else:
            st.warning("⚠ Moderate Performance")
    
    # Best model announcement
    st.markdown("---")
    st.success(f"🏆 **Best Model**: {metrics['best_model']} (based on lowest RMSE)")

    comparison_df = pd.DataFrame({
        'Model': ['SARIMA', 'Prophet'],
        'MAE': [metrics['sarima']['mae'], metrics['prophet']['mae']],
        'RMSE': [metrics['sarima']['rmse'], metrics['prophet']['rmse']],
        'MAPE (%)': [metrics['sarima']['mape'], metrics['prophet']['mape']],
        'R² Score': [metrics['sarima']['r2'], metrics['prophet']['r2']]
    })
    
    # Comparison chart
    st.markdown("### 📈 Visual Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="SARIMA",
        x=["MAE","RMSE","MAPE","R²"],
        y=[
            metrics['sarima']['mae'],
            metrics['sarima']['rmse'],
            metrics['sarima']['mape'],
            metrics['sarima']['r2']
        ],
        marker_color="#47CACF"   # pastel blue
    ))

    fig.add_trace(go.Bar(
        name="Prophet",
        x=["MAE","RMSE","MAPE","R²"],
        y=[
            metrics['prophet']['mae'],
            metrics['prophet']['rmse'],
            metrics['prophet']['mape'],
            metrics['prophet']['r2']
        ],
        marker_color="#FFAF4E"   # pastel orange
    ))

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        height=450,
        legend_title="Model",
        xaxis_title="Metric",
        yaxis_title="Score"
    )

    st.plotly_chart(fig, width='stretch')
    
    st.success("🏆 Prophet shows lower error (MAE & RMSE) and higher R², indicating better forecasting accuracy.")
    
    # Model explanation
    st.markdown("### 📚 Model Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **SARIMA**

        • Interpretable statistical model  
        • Works well with stable seasonality  
        • Suitable for short-term forecasting
        """)
    
    with col2:
        st.info("""
        **Prophet**

        • Automatic trend and seasonality detection  
        • Robust to missing data and outliers  
        • Easy to tune and implement
        """)
        
    # Performance interpretation
    st.markdown("### 🔍 Performance Interpretation")
    
    st.info(f"""
        **Overall Analysis:**
        
        - **Prophet** outperforms SARIMA across all metrics
        - Prophet's RMSE ({metrics['prophet']['rmse']:.2f}) is **{((metrics['sarima']['rmse']/metrics['prophet']['rmse'] - 1) * 100):.1f}% lower** than SARIMA
        - Prophet's R² ({metrics['prophet']['r2']:.4f}) indicates it explains **{metrics['prophet']['r2']*100:.2f}%** of the variance
        - Both models show good performance (R² > 0.8), suitable for forecasting
    """)
    st.info(f"""
        **Recommendation:**
        - Use **Prophet** for production forecasting due to better accuracy
        - Use **SARIMA** for interpretability and understanding seasonality patterns
        - Consider ensemble methods for critical decisions
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p style='margin: 5px 0;'><strong>Air Passengers Forecasting Dashboard</strong></p>
        <p style='margin: 5px 0;'>Built with Streamlit | Time Series Analysis Project | 2026</p>
        <p style='margin: 5px 0; font-size: 0.9em;'>
            Models: SARIMA(1,1,1)(1,1,1,12) & Prophet | 
            Dataset: 1949-1960 Monthly Air Passengers
        </p>
    </div>
""", unsafe_allow_html=True)