"""
Portfolio Risk Cluster Analysis Application

A Streamlit web application for analyzing portfolio risk using clustering techniques
and visualizing asset relationships.

Author: Mahlatse Ndala
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.cluster import KMeans
from streamlit_chat import message
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Application configuration
APP_CONFIG = {
    "page_title": "Risk Cluster Analyzer",
    "layout": "wide",
    "default_start_date": datetime.now() - timedelta(days=365),
    "default_end_date": datetime.now(),
    "min_data_points": 5,
    "rolling_window": 30,
    "annualization_factor": 252,
    "default_portfolios": 1000,
    "kmeans_init": 10
}

# Constants for consistent labeling
LABELS = {
    "num_clusters": "Number of Clusters",
    "expected_risk": "Expected Risk (%)",
    "volatility": "Volatility (%)",
    "asset_list": "Asset List",
    "correlation_coef": "Correlation Coefficient"
}

st.set_page_config(
    page_title=APP_CONFIG["page_title"], 
    layout=APP_CONFIG["layout"],
    page_icon="📊"
)

# Financial terms dictionary for the assistant
FINANCIAL_TERMS: Dict[str, str] = {
    "sharpe ratio": "The Sharpe ratio measures the risk-adjusted return of an investment. Higher is better.",
    "efficient frontier": "A set of optimal portfolios that offer the highest expected return for a defined level of risk.",
    "correlation matrix": "Shows how different assets move in relation to each other. Range from -1 to +1.",
    "max drawdown": "The maximum observed loss from a peak to a trough of a portfolio.",
    "elbow curve": "Method to determine optimal number of clusters by looking at the rate of improvement.",
    "kmeans clustering": "Algorithm that groups similar assets together based on return patterns.",
    "portfolio volatility": "Measures how much portfolio returns fluctuate over time.",
    "risk contribution": "How much each asset or cluster contributes to total portfolio risk.",
    "beta": "Measures an asset's sensitivity to market movements. Beta > 1 means more volatile than market.",
    "alpha": "The excess return of an investment relative to the return of a benchmark index.",
    "var": "Value at Risk - estimates the potential loss in value of a portfolio over a defined period.",
    "standard deviation": "A measure of the amount of variation or dispersion of returns.",
    "covariance": "Measures how two assets move together. Positive means they tend to move in the same direction."
}

def get_bot_response(user_query: str) -> str:
    """
    Get bot response for financial term queries.
    
    Args:
        user_query: User's query string
        
    Returns:
        Explanation of the financial term or default message
    """
    if not user_query or not isinstance(user_query, str):
        return "Please ask a question about financial terms."
    
    query = user_query.lower().strip()
    
    # Check for exact matches first
    for term, explanation in FINANCIAL_TERMS.items():
        if term in query:
            return f"**{term.title()}**: {explanation}"
    
    # Check for partial matches
    for term, explanation in FINANCIAL_TERMS.items():
        if any(word in query for word in term.split()):
            return f"**{term.title()}**: {explanation}"
    
    available_terms = ", ".join(FINANCIAL_TERMS.keys())
    return (f"I'm not sure about that. Try asking about specific terms like: {available_terms}. "
            f"You can also try variations of these terms.")


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(assets: List[str], start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """
    Fetch stock data from Yahoo Finance with comprehensive error handling.
    
    Args:
        assets: List of stock symbols
        start_date: Start date for data fetching
        end_date: End date for data fetching
        
    Returns:
        DataFrame with stock data or None if error occurs
    """
    if not assets:
        st.error("No assets provided for data fetching.")
        return None
    
    if not isinstance(assets, list):
        st.error("Assets must be provided as a list.")
        return None
    
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return None
    
    try:
        # Clean asset symbols
        cleaned_assets = [asset.strip().upper() for asset in assets if asset.strip()]
        
        if not cleaned_assets:
            st.error("No valid asset symbols found after cleaning.")
            return None
        
        # Download data with progress indicator
        with st.spinner(f'Downloading data for {len(cleaned_assets)} assets...'):
            try:
                data = yf.download(
                    cleaned_assets, 
                    start=start_date, 
                    end=end_date,
                    progress=False
                )
            except TypeError as te:
                # Handle yfinance API changes or version incompatibilities
                st.warning("⚠️ Trying alternative download method...")
                try:
                    # Fallback method without progress parameter
                    data = yf.download(cleaned_assets, start=start_date, end=end_date)
                except Exception:
                    # Last resort - download assets one by one
                    st.info("📥 Downloading assets individually...")
                    data_list = []
                    for asset in cleaned_assets:
                        try:
                            asset_data = yf.download(asset, start=start_date, end=end_date)
                            if not asset_data.empty:
                                data_list.append(asset_data)
                        except Exception:
                            st.warning(f"⚠️ Could not download data for {asset}")
                            continue
                    
                    if data_list:
                        # Combine individual downloads
                        data = pd.concat([d['Close'] if 'Close' in d.columns else d for d in data_list], axis=1)
                        data.columns = cleaned_assets[:len(data_list)]
                    else:
                        data = pd.DataFrame()
            except Exception as e:
                raise e
        
        if data.empty:
            st.error("No data downloaded. Please check asset symbols and date range.")
            return None
        
        # Log successful download
        st.success(f"Successfully downloaded data for {len(cleaned_assets)} assets")
        return data
        
    except Exception as e:
        error_msg = f"Error downloading data: {str(e)}"
        st.error(error_msg)
        st.info("Try reducing the number of assets or adjusting the date range.")
        return None

def calculate_efficient_frontier(returns: pd.DataFrame, num_portfolios: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate efficient frontier for portfolio optimization.
    
    Args:
        returns: DataFrame with asset returns
        num_portfolios: Number of random portfolios to generate
        
    Returns:
        Tuple of (portfolio_risks, portfolio_returns)
    """
    if returns.empty:
        raise ValueError("Returns DataFrame is empty")
    
    num_assets = len(returns.columns)
    if num_assets < 2:
        raise ValueError("At least 2 assets are required for portfolio optimization")
    
    returns_mean = returns.mean()
    cov_matrix = returns.cov()
    
    # Validate covariance matrix
    if np.any(np.isnan(cov_matrix.values)) or np.any(np.isinf(cov_matrix.values)):
        raise ValueError("Invalid covariance matrix - contains NaN or infinite values")
    
    # Arrays to store returns and risks
    port_returns = np.zeros(num_portfolios)
    port_risks = np.zeros(num_portfolios)
    
    # Create random number generator for reproducibility
    rng = np.random.default_rng(42)
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = rng.random(num_assets)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        port_returns[i] = np.sum(returns_mean * weights) * APP_CONFIG["annualization_factor"]
        port_risks[i] = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix * APP_CONFIG["annualization_factor"], weights))
        )
    
    return port_risks, port_returns


def find_optimal_k(wcss: List[float]) -> int:
    """
    Find optimal number of clusters using elbow method.
    
    Args:
        wcss: List of within-cluster sum of squares values
        
    Returns:
        Optimal number of clusters
    """
    if len(wcss) < 3:
        return 2  # Default minimum
    
    # Calculate the rate of change in wcss
    diff = np.diff(wcss)
    
    if len(diff) < 2:
        return 2
    
    diff_r = diff[1:] / diff[:-1]
    
    # Find the point where the ratio of differences is closest to 1
    elbow_idx = np.argmin(np.abs(diff_r - 1)) + 2
    
    # Ensure the result is within valid range
    return max(2, min(elbow_idx, len(wcss)))


def validate_data_quality(data: pd.DataFrame, min_periods: int = 30) -> bool:
    """
    Validate data quality for analysis.
    
    Args:
        data: DataFrame to validate
        min_periods: Minimum number of periods required
        
    Returns:
        True if data quality is sufficient, False otherwise
    """
    if data.empty:
        st.error("No data available for analysis")
        return False
    
    if len(data) < min_periods:
        st.warning(f"Limited data available ({len(data)} periods). Consider extending the date range for more reliable results.")
        return len(data) >= APP_CONFIG["min_data_points"]
    
    # Check for excessive missing values
    missing_pct = data.isnull().sum().max() / len(data)
    if missing_pct > 0.2:  # More than 20% missing
        st.warning(f"High percentage of missing data detected ({missing_pct:.1%}). Results may be unreliable.")
    
    return True


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal value as a percentage string."""
    if pd.isna(value) or np.isnan(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places."""
    if pd.isna(value) or np.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}"

# Sidebar controls
with st.sidebar:
    st.header("📊 Data Configuration")
    
    # File upload with better instructions
    uploaded_file = st.file_uploader(
        "Upload asset list (CSV)", 
        type="csv",
        help="CSV file must contain an 'Assets' column with stock ticker symbols"
    )
    
    # Date inputs with validation
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date", 
            value=APP_CONFIG["default_start_date"].date(),
            help="Start date for historical data"
        )
    with col2:
        end_date = st.date_input(
            "End date", 
            value=APP_CONFIG["default_end_date"].date(),
            help="End date for historical data"
        )
    
    # Analysis parameters
    st.subheader("⚙️ Analysis Parameters")
    max_clusters = st.slider(
        "Maximum clusters to test", 
        min_value=2, 
        max_value=40, 
        value=20,
        help="Higher values provide more granular analysis but take longer"
    )
    
    selected_clusters = st.slider(
        "Display details for clusters", 
        min_value=2, 
        max_value=max_clusters, 
        value=3,
        help="Number of clusters to show detailed breakdown for"
    )

# Validate date range
if end_date <= start_date:
    st.error("❌ End date must be after start date")
    st.stop()

# Date range validation
date_diff = (end_date - start_date).days
if date_diff < 30:
    st.warning("⚠️ Short date range may produce unreliable results. Consider using at least 30 days.")
elif date_diff > 3650:  # 10 years
    st.info("ℹ️ Very long date range selected. Analysis may take longer to complete.")

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Enhanced sidebar chat interface
with st.sidebar:
    st.subheader("🤖 Portfolio Analysis Assistant")
    
    # Chat input with placeholder
    user_query = st.text_input(
        "Ask about financial terms:",
        placeholder="e.g., What is Sharpe ratio?",
        help="Ask about any financial term or portfolio concept"
    )
    
    # Handle user input
    if user_query:
        response = get_bot_response(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Chat controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
    with col2:
        if st.button("💡 Show Tips"):
            tips = [
                "Try asking: 'What is beta?'",
                "Ask: 'Explain correlation matrix'",
                "Try: 'What does Sharpe ratio mean?'"
            ]
            for tip in tips:
                st.caption(tip)
    
    # Display chat history with improved styling
    if st.session_state.messages:
        st.markdown("### 💬 Chat History")
        for i, msg in enumerate(st.session_state.messages[-6:]):  # Show last 6 messages
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{i}")
            else:
                message(msg["content"], key=f"assistant_{i}")

# Enhanced main content area
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; margin-bottom: 0.5rem;'>📊 Portfolio Risk Cluster Analysis</h1>
        <p style='color: #666; font-size: 1.1rem; margin-bottom: 0.5rem;'>Created by Mahlatse Ndala</p>
        <p style='color: #888; font-style: italic;'>Explore optimal asset grouping using advanced clustering methodology</p>
    </div>
    """, unsafe_allow_html=True)

# Add information section
if not uploaded_file:
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Upload your asset list**: Use the sidebar to upload a CSV file containing stock ticker symbols
    2. **Select date range**: Choose the historical period for analysis
    3. **Configure parameters**: Adjust clustering settings as needed
    4. **Analyze results**: Review risk metrics, correlation patterns, and optimal portfolio allocations
    
    ### 📋 CSV Format Requirements
    Your CSV file should have an 'Assets' column with stock ticker symbols:
    ```
    Assets
    AAPL
    GOOGL
    MSFT
    TSLA
    ```
    """)
else:
    st.markdown("---")

if uploaded_file:
    try:
        # Load and validate CSV file
        with st.spinner('📁 Processing uploaded file...'):
            assets_df = pd.read_csv(uploaded_file)
            
            # Validate CSV structure
            if 'Assets' not in assets_df.columns:
                st.error("❌ CSV must contain an 'Assets' column with ticker symbols")
                st.info("💡 Ensure your CSV has a column named 'Assets' with stock ticker symbols")
                st.stop()

            # Extract and clean asset list
            assets = assets_df['Assets'].dropna().str.strip().str.upper().unique().tolist()
            
            if not assets:
                st.error('❌ No valid assets found in uploaded file')
                st.info("💡 Make sure the 'Assets' column contains valid stock ticker symbols")
                st.stop()
            
            st.success(f"✅ Successfully loaded {len(assets)} unique assets")
            
            # Display loaded assets in an expandable section
            with st.expander(f"📋 View loaded assets ({len(assets)} total)"):
                # Display assets in columns for better readability
                cols = st.columns(4)
                for i, asset in enumerate(assets):
                    cols[i % 4].write(f"• {asset}")
        
        # Download market data
        data = fetch_stock_data(assets, start_date, end_date)
        if data is None:
            st.stop()
        
        # Process data based on column structure
        with st.spinner('🔄 Processing market data...'):
            # Handle single-level and multi-level column indexes
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    adj_close_data = data['Close'].copy()
                else:
                    st.error("❌ Close price data not found in downloaded data")
                    st.stop()
            else:
                # Single asset case
                if 'Close' in data.columns:
                    adj_close_data = data[['Close']].copy()
                else:
                    adj_close_data = data.copy()  # Assume all data is price data
            
            # Data quality validation
            if not validate_data_quality(adj_close_data):
                st.stop()

            # Clean data
            original_length = len(adj_close_data)
            adj_close_data = adj_close_data.ffill().dropna()
            
            if len(adj_close_data) < APP_CONFIG["min_data_points"]:
                st.error(f"❌ Insufficient data after cleaning ({len(adj_close_data)} points). "
                        f"Minimum {APP_CONFIG['min_data_points']} required.")
                st.stop()
            
            # Show data cleaning results
            if original_length != len(adj_close_data):
                st.info(f"ℹ️ Data cleaned: {original_length} → {len(adj_close_data)} data points")

            # Calculate returns
            returns = adj_close_data.pct_change().dropna()

            if returns.empty:
                st.error("❌ No valid returns data generated")
                st.stop()
            
            st.success(f"✅ Analysis ready: {len(returns)} return periods for {len(returns.columns)} assets")
            
        # Perform clustering analysis
        st.subheader("🔍 Clustering Analysis")
        
        # Initialize progress tracking
        wcss = []
        risks = []
        
        # Create columns for progress display
        col1, col2 = st.columns([3, 1])
        
        with col1:
            progress_bar = st.progress(0)
            status_text = st.empty()
        with col2:
            progress_metric = st.empty()
        
        # Perform clustering for different k values
        for i, n in enumerate(range(1, max_clusters + 1)):
            status_text.text(f"Analyzing {n} clusters...")
            progress_metric.metric("Progress", f"{i+1}/{max_clusters}")
            
            try:
                # Fit K-means clustering
                kmeans = KMeans(
                    n_clusters=n, 
                    n_init=APP_CONFIG["kmeans_init"],
                    random_state=42,  # For reproducibility
                    max_iter=300
                )
                kmeans.fit(returns.T)  # Cluster assets based on return patterns
                wcss.append(kmeans.inertia_)
                
                # Calculate expected risk for each cluster
                labels = kmeans.labels_
                risk_per_cluster = []
                
                for cluster_id in range(n):
                    cluster_returns = returns.iloc[:, labels == cluster_id]
                    if not cluster_returns.empty:
                        cluster_risk = cluster_returns.std().mean() * 100  # Convert to percentage
                        risk_per_cluster.append(cluster_risk)
                    else:
                        risk_per_cluster.append(0)  # Handle empty clusters

                # Store average risk across clusters
                risks.append(np.mean(risk_per_cluster) if risk_per_cluster else 0)
                
            except Exception as e:
                st.warning(f"⚠️ Error in clustering analysis for k={n}: {str(e)}")
                wcss.append(wcss[-1] if wcss else 0)  # Use previous value or 0
                risks.append(risks[-1] if risks else 0)
            
            progress_bar.progress((i + 1) / max_clusters)
        
        # Clear progress indicators
        status_text.empty()
        progress_metric.empty()
        
        st.success("✅ Clustering analysis completed!")
        
        # Prepare visualization data
        num_stocks = list(range(1, max_clusters + 1))
        
        # Create comprehensive plotting data
        plot_data = pd.DataFrame({
            LABELS["num_clusters"]: num_stocks,
            LABELS["expected_risk"]: risks,
            'WCSS': wcss
        })

        # Enhanced visualizations
        st.subheader("📈 Risk Analysis Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Risk vs Clusters", "📉 Elbow Curve", "📋 Summary"])
        
        with tab1:
            # Enhanced risk visualization
            fig_risk = px.scatter(
                plot_data, 
                x=LABELS["num_clusters"], 
                y=LABELS["expected_risk"],
                title='Expected Portfolio Risk by Number of Clusters',
                labels={
                    LABELS["expected_risk"]: LABELS["expected_risk"], 
                    LABELS["num_clusters"]: LABELS["num_clusters"]
                },
                trendline='ols',
                hover_data={'WCSS': ':.2f'}
            )
            
            fig_risk.update_layout(
                title_x=0.5,
                showlegend=True,
                height=500
            )
            
            # Add annotations for key points
            if len(risks) > 2:
                min_risk_idx = np.argmin(risks)
                fig_risk.add_annotation(
                    x=min_risk_idx + 1,
                    y=risks[min_risk_idx],
                    text=f"Minimum Risk: {risks[min_risk_idx]:.2f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red"
                )
            
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with tab2:
            # Enhanced elbow plot
            fig_elbow = px.line(
                x=range(1, max_clusters + 1), 
                y=wcss,
                labels={'x': 'Number of Clusters', 'y': 'Within-Cluster Sum of Squares (WCSS)'},
                title='Elbow Method for Optimal Cluster Number',
                markers=True
            )
            
            # Find and highlight optimal k
            try:
                optimal_k = find_optimal_k(wcss)
                fig_elbow.add_vline(
                    x=optimal_k, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Optimal k = {optimal_k}"
                )
            except Exception:
                optimal_k = 3  # Default fallback
                st.warning(f"⚠️ Could not determine optimal k automatically. Using default: {optimal_k}")
            
            fig_elbow.update_layout(
                title_x=0.5,
                height=500
            )
            
            st.plotly_chart(fig_elbow, use_container_width=True)
        
        with tab3:
            # Analysis summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Optimal Clusters", 
                    optimal_k,
                    help="Based on elbow method analysis"
                )
            
            with col2:
                min_risk = min(risks) if risks else 0
                st.metric(
                    "Minimum Risk", 
                    format_percentage(min_risk/100),
                    help="Lowest portfolio risk achieved"
                )
            
            with col3:
                max_risk = max(risks) if risks else 0
                risk_reduction = ((max_risk - min_risk) / max_risk * 100) if max_risk > 0 else 0
                st.metric(
                    "Risk Reduction", 
                    f"{risk_reduction:.1f}%",
                    help="Maximum possible risk reduction through clustering"
                )
        
        # Key insights
        st.markdown("### 🎯 Key Insights")
        insights = []
        
        if optimal_k <= 5:
            insights.append(f"✅ **Optimal clustering**: {optimal_k} clusters provide the best risk-return balance")
        else:
            insights.append(f"⚠️ **High complexity**: {optimal_k} clusters suggested - consider simpler groupings")
        
        if len(risks) > 2:
            risk_trend = "decreasing" if risks[-1] < risks[0] else "increasing"
            insights.append(f"📊 **Risk trend**: Portfolio risk is generally {risk_trend} with more clusters")
        
        if min_risk < max_risk * 0.8:  # More than 20% risk reduction possible
            insights.append("🎯 **Significant opportunity**: Clustering can substantially reduce portfolio risk")
        
        for insight in insights:
            st.markdown(insight)
        
        # Download enhanced results
        enhanced_results = plot_data.copy()
        enhanced_results['Optimal_K'] = optimal_k
        enhanced_results['Risk_Reduction_Potential'] = f"{risk_reduction:.1f}%"
        
        csv_data = enhanced_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Analysis Results",
            data=csv_data,
            file_name=f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Download complete analysis results as CSV"
        )

        # Enhanced Portfolio Statistics
        st.subheader("📊 Portfolio Performance Metrics")
        
        # Calculate comprehensive statistics
        try:
            # Basic portfolio metrics
            portfolio_vol = returns.std().mean()
            portfolio_return = returns.mean().mean()
            
            # Risk-adjusted metrics
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol != 0 else 0
            
            # Drawdown calculation
            cumulative_returns = (returns + 1).cumprod()
            running_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns / running_max - 1)
            max_drawdown = drawdown.min().min()
            
            # Additional metrics
            skewness = returns.skew().mean()
            kurtosis = returns.kurtosis().mean()
            
            # Display metrics in enhanced grid
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📈 Portfolio Volatility", 
                    format_percentage(portfolio_vol),
                    help="Average volatility across all assets"
                )
                
            with col2:
                st.metric(
                    "💰 Average Return", 
                    format_percentage(portfolio_return),
                    help="Mean daily return across all assets"
                )
                
            with col3:
                st.metric(
                    "⚡ Sharpe Ratio", 
                    format_number(sharpe_ratio),
                    help="Risk-adjusted return measure"
                )
                
            with col4:
                st.metric(
                    "📉 Max Drawdown", 
                    format_percentage(max_drawdown),
                    help="Largest peak-to-trough decline"
                )
            
            # Additional metrics row
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric(
                    "📊 Skewness", 
                    format_number(skewness),
                    help="Asymmetry of return distribution"
                )
                
            with col6:
                st.metric(
                    "📈 Kurtosis", 
                    format_number(kurtosis),
                    help="Tail heaviness of return distribution"
                )
                
            with col7:
                correlation_avg = returns.corr().mean().mean()
                st.metric(
                    "🔗 Avg Correlation", 
                    format_number(correlation_avg),
                    help="Average correlation between assets"
                )
                
            with col8:
                diversification_ratio = len(returns.columns)
                st.metric(
                    "🎯 Assets Count", 
                    diversification_ratio,
                    help="Number of assets in portfolio"
                )
                
        except Exception as e:
            st.error(f"❌ Error calculating portfolio statistics: {str(e)}")
            st.info("💡 This may be due to insufficient data or data quality issues.")
                
        # Enhanced Correlation Analysis
        st.subheader("🔗 Asset Correlation Analysis")
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create tabs for different correlation views
        corr_tab1, corr_tab2 = st.tabs(["🔥 Heatmap", "📈 Network"])
        
        with corr_tab1:
            # Enhanced correlation heatmap
            fig_corr = px.imshow(
                corr_matrix,
                labels={
                    "color": LABELS["correlation_coef"]
                },
                title="Asset Correlation Heatmap",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            
            fig_corr.update_layout(
                title_x=0.5,
                height=max(400, len(corr_matrix) * 20),
                margin={
                    "l": 100, 
                    "r": 50, 
                    "t": 80, 
                    "b": 100
                }
            )
            
            # Add correlation statistics
            high_corr_pairs = []
            low_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    pair = f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}"
                    
                    if corr_val > 0.7:
                        high_corr_pairs.append((pair, corr_val))
                    elif corr_val < -0.3:
                        low_corr_pairs.append((pair, corr_val))
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Show correlation insights
            col1, col2 = st.columns(2)
            
            with col1:
                if high_corr_pairs:
                    st.markdown("#### 🔴 Highly Correlated Pairs (>0.7)")
                    for pair, corr in sorted(high_corr_pairs, key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"• {pair}: {corr:.3f}")
                else:
                    st.info("No highly correlated pairs found")
            
            with col2:
                if low_corr_pairs:
                    st.markdown("#### 🔵 Negatively Correlated Pairs (<-0.3)")
                    for pair, corr in sorted(low_corr_pairs, key=lambda x: x[1])[:5]:
                        st.write(f"• {pair}: {corr:.3f}")
                else:
                    st.info("No significantly negative correlations found")
        
        with corr_tab2:
            # Correlation distribution
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            
            fig_dist = px.histogram(
                x=corr_values,
                nbins=20,
                title="Distribution of Asset Correlations",
                labels={'x': 'Correlation Coefficient', 'y': 'Frequency'}
            )
            
            fig_dist.add_vline(
                x=np.mean(corr_values), 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {np.mean(corr_values):.3f}"
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Rolling Volatility Analysis
        st.subheader("📊 Rolling Volatility Analysis")
        
        # Calculate rolling volatility with multiple windows
        rolling_windows = [7, 30, 90]
        volatility_data = pd.DataFrame(index=returns.index)
        
        for window in rolling_windows:
            if len(returns) >= window:
                rolling_vol = returns.rolling(window=window).std() * np.sqrt(APP_CONFIG["annualization_factor"]) * 100
                volatility_data[f'{window}-day'] = rolling_vol.mean(axis=1)
        
        if not volatility_data.empty:
            fig_vol = px.line(
                volatility_data,
                title="Rolling Volatility Analysis (Annualized %)",
                labels={
                    'value': LABELS["volatility"], 
                    'index': 'Date'
                }
            )
            
            fig_vol.update_layout(
                title_x=0.5,
                height=400,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": 1.02,
                    "xanchor": "right",
                    "x": 1
                }
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.warning("⚠️ Insufficient data for rolling volatility analysis")
        
        # Enhanced Cluster Analysis
        st.subheader(f"🎯 Detailed Cluster Analysis ({selected_clusters} clusters)")
        
        # Perform clustering with selected number of clusters
        kmeans_final = KMeans(
            n_clusters=selected_clusters, 
            n_init=APP_CONFIG["kmeans_init"],
            random_state=42
        )
        labels = kmeans_final.fit_predict(returns.T)
        
        # Calculate cluster statistics
        cluster_stats = []
        
        for i in range(selected_clusters):
            cluster_mask = labels == i
            cluster_returns = returns.iloc[:, cluster_mask]
            assets_in_cluster = returns.columns[cluster_mask].tolist()
            
            if not cluster_returns.empty:
                cluster_vol = cluster_returns.std().mean() * 100
                cluster_ret = cluster_returns.mean().mean() * 100
                cluster_sharpe = cluster_ret / cluster_vol if cluster_vol != 0 else 0
                
                cluster_stats.append({
                    'Cluster': f'Cluster {i+1}',
                    'Assets': len(assets_in_cluster),
                    LABELS["volatility"]: cluster_vol,
                    'Return (%)': cluster_ret,
                    'Sharpe Ratio': cluster_sharpe,
                    LABELS["asset_list"]: assets_in_cluster
                })
        
        # Display cluster summary
        cluster_df = pd.DataFrame(cluster_stats)
        if not cluster_df.empty:
            # Remove Asset List for display
            display_df = cluster_df.drop(LABELS["asset_list"], axis=1)
            st.dataframe(display_df, use_container_width=True)
            
            # Risk contribution visualization
            fig_cluster_risk = px.bar(
                cluster_df,
                x='Cluster',
                y=LABELS["volatility"],
                color='Sharpe Ratio',
                title='Risk and Risk-Adjusted Returns by Cluster',
                text='Assets',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig_cluster_risk, use_container_width=True)
            
            # Detailed cluster breakdown
            st.markdown("### 📋 Cluster Composition")
            for i, row in cluster_df.iterrows():
                with st.expander(
                    f"{row['Cluster']} - {row['Assets']} assets "
                    f"(Vol: {row[LABELS['volatility']]:.2f}%, Sharpe: {row['Sharpe Ratio']:.2f})"
                ):
                    # Display assets in a grid
                    assets_list = row[LABELS["asset_list"]]
                    cols = st.columns(min(4, len(assets_list)))
                    for idx, asset in enumerate(assets_list):
                        cols[idx % 4].write(f"• **{asset}**")
        
        # Enhanced Efficient Frontier
        st.subheader("📈 Efficient Frontier Analysis")
        
        try:
            risks_ef, returns_ef = calculate_efficient_frontier(returns, APP_CONFIG["default_portfolios"])
            
            # Create efficient frontier plot
            ef_fig = px.scatter(
                x=risks_ef * 100, 
                y=returns_ef * 100,
                title='Portfolio Efficient Frontier',
                labels={'x': 'Portfolio Risk (%)', 'y': 'Expected Return (%)'},
                opacity=0.6
            )
            
            # Add current portfolio point
            current_risk = returns.std().mean() * np.sqrt(APP_CONFIG["annualization_factor"]) * 100
            current_return = returns.mean().mean() * APP_CONFIG["annualization_factor"] * 100
            
            ef_fig.add_scatter(
                x=[current_risk],
                y=[current_return],
                mode='markers',
                marker={
                    "size": 15, 
                    "color": 'red', 
                    "symbol": 'star'
                },
                name='Current Portfolio',
                text=['Current Portfolio'],
                textposition='top center'
            )
            
            # Add optimal portfolio (highest Sharpe ratio)
            sharpe_ratios = returns_ef / risks_ef
            optimal_idx = np.argmax(sharpe_ratios)
            
            ef_fig.add_scatter(
                x=[risks_ef[optimal_idx] * 100],
                y=[returns_ef[optimal_idx] * 100],
                mode='markers',
                marker={
                    "size": 15, 
                    "color": 'green', 
                    "symbol": 'diamond'
                },
                name='Optimal Portfolio',
                text=['Optimal (Max Sharpe)'],
                textposition='top center'
            )
            
            ef_fig.update_layout(
                title_x=0.5,
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(ef_fig, use_container_width=True)
            
            # Efficient frontier insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Risk",
                    f"{current_risk:.2f}%",
                    help="Current portfolio risk level"
                )
            
            with col2:
                st.metric(
                    "Optimal Risk",
                    f"{risks_ef[optimal_idx] * 100:.2f}%",
                    help="Risk level at optimal portfolio"
                )
            
            with col3:
                optimal_sharpe = sharpe_ratios[optimal_idx]
                st.metric(
                    "Max Sharpe Ratio",
                    f"{optimal_sharpe:.3f}",
                    help="Highest achievable risk-adjusted return"
                )
                
        except Exception as e:
            st.error(f"❌ Error generating efficient frontier: {str(e)}")
            st.info("💡 This may be due to insufficient data or numerical issues.")

    except KeyError as e:
        st.error(f"❌ Invalid CSV format: {str(e)}")
        st.info("💡 Ensure your CSV has an 'Assets' column with valid ticker symbols")
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.info("💡 Please check your data format and try again")
        
        # Show debug information in expander
        with st.expander("🔍 Debug Information"):
            st.write("**Error details:**", str(e))
            st.write("**Error type:**", type(e).__name__)
            if uploaded_file:
                st.write("**File name:**", uploaded_file.name)
                st.write("**File size:**", uploaded_file.size, "bytes")

# Add footer with additional information
st.markdown("---")
st.markdown("""
### 📚 About This Application

This Portfolio Risk Cluster Analysis tool uses advanced machine learning techniques to help you:

- **Optimize portfolio construction** through intelligent asset grouping
- **Minimize risk** while maintaining expected returns
- **Identify correlation patterns** between different assets
- **Visualize efficient frontiers** for optimal allocation strategies

#### 🔬 Methodology
- **K-means clustering** groups assets based on return patterns
- **Elbow method** determines optimal number of clusters
- **Monte Carlo simulation** generates efficient frontier
- **Risk metrics** include volatility, Sharpe ratio, and maximum drawdown

#### 💡 Tips for Better Results
- Use at least 30-50 assets for meaningful clustering
- Select date ranges with sufficient market data (1+ years)
- Consider different market conditions in your analysis
- Regularly update analysis with fresh data

---
<div style='text-align: center; color: #666; padding: 1rem 0;'>
    <p>© 2025 Mahlatse Ndala | Portfolio Risk Cluster Analysis v2.0</p>
    <p style='font-size: 0.8rem;'>Built with ❤️ using Streamlit, Plotly, and scikit-learn</p>
</div>
""", unsafe_allow_html=True)
