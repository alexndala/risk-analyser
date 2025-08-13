import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.cluster import KMeans
from streamlit_chat import message

st.set_page_config(page_title="Risk Cluster Analyzer", layout="wide")

FINANCIAL_TERMS = {
    "sharpe ratio": "The Sharpe ratio measures the risk-adjusted return of an investment. Higher is better.",
    "efficient frontier": "A set of optimal portfolios that offer the highest expected return for a defined level of risk.",
    "correlation matrix": "Shows how different assets move in relation to each other. Range from -1 to +1.",
    "max drawdown": "The maximum observed loss from a peak to a trough of a portfolio.",
    "elbow curve": "Method to determine optimal number of clusters by looking at the rate of improvement.",
    "kmeans clustering": "Algorithm that groups similar assets together based on return patterns.",
    "portfolio volatility": "Measures how much portfolio returns fluctuate over time.",
    "risk contribution": "How much each asset or cluster contributes to total portfolio risk."
}

def get_bot_response(user_query):
    query = user_query.lower()
    for term, explanation in FINANCIAL_TERMS.items():
        if term in query:
            return explanation
    return "I'm not sure about that. Try asking about specific terms like 'Sharpe ratio' or 'efficient frontier'."

@st.cache_data
def fetch_stock_data(assets, start_date, end_date):
    try:
        data = yf.download(assets, start=start_date, end=end_date)
        if data is None or data.empty:
            st.error("No data downloaded. Check assets and date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def calculate_efficient_frontier(returns, num_portfolios=1000):
    num_assets = len(returns.columns)
    returns_mean = returns.mean()
    cov_matrix = returns.cov()
    
    port_returns = np.zeros(num_portfolios)
    port_risks = np.zeros(num_portfolios)
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        # Annualized expected return and risk
        port_returns[i] = np.sum(returns_mean * weights) * 252
        port_risks[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    
    return port_risks, port_returns

def find_optimal_k(wcss):
    # Need at least 3 points for second diff ratio
    if len(wcss) < 3:
        return max(1, len(wcss))
    diff = np.diff(wcss)
    denom = np.where(diff[:-1] == 0, np.nan, diff[:-1])
    diff_r = diff[1:] / denom
    # Handle possible NaNs/Infs in ratio
    valid = np.isfinite(diff_r)
    if not np.any(valid):
        return max(1, len(wcss))
    valid_indices = np.where(valid)[0]
    best_local = np.argmin(np.abs(diff_r[valid] - 1))
    elbow_idx = int(valid_indices[best_local]) + 2  # +2 because of diff and diff_r indexing
    return int(np.clip(elbow_idx, 1, len(wcss)))

# Sidebar controls
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload asset list (CSV)", type="csv")
    start_date = st.date_input("Start date", value=pd.to_datetime('2024-01-01'))
    end_date = st.date_input("End date", value=pd.to_datetime('2024-12-31'))
    max_clusters = st.slider("Maximum clusters to test", 2, 40, 20)
    selected_clusters = st.slider(
        "Select number of clusters to display details",
        min_value=2,
        max_value=max_clusters,
        value=3
    )

# Check if end_date is after start_date
if end_date <= start_date:
    st.error("End date must be after start date")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar chat interface
with st.sidebar:
    st.subheader("Portfolio Analysis Assistant")
    user_query = st.text_input("Ask about any financial term:")
    
    if user_query:
        response = get_bot_response(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
    
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            message(msg["content"], is_user=True)
        else:
            message(msg["content"])

# Main content area
st.markdown("""
    <h1 style='text-align: center;'>Portfolio Risk Cluster Analysis</h1>
    <p style='text-align: center;'>Created by Mahlatse Ndala</p>
    """, unsafe_allow_html=True)

st.markdown("""
    <br>
    <p style='text-align: center;'>Explore optimal asset grouping using elbow curve methodology.</p>
    """, unsafe_allow_html=True)

if uploaded_file:
    try:
        assets_df = pd.read_csv(uploaded_file)
        if 'Assets' not in assets_df.columns:
            st.error("CSV must contain an 'Assets' column with tickers")
            st.stop()

        assets = list(assets_df['Assets'].dropna().astype(str).str.strip().unique())
        if not assets:
            st.error('No assets found in uploaded file')
            st.stop()
        
        with st.spinner('Downloading market data...'):
            data = fetch_stock_data(assets, start_date, end_date)
            if data is None:
                st.stop()
            
            # Choose Adj Close if available; otherwise Close
            if isinstance(data.columns, pd.MultiIndex):
                top_level_cols = data.columns.get_level_values(0)
                price_key = 'Adj Close' if 'Adj Close' in set(top_level_cols) else 'Close'
                price_data = data[price_key].copy()
            else:
                price_key = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                price_data = data[[price_key]].copy()
                # Rename single ticker column to the ticker symbol if applicable
                if len(assets) == 1:
                    price_data.columns = [assets[0]]

            # Drop assets with all-NaN price history, then forward-fill and drop remaining NaNs
            price_data = price_data.dropna(axis=1, how='all')
            if price_data.shape[1] == 0:
                st.error("No valid price data for the provided tickers and date range.")
                st.stop()

            price_data.ffill(inplace=True)
            price_data.dropna(inplace=True)

            if len(price_data) < 5:
                st.error("Insufficient data after cleaning. Try a wider date range.")
                st.stop()

            # Calculate daily returns
            returns = price_data.pct_change().dropna()
            if returns.empty:
                st.error("No valid data found for the selected date range")
                st.stop()

            num_assets = returns.shape[1]
            if num_assets < 2:
                st.warning("At least 2 assets are required for clustering. Displaying portfolio statistics only.")
            
            # Build elbow WCSS and expected risk by number of clusters (capped at number of assets)
            kmax = int(min(max_clusters, num_assets)) if num_assets >= 2 else 0
            wcss = []
            exp_risks = []

            if kmax >= 1:
                progress_bar = st.progress(0.0)
                for i, n in enumerate(range(1, kmax + 1)):
                    kmeans = KMeans(n_clusters=n, n_init=10, random_state=42)
                    kmeans.fit(returns.T)  # assets x features
                    wcss.append(kmeans.inertia_)
                    
                    # Average cluster risk (annualized)
                    labels = kmeans.labels_
                    cluster_risks = []
                    for cluster_id in range(n):
                        cols_in_cluster = returns.columns[labels == cluster_id]
                        if len(cols_in_cluster) > 0:
                            cluster_returns = returns[cols_in_cluster]
                            cluster_vol = cluster_returns.std().mean() * np.sqrt(252) * 100.0
                            cluster_risks.append(cluster_vol)
                        else:
                            cluster_risks.append(0.0)
                    exp_risks.append(float(np.mean(cluster_risks)))
                    progress_bar.progress((i + 1) / kmax)

            # Plot Expected Risk vs Number of Clusters (if at least 1 k was computed)
            if len(exp_risks) > 0:
                ks = list(range(1, kmax + 1))
                plot_data = pd.DataFrame({
                    'Number of Clusters': ks,
                    'Expected Risk (%)': exp_risks
                })
                # Try OLS trendline if statsmodels is available
                try:
                    fig = px.scatter(
                        plot_data, x='Number of Clusters', y='Expected Risk (%)',
                        title='Expected Risk vs Number of Clusters',
                        trendline='ols'
                    )
                except Exception:
                    fig = px.scatter(
                        plot_data, x='Number of Clusters', y='Expected Risk (%)',
                        title='Expected Risk vs Number of Clusters'
                    )
                st.plotly_chart(fig, use_container_width=True)

                # Elbow plot
                fig_elbow = px.line(
                    x=ks, y=wcss,
                    labels={'x': 'Number of Clusters', 'y': 'Within-Cluster Variance (Inertia)'},
                    title='Elbow Method For Optimal k'
                )
                st.plotly_chart(fig_elbow, use_container_width=True)

                optimal_k = find_optimal_k(wcss)
                st.write(f"### Optimal Number of Clusters: {optimal_k}")
                st.write("Based on the elbow curve methodology, this is where the rate of improvement significantly slows down.")

                # Allow CSV download for the risk-vs-clusters data
                csv = plot_data.to_csv(index=False)
                st.download_button(
                    label="Download analysis results",
                    data=csv,
                    file_name="risk_analysis.csv",
                    mime="text/csv"
                )

            # Portfolio statistics
            st.subheader("Portfolio Statistics")
            col1, col2, col3, col4 = st.columns(4)
            try:
                daily_vol = returns.std().mean()
                daily_ret = returns.mean().mean()

                ann_vol = daily_vol * np.sqrt(252)
                ann_ret = daily_ret * 252

                with col1:
                    st.metric("Portfolio Volatility", f"{ann_vol*100:.2f}%")

                with col2:
                    st.metric("Average Return", f"{ann_ret*100:.2f}%")

                with col3:
                    sharpe = (ann_ret / ann_vol) if np.isfinite(ann_vol) and ann_vol > 0 else np.nan
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")

                # Equal-weighted portfolio drawdown
                try:
                    port_series = returns.mean(axis=1)
                    cum = (1 + port_series).cumprod()
                    dd_series = cum / cum.cummax() - 1
                    drawdown = dd_series.min()
                    drawdown_display = f"{drawdown*100:.2f}%" if not np.isnan(drawdown) else "N/A"
                except Exception:
                    drawdown_display = "N/A"
                with col4:
                    st.metric("Max Drawdown", drawdown_display)

            except Exception as e:
                st.error(f"Error calculating portfolio statistics: {str(e)}")

            # Correlation matrix
            st.subheader("Correlation Matrix")
            corr_matrix = returns.corr()
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                title="Asset Correlation Heatmap"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Rolling volatility (annualized)
            st.subheader("Rolling Volatility")
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
            fig_vol = px.line(
                rolling_vol,
                title="30-Day Rolling Volatility",
                labels={'value': 'Volatility (%)', 'index': 'Date'}
            )
            st.plotly_chart(fig_vol, use_container_width=True)

            # Risk contribution per cluster with asset details (only if at least 2 assets)
            if num_assets >= 2:
                st.subheader(f"Risk Contribution (for up to {selected_clusters} clusters)")
                # Clamp selected clusters to available assets
                effective_clusters = int(min(selected_clusters, num_assets))
                if effective_clusters < 2:
                    st.info("Not enough assets for clustering.")
                else:
                    kmeans = KMeans(n_clusters=effective_clusters, n_init=10, random_state=42)
                    labels = kmeans.fit_predict(returns.T)

                    cluster_risks = []
                    st.write("### Assets in Each Cluster:")
                    for i in range(effective_clusters):
                        cols_in_cluster = returns.columns[labels == i]
                        cluster_returns = returns[cols_in_cluster] if len(cols_in_cluster) > 0 else pd.DataFrame()
                        if not cluster_returns.empty:
                            cluster_vol = cluster_returns.std().mean() * np.sqrt(252) * 100.0
                            assets_in_cluster = cols_in_cluster.tolist()
                        else:
                            cluster_vol = 0.0
                            assets_in_cluster = []

                        cluster_risks.append({
                            'Cluster': f'Cluster {i+1}',
                            'Risk (%)': cluster_vol,
                            'Assets': len(assets_in_cluster)
                        })

                        with st.expander(f"Cluster {i+1} Assets ({len(assets_in_cluster)} stocks)"):
                            st.write(", ".join(assets_in_cluster) if assets_in_cluster else "No assets")

                    risk_df = pd.DataFrame(cluster_risks)
                    fig_risk = px.bar(
                        risk_df,
                        x='Cluster',
                        y='Risk (%)',
                        title='Risk per Cluster',
                        text='Assets'
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)

            # Efficient Frontier
            st.subheader("Efficient Frontier")
            ef_risks, ef_rets = calculate_efficient_frontier(returns)
            ef_fig = px.scatter(
                x=ef_risks * 100,
                y=ef_rets * 100,
                title='Portfolio Efficient Frontier',
                labels={'x': 'Portfolio Risk (%)', 'y': 'Expected Return (%)'}
            )

            current_risk = returns.std().mean() * np.sqrt(252) * 100
            current_return = returns.mean().mean() * 252 * 100
            ef_fig.add_scatter(
                x=[current_risk],
                y=[current_return],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='Current Portfolio'
            )
            st.plotly_chart(ef_fig, use_container_width=True)

    except KeyError:
        st.error("Invalid CSV format. Must contain 'Assets' column.")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
