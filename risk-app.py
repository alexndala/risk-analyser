import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Risk Cluster Analyzer", layout="wide")

@st.cache_data
def fetch_stock_data(assets, start_date, end_date):
    return yf.download(assets, start=start_date, end=end_date)

def calculate_efficient_frontier(returns, num_portfolios=1000):
    num_assets = len(returns.columns)
    returns_mean = returns.mean()
    cov_matrix = returns.cov()
    
    # Arrays to store returns and risks
    port_returns = np.zeros(num_portfolios)
    port_risks = np.zeros(num_portfolios)
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        
        # Calculate portfolio metrics
        port_returns[i] = np.sum(returns_mean * weights) * 252
        port_risks[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    
    return port_risks, port_returns

def find_optimal_k(wcss):
    # Calculate the rate of change in wcss
    diff = np.diff(wcss)
    diff_r = diff[1:] / diff[:-1]
    
    # Find the point where the ratio of differences is closest to 1
    elbow_idx = np.argmin(np.abs(diff_r - 1)) + 2
    return elbow_idx

# Sidebar controls
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload asset list (CSV)", type="csv")
    start_date = st.date_input("Start date", value=pd.to_datetime('2024-01-01'))
    end_date = st.date_input("End date", value=pd.to_datetime('2024-12-31'))
    max_clusters = st.slider("Maximum clusters to test", 2, 40, 20)  # Increased range
    # Add cluster selection
    selected_clusters = st.slider("Select number of clusters to display details", 
                                min_value=2, 
                                max_value=max_clusters, 
                                value=3)

# Check if end_date is after start_date
if end_date <= start_date:
    st.error("End date must be after start date")
    st.stop()

# Main content area
st.title("Portfolio Risk Cluster Analysis")
st.markdown("""Explore optimal asset grouping using elbow curve methodology.""")

if uploaded_file:
    assets_df = pd.read_csv(uploaded_file)
    assets = list(assets_df['Assets'].unique())
    
    with st.spinner('Downloading market data...'):
        try:
            # Downloading data from Yahoo Finance
            data = fetch_stock_data(assets, start_date, end_date)
            
            # Handle single-level and multi-level column indexes
            if isinstance(data.columns, pd.MultiIndex):
                adj_close_data = data['Close']
            else:
                adj_close_data = data[['Close']]  # Single-level index
            
            # Calculate returns
            returns = adj_close_data.pct_change().dropna()

            if returns.empty or returns.isnull().all().all():
                st.error("No valid data found for the selected date range")
                st.stop()
            
            # Calculate WCSS for elbow curve and standard deviation for clusters
            wcss = []
            risks = []
            progress_bar = st.progress(0)
            for i, n in enumerate(range(1, max_clusters + 1)):
                kmeans = KMeans(n_clusters=n, n_init=10)
                kmeans.fit(returns.T)  # Cluster assets based on return patterns
                wcss.append(kmeans.inertia_)
                
                # Calculate expected risk (standard deviation) for each cluster
                labels = kmeans.labels_
                risk_per_cluster = []
                for cluster_id in range(n):
                    cluster_returns = returns.iloc[:, labels == cluster_id]
                    if not cluster_returns.empty:
                        risk_per_cluster.append(cluster_returns.std().mean() * 100)  # Convert to percentage
                    else:
                        risk_per_cluster.append(0)  # Handle empty clusters

                risks.append(np.mean(risk_per_cluster))  # Average risk across clusters
                progress_bar.progress((i + 1) / max_clusters)

            # Prepare data for plotting
            num_stocks = list(range(1, max_clusters + 1))
            risk_percentage = [np.mean([returns.iloc[:, kmeans.labels_ == i].std().mean() * 100 for i in range(n)]) for n in num_stocks]

            # Create a DataFrame for plotting
            plot_data = pd.DataFrame({
                'Number of Stocks': num_stocks,
                'Expected Risk (%)': risk_percentage
            })

            # Plotting expected risk vs number of stocks
            fig = px.scatter(plot_data, x='Number of Stocks', y='Expected Risk (%)',
                             title='Expected Risk vs Number of Stocks',
                             labels={'Expected Risk (%)': 'Expected Risk (%)', 'Number of Stocks': 'Number of Stocks'},
                             trendline='ols')
            st.plotly_chart(fig, use_container_width=True)

            # Show elbow plot
            fig_elbow = px.line(x=range(1, max_clusters + 1), y=wcss,
                                labels={'x': 'Number of Clusters', 'y': 'Within-Cluster Variance'},
                                title='Elbow Method For Optimal k')
            st.plotly_chart(fig_elbow, use_container_width=True)

            optimal_k = find_optimal_k(wcss)
            st.write(f"### Optimal Number of Clusters: {optimal_k}")
            st.write("Based on the elbow curve methodology, this is where the rate of improvement significantly slows down.")

            if 'plot_data' in locals():
                csv = plot_data.to_csv(index=False)
                st.download_button(
                    label="Download analysis results",
                    data=csv,
                    file_name="risk_analysis.csv",
                    mime="text/csv"
                )

            if 'returns' in locals():
                st.subheader("Portfolio Statistics")
                
                # Create 4 columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                try:
                    # Calculate metrics safely with NaN handling
                    portfolio_vol = returns.std().mean()
                    portfolio_return = returns.mean().mean()
                    
                    # Format metrics with NaN checks
                    with col1:
                        vol_display = f"{portfolio_vol*100:.2f}%" if not np.isnan(portfolio_vol) else "N/A"
                        st.metric("Portfolio Volatility", vol_display)
                    
                    with col2:
                        ret_display = f"{portfolio_return*100:.2f}%" if not np.isnan(portfolio_return) else "N/A"
                        st.metric("Average Return", ret_display)
                    
                    with col3:
                        if not np.isnan(portfolio_vol) and portfolio_vol != 0:
                            sharpe_ratio = portfolio_return / portfolio_vol
                            sharpe_display = f"{sharpe_ratio:.2f}"
                        else:
                            sharpe_display = "N/A"
                        st.metric("Sharpe Ratio", sharpe_display)
                    
                    with col4:
                        try:
                            cumulative_returns = (returns + 1).cumprod()
                            running_max = cumulative_returns.cummax()
                            drawdown = (cumulative_returns / running_max - 1).min()
                            drawdown_display = f"{drawdown*100:.2f}%" if not np.isnan(drawdown) else "N/A"
                        except:
                            drawdown_display = "N/A"
                        st.metric("Max Drawdown", drawdown_display)
                        
                except Exception as e:
                    st.error(f"Error calculating portfolio statistics: {str(e)}")
                
                # Enhanced correlation matrix
                st.subheader("Correlation Matrix")
                corr_matrix = returns.corr()
                fig_corr = px.imshow(corr_matrix,
                                     labels=dict(color="Correlation"),
                                     title="Asset Correlation Heatmap",
                                     width=800,
                                     height=1000)
                fig_corr.update_layout(
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Add rolling volatility
                st.subheader("Rolling Volatility")
                rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100  # Annualized
                fig_vol = px.line(rolling_vol, 
                                  title="30-Day Rolling Volatility",
                                  labels={'value': 'Volatility (%)', 'index': 'Date'})
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Risk contribution per cluster with asset details
                st.subheader(f"Risk Contribution (for {selected_clusters} clusters)")
                kmeans = KMeans(n_clusters=selected_clusters, n_init=10)
                labels = kmeans.fit_predict(returns.T)
                
                cluster_risks = []
                st.write("### Assets in Each Cluster:")
                for i in range(selected_clusters):
                    cluster_returns = returns.iloc[:, labels == i]
                    cluster_vol = cluster_returns.std().mean() * 100
                    assets_in_cluster = returns.columns[labels == i].tolist()
                    
                    cluster_risks.append({
                        'Cluster': f'Cluster {i+1}',
                        'Risk (%)': cluster_vol,
                        'Assets': len(assets_in_cluster)
                    })
                    
                    # Display assets in expandable section
                    with st.expander(f"Cluster {i+1} Assets ({len(assets_in_cluster)} stocks)"):
                        st.write(", ".join(assets_in_cluster))

                risk_df = pd.DataFrame(cluster_risks)
                fig_risk = px.bar(risk_df,
                                  x='Cluster',
                                  y='Risk (%)',
                                  title='Risk per Cluster',
                                  text='Assets')
                st.plotly_chart(fig_risk, use_container_width=True)

                # Add Efficient Frontier plot
                st.subheader("Efficient Frontier")
                risks, rets = calculate_efficient_frontier(returns)

                ef_fig = px.scatter(
                    x=risks*100, 
                    y=rets*100,
                    title='Portfolio Efficient Frontier',
                    labels={'x': 'Portfolio Risk (%)', 'y': 'Expected Return (%)'}
                )

                # Add current portfolio point
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

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please upload a CSV file containing asset tickers to begin analysis.")
