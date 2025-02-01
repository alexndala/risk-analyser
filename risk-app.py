import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Risk Cluster Analyzer", layout="wide")

# Sidebar controls
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload asset list (CSV)", type="csv")
    start_date = st.date_input("Start date", value=pd.to_datetime('2024-01-01'))
    end_date = st.date_input("End date", value=pd.to_datetime('2024-12-31'))
    max_clusters = st.slider("Maximum clusters to test", 2, 40, 20)  # Increased range

# Main content area
st.title("Portfolio Risk Cluster Analysis")
st.markdown("""Explore optimal asset grouping using elbow curve methodology.""")

if uploaded_file:
    assets_df = pd.read_csv(uploaded_file)
    assets = list(assets_df['Assets'].unique())
    
    with st.spinner('Downloading market data...'):
        try:
            # Downloading data from Yahoo Finance
            data = yf.download(assets, start=start_date, end=end_date)
            
            # Handle single-level and multi-level column indexes
            if isinstance(data.columns, pd.MultiIndex):
                adj_close_data = data['Close']
            else:
                adj_close_data = data[['Close']]  # Single-level index
            
            # Calculate returns
            returns = adj_close_data.pct_change().dropna()
            
            # Calculate WCSS for elbow curve and standard deviation for clusters
            wcss = []
            risks = []
            for n in range(1, max_clusters + 1):
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

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
else:
    st.info("Please upload a CSV file containing asset tickers to begin analysis.")

