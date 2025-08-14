# 📊 Portfolio Risk Cluster Analysis v2.0

A comprehensive Streamlit web application for analyzing portfolio risk using advanced clustering techniques and visualizing asset relationships with modern data science methodologies.

## ✨ Enhanced Features

### 🔍 Core Analysis
- **Intelligent Asset Clustering**: K-means clustering with optimal cluster determination
- **Interactive Date Range Selection**: Flexible historical data analysis
- **Advanced Risk Metrics**: Sharpe ratio, maximum drawdown, skewness, kurtosis
- **Efficient Frontier Visualization**: Monte Carlo simulation for portfolio optimization
- **Rolling Volatility Analysis**: Multiple time windows (7, 30, 90 days)

### 📈 Visualizations
- **Enhanced Correlation Heatmaps**: Interactive correlation analysis with insights
- **Risk Distribution Charts**: Comprehensive risk profiling across clusters
- **Elbow Curve Analysis**: Automated optimal cluster detection
- **Efficient Frontier Plots**: Risk-return optimization visualization

### 🤖 AI Assistant
- **Financial Terms Chatbot**: Interactive assistant for portfolio concepts
- **Contextual Help**: Real-time explanations of financial metrics
- **Smart Recommendations**: Data-driven insights and suggestions

### 🛡️ Robust Architecture
- **Comprehensive Error Handling**: Graceful handling of data issues
- **Data Quality Validation**: Automatic data cleaning and validation
- **Performance Optimization**: Cached data fetching and efficient processing
- **Type Safety**: Full type hints for better code reliability

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/risk-analyser.git
cd risk-analyser
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run risk-app.py
```

## 📋 Data Format

Your CSV file should contain an 'Assets' column with stock ticker symbols:

```csv
Assets
FSR.JO
MTN.JO
KIO.JO
TSLA
NVDA
```

## 🔧 Configuration

The application includes configurable parameters:
- **Maximum clusters**: Adjust analysis granularity (2-40 clusters)
- **Date ranges**: Flexible historical period selection
- **Rolling windows**: Customizable volatility analysis periods

## 📊 Analysis Output

### Risk Metrics
- Portfolio volatility and returns
- Sharpe ratio and risk-adjusted returns
- Maximum drawdown analysis
- Distribution characteristics (skewness, kurtosis)

### Clustering Results
- Optimal cluster number determination
- Risk contribution by cluster
- Asset composition breakdown
- Correlation pattern analysis

### Optimization Tools
- Efficient frontier generation
- Optimal portfolio identification
- Risk reduction opportunities
- Performance benchmarking

## 🏗️ Technical Architecture

### Dependencies
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization
- **scikit-learn**: Machine learning algorithms
- **yfinance**: Financial data API
- **pandas/numpy**: Data processing

### Key Enhancements in v2.0
- ✅ Type hints throughout codebase
- ✅ Comprehensive error handling
- ✅ Enhanced user interface
- ✅ Performance optimizations
- ✅ Expanded financial metrics
- ✅ Interactive visualizations
- ✅ Data quality validation
- ✅ Modular code structure

## 📈 Use Cases

1. **Portfolio Construction**: Build diversified portfolios using clustering insights
2. **Risk Management**: Identify and mitigate concentration risks
3. **Asset Allocation**: Optimize weights based on efficient frontier analysis
4. **Performance Analysis**: Track rolling metrics and correlation changes
5. **Educational Tool**: Learn about modern portfolio theory concepts

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
**Created by Mahlatse Ndala** | Portfolio Risk Analysis Specialist
