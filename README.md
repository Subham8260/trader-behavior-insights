# 📊 Bitcoin Trader Behavior Insights — Hyperliquid × Fear & Greed Index

## Overview
Deep-dive analysis of **211,224 trades** from **32 traders** on Hyperliquid (May 2023 – May 2025), merged with the **Bitcoin Fear & Greed Index** to uncover how market sentiment drives trader performance, behavior, and profitability.

## 🔍 Objectives
- Explore the relationship between Bitcoin market sentiment and trader PnL
- Identify which sentiment regimes produce the best/worst trading outcomes
- Cluster traders into behavioral archetypes using unsupervised ML
- Build a predictive model to classify profitable trades using sentiment features

## 📊 Key Findings

| Metric | Result |
|--------|--------|
| Total Trades Analysed | 211,224 |
| Closed Trades | 84,691 |
| Unique Traders | 32 |
| Overall Win Rate | 83.5% |
| Best Sentiment to Trade | Fear (88.6% win rate, $126 avg PnL) |
| Worst Sentiment to Trade | Greed (76.1% win rate, $69 avg PnL) |
| ML Model ROC-AUC | 89.2% (Random Forest & Gradient Boost) |
| Trader Archetypes Found | 4 Clusters |

## 📈 Visualizations

| Chart | Description |
|-------|-------------|
| `fig1_sentiment_distribution.png` | Pie + bar chart of Fear/Greed day distribution |
| `fig2_pnl_by_sentiment.png` | Avg, median & total PnL across sentiment classes |
| `fig3_winrate_by_sentiment.png` | Win rate % and trade volume by sentiment |
| `fig4_heatmap_coin_sentiment.png` | Avg PnL heatmap: Top 10 coins × Sentiment |
| `fig5_long_short_sentiment.png` | Long vs Short performance by sentiment |
| `fig6_timeseries.png` | Cumulative PnL overlaid with Fear & Greed Index |
| `fig7_clustering.png` | K-Means trader segmentation (PCA + metrics) |
| `fig8_ml_model.png` | ML model comparison + feature importance |
| `fig9_violin_pnl.png` | PnL distribution violin plot by sentiment |

## 💡 Strategy Insights

1. **Buy Fear, Fade Greed** — Long trades during Fear and Short trades during Greed consistently outperform
2. **Sentiment as a Signal** — Fear & Greed index value is a top-3 feature in the ML model
3. **Coin Selection Matters** — HYPE outperforms across all regimes; BTC/ETH are best during Fear
4. **Elite traders** increase activity during Fear periods, confirming contrarian edge

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **Visualization** — matplotlib, seaborn
- **Machine Learning** — scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **Clustering** — K-Means + PCA
- **Datasets** — Hyperliquid Historical Trades + Alternative.me Fear & Greed Index
- 
## 📁 Files

| File | Description |
|------|-------------|
| `Trader_Behavior_Insights.ipynb` | Full Jupyter notebook with code + insights |
| `analysis.py` | Standalone Python script |
| `fig1–fig9.png` | All 9 visualizations |

## 👤 Author
Subham Kumar Dash
