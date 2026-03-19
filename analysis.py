import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── Color palette ──────────────────────────────────────────────────────────────
PALETTE = {
    'Extreme Fear': '#d32f2f',
    'Fear':         '#ff7043',
    'Neutral':      '#ffd600',
    'Greed':        '#66bb6a',
    'Extreme Greed':'#1565c0',
}
SENTIMENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
BG      = '#0d1117'
CARD    = '#161b22'
TEXT    = '#e6edf3'
ACCENT  = '#58a6ff'
GREEN   = '#3fb950'
RED     = '#f85149'
YELLOW  = '#d29922'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'axes.edgecolor': '#30363d', 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': '#21262d',
    'grid.linestyle': '--', 'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv('/home/claude/historical_data.csv')
fg = pd.read_csv('/mnt/user-data/uploads/fear_greed_index.csv')

# Parse dates
df['datetime'] = pd.to_datetime(df['Timestamp IST'], dayfirst=True, errors='coerce')
df['date']     = df['datetime'].dt.date.astype(str)
fg['date']     = pd.to_datetime(fg['date']).dt.date.astype(str)

# Merge
df = df.merge(fg[['date','value','classification']], on='date', how='left')
df.rename(columns={'value':'fg_value', 'classification':'sentiment'}, inplace=True)

# Filter only rows with closed trades (PnL != 0 for closed trades, but keep all for volume analysis)
df['is_closed'] = df['Direction'].isin(['Close Long', 'Close Short'])
closed = df[df['is_closed']].copy()
closed['profit'] = closed['Closed PnL'] > 0

# Sentiment order
df['sentiment'] = pd.Categorical(df['sentiment'], categories=SENTIMENT_ORDER, ordered=True)
closed['sentiment'] = pd.Categorical(closed['sentiment'], categories=SENTIMENT_ORDER, ordered=True)

print(f"Total trades: {len(df):,}  |  Closed trades: {len(closed):,}  |  Accounts: {df['Account'].nunique()}")
print(f"Date range: {df['date'].min()} → {df['date'].max()}")
print("Merge check – sentiment coverage:", df['sentiment'].notna().mean()*100, "%")

# ══════════════════════════════════════════════════════════════════════════════
# 2. HELPER
# ══════════════════════════════════════════════════════════════════════════════
def savefig(name, tight=True):
    path = f'/home/claude/{name}.png'
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  ✓ saved {name}.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIG 1 – SENTIMENT DISTRIBUTION  (pie + bar)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Sentiment distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle('Bitcoin Market Sentiment Distribution', fontsize=18, fontweight='bold', color=TEXT, y=1.01)

sent_counts = fg['classification'].value_counts().reindex(SENTIMENT_ORDER).dropna()
colors = [PALETTE[s] for s in sent_counts.index]

# Pie
axes[0].pie(sent_counts, labels=sent_counts.index, colors=colors,
            autopct='%1.1f%%', startangle=140,
            textprops={'color': TEXT, 'fontsize': 11},
            wedgeprops={'edgecolor': BG, 'linewidth': 2})
axes[0].set_title('Overall Sentiment Split\n(2018–2025)', color=TEXT, fontsize=13)

# Bar
bars = axes[1].bar(sent_counts.index, sent_counts.values, color=colors, edgecolor=BG, linewidth=1.5)
axes[1].set_title('Days per Sentiment Category', color=TEXT, fontsize=13)
axes[1].set_xlabel('Sentiment', fontsize=11)
axes[1].set_ylabel('Number of Days', fontsize=11)
axes[1].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, sent_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(int(val)), ha='center', va='bottom', color=TEXT, fontsize=10, fontweight='bold')
axes[1].grid(axis='y')
savefig('fig1_sentiment_distribution')


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIG 2 – PnL BY SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 2] PnL by sentiment...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
fig.suptitle('Trader PnL Performance Across Market Sentiment', fontsize=17, fontweight='bold', color=TEXT)

sent_pnl = closed.groupby('sentiment', observed=True)['Closed PnL'].agg(['mean','median','sum']).reindex(SENTIMENT_ORDER).dropna()
colors_ord = [PALETTE[s] for s in sent_pnl.index]

# Mean PnL
bars = axes[0].bar(sent_pnl.index, sent_pnl['mean'], color=colors_ord, edgecolor=BG)
axes[0].set_title('Avg PnL per Closed Trade', color=TEXT, fontsize=12)
axes[0].set_ylabel('USD', fontsize=11)
axes[0].axhline(0, color=TEXT, linewidth=0.8, linestyle='--', alpha=0.6)
axes[0].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, sent_pnl['mean']):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+(2 if val>=0 else -6),
                 f'${val:.1f}', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')

# Median PnL
axes[1].bar(sent_pnl.index, sent_pnl['median'], color=colors_ord, edgecolor=BG)
axes[1].set_title('Median PnL per Closed Trade', color=TEXT, fontsize=12)
axes[1].set_ylabel('USD', fontsize=11)
axes[1].axhline(0, color=TEXT, linewidth=0.8, linestyle='--', alpha=0.6)
axes[1].tick_params(axis='x', rotation=20)

# Total PnL
axes[2].bar(sent_pnl.index, sent_pnl['sum']/1e6, color=colors_ord, edgecolor=BG)
axes[2].set_title('Total Cumulative PnL', color=TEXT, fontsize=12)
axes[2].set_ylabel('Million USD', fontsize=11)
axes[2].axhline(0, color=TEXT, linewidth=0.8, linestyle='--', alpha=0.6)
axes[2].tick_params(axis='x', rotation=20)

savefig('fig2_pnl_by_sentiment')


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIG 3 – WIN RATE BY SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 3] Win rate...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle('Win Rate & Trade Volume by Sentiment', fontsize=17, fontweight='bold', color=TEXT)

wr = closed.groupby('sentiment', observed=True)['profit'].agg(['mean','count']).reindex(SENTIMENT_ORDER).dropna()
wr['win_rate'] = wr['mean'] * 100
colors_ord = [PALETTE[s] for s in wr.index]

bars = axes[0].bar(wr.index, wr['win_rate'], color=colors_ord, edgecolor=BG)
axes[0].axhline(50, color=YELLOW, linewidth=1.5, linestyle='--', label='50% baseline')
axes[0].set_title('Win Rate (%) by Sentiment', color=TEXT, fontsize=13)
axes[0].set_ylabel('Win Rate %', fontsize=11)
axes[0].set_ylim(0, 80)
axes[0].legend(fontsize=10)
axes[0].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, wr['win_rate']):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.1f}%', ha='center', va='bottom', color=TEXT, fontsize=10, fontweight='bold')

bars2 = axes[1].bar(wr.index, wr['count'], color=colors_ord, edgecolor=BG)
axes[1].set_title('# Closed Trades by Sentiment', color=TEXT, fontsize=13)
axes[1].set_ylabel('Number of Trades', fontsize=11)
axes[1].tick_params(axis='x', rotation=15)
for bar, val in zip(bars2, wr['count']):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                 f'{int(val):,}', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')

savefig('fig3_winrate_by_sentiment')


# ══════════════════════════════════════════════════════════════════════════════
# 6. FIG 4 – TOP COINS BY SENTIMENT HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 4] Heatmap coin × sentiment...")
top_coins = closed['Coin'].value_counts().head(10).index.tolist()
heat_data = closed[closed['Coin'].isin(top_coins)].groupby(
    ['Coin','sentiment'], observed=True)['Closed PnL'].mean().unstack(fill_value=0)
heat_data = heat_data.reindex(columns=SENTIMENT_ORDER, fill_value=0)

fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
sns.heatmap(heat_data, ax=ax, cmap='RdYlGn', center=0, annot=True, fmt='.0f',
            linewidths=0.5, linecolor=BG,
            cbar_kws={'label': 'Avg PnL (USD)'})
ax.set_title('Average PnL per Trade: Top Coins × Market Sentiment', color=TEXT, fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Market Sentiment', fontsize=12)
ax.set_ylabel('Coin', fontsize=12)
ax.tick_params(axis='x', rotation=15)
savefig('fig4_heatmap_coin_sentiment')


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIG 5 – LONG vs SHORT PERFORMANCE BY SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 5] Long vs Short...")
closed['trade_type'] = np.where(closed['Direction']=='Close Long','Long','Short')
ls = closed.groupby(['sentiment','trade_type'], observed=True)['Closed PnL'].mean().unstack(fill_value=0)
ls = ls.reindex(SENTIMENT_ORDER).dropna()

x = np.arange(len(ls))
width = 0.35
fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
bars1 = ax.bar(x - width/2, ls.get('Long', 0), width, label='Long', color=GREEN, edgecolor=BG, alpha=0.9)
bars2 = ax.bar(x + width/2, ls.get('Short', 0), width, label='Short', color=RED, edgecolor=BG, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(ls.index, rotation=15)
ax.set_title('Average PnL: Long vs Short Trades by Sentiment', color=TEXT, fontsize=14, fontweight='bold')
ax.set_ylabel('Avg Closed PnL (USD)', fontsize=12)
ax.axhline(0, color=TEXT, linewidth=0.8, linestyle='--', alpha=0.5)
ax.legend(fontsize=11)
ax.grid(axis='y')
savefig('fig5_long_short_sentiment')


# ══════════════════════════════════════════════════════════════════════════════
# 8. FIG 6 – TIME SERIES: FG INDEX + CUMULATIVE PnL
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 6] Time series...")
daily_pnl = closed.groupby('date')['Closed PnL'].sum().reset_index()
daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
daily_pnl = daily_pnl.sort_values('date')
daily_pnl['cum_pnl'] = daily_pnl['Closed PnL'].cumsum()

fg_plot = fg.copy()
fg_plot['date'] = pd.to_datetime(fg_plot['date'])
fg_plot = fg_plot[(fg_plot['date'] >= daily_pnl['date'].min()) &
                  (fg_plot['date'] <= daily_pnl['date'].max())]

fig, ax1 = plt.subplots(figsize=(16, 6), facecolor=BG)
ax2 = ax1.twinx()

# FG background shading
for _, row in fg_plot.iterrows():
    ax1.axvspan(row['date'], row['date'] + pd.Timedelta(days=1),
                color=PALETTE.get(row['classification'], '#888'), alpha=0.12)

ax2.plot(daily_pnl['date'], daily_pnl['cum_pnl']/1e6, color=ACCENT, linewidth=2, label='Cumulative PnL')
ax2.set_ylabel('Cumulative PnL (M USD)', color=ACCENT, fontsize=11)
ax2.tick_params(axis='y', colors=ACCENT)

ax1.plot(fg_plot['date'], fg_plot['value'], color=YELLOW, linewidth=1.5, alpha=0.8, label='Fear/Greed Index')
ax1.set_ylabel('Fear & Greed Index (0–100)', color=YELLOW, fontsize=11)
ax1.tick_params(axis='y', colors=YELLOW)
ax1.set_xlabel('Date', fontsize=11)
ax1.set_title('Cumulative PnL vs Bitcoin Fear & Greed Index Over Time', color=TEXT, fontsize=14, fontweight='bold')

patches = [mpatches.Patch(color=PALETTE[s], label=s, alpha=0.6) for s in SENTIMENT_ORDER]
ax1.legend(handles=patches, loc='upper left', fontsize=9, framealpha=0.3)
ax2.legend(loc='lower right', fontsize=10, framealpha=0.3)
ax1.grid(axis='x', alpha=0.3)
savefig('fig6_timeseries')


# ══════════════════════════════════════════════════════════════════════════════
# 9. FIG 7 – TRADER CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 7] Trader clustering...")
trader_feat = closed.groupby('Account').agg(
    total_pnl=('Closed PnL','sum'),
    avg_pnl=('Closed PnL','mean'),
    win_rate=('profit','mean'),
    num_trades=('Closed PnL','count'),
    avg_size=('Size USD','mean'),
    pnl_std=('Closed PnL','std'),
).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(trader_feat)

# PCA for viz
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
trader_feat['cluster'] = kmeans.fit_predict(X_scaled)

cluster_names = {}
for c in range(4):
    sub = trader_feat[trader_feat['cluster']==c]
    if sub['total_pnl'].mean() > 5000 and sub['win_rate'].mean() > 0.5:
        cluster_names[c] = '🏆 Elite Performers'
    elif sub['total_pnl'].mean() < -1000:
        cluster_names[c] = '📉 Struggling Traders'
    elif sub['num_trades'].mean() > trader_feat['num_trades'].mean():
        cluster_names[c] = '⚡ High Frequency'
    else:
        cluster_names[c] = '🔵 Moderate Traders'

cluster_colors = ['#58a6ff','#3fb950','#f85149','#d29922']
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig.suptitle('Trader Segmentation via K-Means Clustering', fontsize=16, fontweight='bold', color=TEXT)

for c in range(4):
    mask = trader_feat['cluster'].values == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=cluster_colors[c], label=cluster_names[c], s=120, edgecolors=BG, linewidth=1.5, alpha=0.9)
axes[0].set_title('PCA: Trader Clusters', color=TEXT, fontsize=13)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=10)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=10)
axes[0].legend(fontsize=9, framealpha=0.3)

# Cluster stats bar
cluster_stats = trader_feat.groupby('cluster')[['win_rate','avg_pnl']].mean()
cluster_stats.index = [cluster_names[c] for c in cluster_stats.index]
x = np.arange(len(cluster_stats))
axes[1].bar(x - 0.2, cluster_stats['win_rate']*100, 0.4, label='Win Rate %', color=GREEN, edgecolor=BG)
ax_r = axes[1].twinx()
ax_r.bar(x + 0.2, cluster_stats['avg_pnl'], 0.4, label='Avg PnL $', color=ACCENT, edgecolor=BG)
axes[1].set_xticks(x)
axes[1].set_xticklabels(cluster_stats.index, rotation=12, fontsize=9)
axes[1].set_ylabel('Win Rate %', color=GREEN, fontsize=11)
ax_r.set_ylabel('Avg PnL (USD)', color=ACCENT, fontsize=11)
axes[1].set_title('Cluster Performance Metrics', color=TEXT, fontsize=13)
axes[1].legend(loc='upper left', fontsize=9, framealpha=0.3)
ax_r.legend(loc='upper right', fontsize=9, framealpha=0.3)
savefig('fig7_clustering')


# ══════════════════════════════════════════════════════════════════════════════
# 10. FIG 8 – ML: PREDICT PROFITABLE TRADE
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 8] ML model...")
ml = closed.copy()
ml['sentiment_enc'] = LabelEncoder().fit_transform(ml['sentiment'].astype(str))
ml['trade_type_enc'] = (ml['trade_type'] == 'Long').astype(int)
ml['hour'] = ml['datetime'].dt.hour
ml['day_of_week'] = ml['datetime'].dt.dayofweek

features = ['Size USD','fg_value','sentiment_enc','trade_type_enc','hour','day_of_week']
ml_clean = ml[features + ['profit']].dropna()
X = ml_clean[features]
y = ml_clean['profit'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s  = sc.transform(X_test)

rf  = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
gb  = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
lr  = LogisticRegression(max_iter=500, random_state=42)

results = {}
for name, model, Xtr, Xte in [('Random Forest', rf, X_train, X_test),
                                ('Gradient Boost', gb, X_train, X_test),
                                ('Logistic Reg', lr, X_train_s, X_test_s)]:
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    prob  = model.predict_proba(Xte)[:,1]
    results[name] = {
        'acc': (preds==y_test).mean(),
        'auc': roc_auc_score(y_test, prob),
        'cv':  cross_val_score(model, Xtr, y_train, cv=5, scoring='roc_auc').mean()
    }
    print(f"  {name}: Acc={results[name]['acc']:.3f}, AUC={results[name]['auc']:.3f}")

# Feature importance from RF
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)
fig.suptitle('ML Model: Predicting Profitable Trades', fontsize=16, fontweight='bold', color=TEXT)

# Model comparison
names_  = list(results.keys())
accs    = [results[n]['acc']*100 for n in names_]
aucs    = [results[n]['auc']*100 for n in names_]
x_ = np.arange(len(names_))
axes[0].bar(x_ - 0.2, accs, 0.4, label='Accuracy %', color=ACCENT, edgecolor=BG)
axes[0].bar(x_ + 0.2, aucs, 0.4, label='ROC-AUC %',  color=GREEN, edgecolor=BG)
axes[0].set_xticks(x_)
axes[0].set_xticklabels(names_, fontsize=10)
axes[0].set_ylabel('%', fontsize=11)
axes[0].set_ylim(40, 100)
axes[0].set_title('Model Comparison', color=TEXT, fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(axis='y')
for i, (a, u) in enumerate(zip(accs, aucs)):
    axes[0].text(i-0.2, a+0.5, f'{a:.1f}', ha='center', fontsize=9, color=TEXT)
    axes[0].text(i+0.2, u+0.5, f'{u:.1f}', ha='center', fontsize=9, color=TEXT)

# Feature importance
colors_fi = [GREEN if v > fi.mean() else ACCENT for v in fi.values]
axes[1].barh(fi.index, fi.values, color=colors_fi, edgecolor=BG)
axes[1].set_title('Feature Importance (Random Forest)', color=TEXT, fontsize=13)
axes[1].set_xlabel('Importance Score', fontsize=11)
axes[1].grid(axis='x')
savefig('fig8_ml_model')


# ══════════════════════════════════════════════════════════════════════════════
# 11. FIG 9 – PnL DISTRIBUTION VIOLIN
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig 9] PnL violin plot...")
fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
clip_pnl = closed[closed['Closed PnL'].between(-2000, 2000)].copy()

parts = ax.violinplot(
    [clip_pnl[clip_pnl['sentiment']==s]['Closed PnL'].dropna().values
     for s in SENTIMENT_ORDER if s in clip_pnl['sentiment'].values],
    positions=range(len(SENTIMENT_ORDER)),
    showmedians=True, showextrema=True
)
for i, (pc, s) in enumerate(zip(parts['bodies'], SENTIMENT_ORDER)):
    pc.set_facecolor(PALETTE[s])
    pc.set_alpha(0.75)
parts['cmedians'].set_color(TEXT)
parts['cmaxes'].set_color(TEXT)
parts['cmins'].set_color(TEXT)
parts['cbars'].set_color(TEXT)

ax.set_xticks(range(len(SENTIMENT_ORDER)))
ax.set_xticklabels(SENTIMENT_ORDER, fontsize=11)
ax.axhline(0, color=YELLOW, linewidth=1.2, linestyle='--', alpha=0.7, label='Break-even')
ax.set_title('PnL Distribution per Trade by Sentiment (clipped ±$2000)', color=TEXT, fontsize=14, fontweight='bold')
ax.set_ylabel('Closed PnL (USD)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(axis='y')
savefig('fig9_violin_pnl')


# ══════════════════════════════════════════════════════════════════════════════
# 12. SUMMARY STATS for report
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("KEY STATS SUMMARY")
print("="*60)
print(f"Total trades analysed : {len(df):,}")
print(f"Closed trades         : {len(closed):,}")
print(f"Unique traders        : {df['Account'].nunique()}")
print(f"Date range            : {df['date'].min()} → {df['date'].max()}")
print(f"\nOverall win rate      : {closed['profit'].mean()*100:.1f}%")
print(f"Overall avg PnL/trade : ${closed['Closed PnL'].mean():.2f}")
print(f"Total cumulative PnL  : ${closed['Closed PnL'].sum():,.0f}")
print("\n--- Win Rate by Sentiment ---")
print(closed.groupby('sentiment', observed=True)['profit'].mean().mul(100).round(1).to_string())
print("\n--- Avg PnL by Sentiment ---")
print(closed.groupby('sentiment', observed=True)['Closed PnL'].mean().round(2).to_string())
print("\n--- Best ML Model ---")
best = max(results, key=lambda k: results[k]['auc'])
print(f"{best}: AUC={results[best]['auc']:.3f}, Acc={results[best]['acc']:.3f}")
print("\nAll figures saved! ✅")
