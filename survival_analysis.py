"""
Homework 3 – Survival Analysis  |  Karen Hovhannisyan
"""
import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter, GeneralizedGammaRegressionFitter
from lifelines.utils import concordance_index
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

# 1. Load & prepare
df = pd.read_csv('/mnt/user-data/uploads/telco.csv')
df['event'] = (df['churn'] == 'Yes').astype(int)
cat_cols = ['region','marital','ed','retire','gender','voice','internet','forward','custcat']
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df_enc.columns = df_enc.columns.str.replace(' ','_').str.replace('/','_').str.replace('-','_')
feature_cols = [c for c in df_enc.columns if c not in ['ID','churn','event','tenure']]
fit_df = df_enc[['tenure','event']+feature_cols].dropna()
print(f"Dataset: {df.shape[0]} subscribers | Churn rate: {df['event'].mean():.1%}")

# 2. Fit all AFT models
gg = GeneralizedGammaRegressionFitter(penalizer=1.0)
gg._scipy_fit_method = 'SLSQP'
MODELS = {'Weibull': WeibullAFTFitter(penalizer=0.01),
          'Log-Normal': LogNormalAFTFitter(penalizer=0.01),
          'Log-Logistic': LogLogisticAFTFitter(penalizer=0.01),
          'Gen. Gamma': gg}

results = {}
for name, m in MODELS.items():
    m.fit(fit_df, duration_col='tenure', event_col='event')
    pred = m.predict_median(fit_df)
    ci = concordance_index(fit_df['tenure'], pred, fit_df['event'])
    results[name] = dict(model=m, AIC=m.AIC_, log_lik=m.log_likelihood_, C=ci)
    print(f"  {name:<15} AIC={m.AIC_:9.2f}  LL={m.log_likelihood_:9.2f}  C={ci:.4f}")

comp = pd.DataFrame({k:{kk:vv for kk,vv in v.items() if kk!='model'} for k,v in results.items()}).T.astype(float).sort_values('AIC')
print("\nModel comparison:\n", comp.round(2).to_string())
best_name = comp.index[0]
print(f"\n-> Best model by AIC: {best_name}")

# 3. Plot all curves
median_row = fit_df[feature_cols].median().to_frame().T.reset_index(drop=True)
t_range = np.linspace(0, df['tenure'].max(), 300)
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ['#2196F3','#E91E63','#4CAF50','#FF9800']
for (name, res), color in zip(results.items(), colors):
    sf = res['model'].predict_survival_function(median_row, times=t_range)
    ax.plot(t_range, sf.values.flatten(), label=f"{name} (AIC={res['AIC']:.0f})", color=color, lw=2.2)
ax.set_xlabel('Tenure (months)', fontsize=12); ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('AFT Models – Survival Curves (median customer)', fontsize=13)
ax.legend(fontsize=10); ax.set_ylim(0,1); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('/mnt/user-data/outputs/aft_model_comparison.png', dpi=150); plt.close()
print("Saved: aft_model_comparison.png")

# 4. Significant features
best_m = results[best_name]['model']
summary = best_m.summary
if hasattr(summary.index, 'levels'):
    lvl0 = summary.index.get_level_values(0)
    sub = summary[lvl0 == 'lambda_']
    if len(sub)==0: sub = summary[lvl0 == 'mu_']
    if len(sub)==0: sub = summary
else:
    sub = summary
sig_features = [f for f in sub.index.get_level_values(-1)[sub['p']<0.05] if f != 'Intercept']
print(f"\nSignificant features (p<0.05): {len(sig_features)}")
for f in sig_features:
    row = sub.loc[sub.index.get_level_values(-1)==f]
    print(f"  {f:<45} coef={row['coef'].values[0]:+.3f}")

final_fit_df = df_enc[['tenure','event']+sig_features].dropna() if sig_features else fit_df
if not sig_features: sig_features = feature_cols
FinalClass = type(best_m)
final_model = FinalClass(penalizer=0.01)
if best_name == 'Gen. Gamma': final_model._scipy_fit_method = 'SLSQP'
final_model.fit(final_fit_df, duration_col='tenure', event_col='event')
pred_f = final_model.predict_median(final_fit_df)
ci_f = concordance_index(final_fit_df['tenure'], pred_f, final_fit_df['event'])
print(f"\nFinal model AIC={final_model.AIC_:.2f}  C-index={ci_f:.4f}")
print(final_model.summary[['coef','exp(coef)','p']].to_string())

# 5. CLV
DISCOUNT=0.10; MARGIN=0.20; HORIZON=60
months=np.arange(1,HORIZON+1); discount=(1+DISCOUNT/12)**(months-1)
clv_df = df_enc[['ID','income']+sig_features].copy()
clv_df['tenure']=df_enc['tenure']; clv_df['event']=df_enc['event']
clv_df=clv_df.dropna().reset_index(drop=True)
sf_matrix = final_model.predict_survival_function(clv_df[sig_features], times=months)
clv_df['monthly_margin'] = (clv_df['income']*1000/12)*MARGIN
clv_df['CLV'] = clv_df['monthly_margin'].values * (sf_matrix.values / discount[:,None]).sum(axis=0)
print(f"\nCLV summary:\n{clv_df['CLV'].describe().round(0).to_string()}")

# 6. Segments
clv_df['churn']=df.loc[clv_df.index,'churn'].values
clv_df['custcat']=df.loc[clv_df.index,'custcat'].values
clv_df['gender']=df.loc[clv_df.index,'gender'].values
clv_df['region']=df.loc[clv_df.index,'region'].values
clv_df['age_group']=pd.cut(df.loc[clv_df.index,'age'],bins=[0,30,45,65,100],labels=['<30','30-45','45-65','65+'])

fig, axes = plt.subplots(2,2,figsize=(13,9))
axes=axes.flatten()
palette=['#2196F3','#E91E63','#4CAF50','#FF9800','#9C27B0','#00BCD4']
for ax,col in zip(axes,['custcat','gender','age_group','region']):
    groups=clv_df.groupby(col)['CLV'].mean().sort_values(ascending=False)
    bars=ax.bar(range(len(groups)),groups.values,color=palette[:len(groups)],edgecolor='white')
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups.index,fontsize=9,rotation=20,ha='right')
    ax.set_title(f'Mean CLV by {col}',fontsize=11); ax.set_ylabel('CLV ($)',fontsize=9); ax.grid(axis='y',alpha=0.3)
    for bar,val in zip(bars,groups.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500, f'${val:,.0f}',ha='center',fontsize=8)
plt.suptitle('Customer Lifetime Value by Segment',fontsize=14); plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/clv_by_segment.png',dpi=150,bbox_inches='tight'); plt.close()
print("Saved: clv_by_segment.png")

# 7. Retention budget
sf_12=sf_matrix.loc[12].values; clv_df['surv_12m']=sf_12
at_risk=clv_df[clv_df['surv_12m']<0.5]
print(f"\nAt-risk (S(12m)<0.5): {len(at_risk)}")
print(f"Total CLV at-risk: ${at_risk['CLV'].sum():,.0f}")
print(f"Suggested budget (30%): ${at_risk['CLV'].sum()*0.30:,.0f}")
top=clv_df.groupby('custcat')['CLV'].mean().sort_values(ascending=False)
print(f"\nTop segment: {top.index[0]} (${top.iloc[0]:,.0f})")
print(f"Bottom segment: {top.index[-1]} (${top.iloc[-1]:,.0f})")
