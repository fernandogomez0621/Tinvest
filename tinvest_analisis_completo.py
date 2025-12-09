# -*- coding: utf-8 -*-
"""
EDA Completo - Fintech Tinvest
Análisis Exploratorio, Cohortes, RFM, Visualizaciones y Modelos Predictivos
"""

# %% [markdown]
# # Análisis Completo - Tinvest
# 
# Este notebook contiene el análisis completo de datos de Tinvest incluyendo:
# - Análisis Exploratorio de Datos (EDA)
# - Análisis de Cohortes y Retención
# - Segmentación RFM
# - Visualizaciones Ejecutivas
# - Modelos Predictivos de Churn
# - Forecasting con Prophet

# %% Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, f1_score, accuracy_score)

# Prophet
try:
    from prophet import Prophet
    PROPHET_DISPONIBLE = True
except ImportError:
    PROPHET_DISPONIBLE = False
    print("⚠️ Prophet no instalado. Ejecuta: pip install prophet")

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')

# %% [markdown]
# ## 1. Carga y Validación de Datos

# %% Carga de datos
clients = pd.read_csv('clients.csv')
transactions = pd.read_csv('transactions.csv')
portfolio = pd.read_csv('portfolio_balance.csv')

print(f"✓ Clientes: {len(clients):,} registros")
print(f"✓ Transacciones: {len(transactions):,} registros")
print(f"✓ Portfolio: {len(portfolio):,} registros")

# %% Conversión de fechas
clients['registration_date'] = pd.to_datetime(clients['registration_date'])
transactions['date'] = pd.to_datetime(transactions['date'])
portfolio['date'] = pd.to_datetime(portfolio['date'])

print(f"\nRango temporal clientes: {clients['registration_date'].min()} a {clients['registration_date'].max()}")
print(f"Rango temporal transacciones: {transactions['date'].min()} a {transactions['date'].max()}")
print(f"Rango temporal portfolio: {portfolio['date'].min()} a {portfolio['date'].max()}")

# %% [markdown]
# ## 2. Análisis Exploratorio de Datos (EDA)

# %% Calidad de datos
print("=== CALIDAD DE DATOS ===\n")

for nombre, df in [("CLIENTS", clients), ("TRANSACTIONS", transactions), ("PORTFOLIO", portfolio)]:
    print(f"{nombre}:")
    print(f"  Registros: {len(df):,}")
    print(f"  Duplicados: {df.duplicated().sum():,}")
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        print(f"  Nulos: {nulos[nulos > 0].to_dict()}")
    else:
        print("  Nulos: 0")
    print()

# %% Estadísticas descriptivas - Clientes
print("=== ESTADÍSTICAS - CLIENTES ===\n")
print("Variables Numéricas:")
print(clients[['age', 'income_monthly', 'risk_score']].describe())

print("\nDistribución por Segmento:")
print(clients['segment'].value_counts())
print("\nProporción (%):")
print((clients['segment'].value_counts(normalize=True) * 100).round(1))

# %% Estadísticas descriptivas - Transacciones
print("\n=== ESTADÍSTICAS - TRANSACCIONES ===\n")
print("Distribución por Tipo:")
print(transactions['type'].value_counts())

print("\nDistribución por Producto:")
print(transactions['product'].value_counts())

print("\nEstadísticas por Tipo:")
print(transactions.groupby('type')['amount'].describe())

total_deposits = transactions[transactions['type'] == 'deposit']['amount'].sum()
total_withdrawals = transactions[transactions['type'] == 'withdrawal']['amount'].sum()
net_flow = total_deposits - total_withdrawals

print(f"\n💰 FLUJO DE DINERO:")
print(f"Total Depósitos: ${total_deposits:,.0f}")
print(f"Total Retiros: ${total_withdrawals:,.0f}")
print(f"NNM (Net New Money): ${net_flow:,.0f}")

# %% [markdown]
# ## 3. Análisis de Cohortes y Retención

# %% Crear cohortes
fecha_corte = transactions['date'].max()
clients['cohort'] = clients['registration_date'].dt.to_period('M')

print(f"=== ANÁLISIS DE COHORTES ===")
print(f"Fecha de corte: {fecha_corte}\n")

cohort_sizes = clients.groupby('cohort').size()
print(f"Total de cohortes: {len(cohort_sizes)}")
print(f"Rango: {cohort_sizes.index.min()} a {cohort_sizes.index.max()}")

# %% Análisis de retención
trans_with_cohort = transactions.merge(
    clients[['client_id', 'cohort', 'registration_date']],
    on='client_id'
)

trans_with_cohort['periodo'] = (
    (trans_with_cohort['date'].dt.to_period('M') - trans_with_cohort['cohort'])
    .apply(lambda x: x.n)
)

retention_table = trans_with_cohort.groupby(['cohort', 'periodo'])['client_id'].nunique().reset_index()
retention_table.columns = ['cohort', 'periodo', 'clientes_activos']

retention_matrix = retention_table.pivot(index='cohort', columns='periodo', values='clientes_activos')
cohort_sizes_dict = cohort_sizes.to_dict()
retention_rate = retention_matrix.divide(retention_matrix.index.map(cohort_sizes_dict), axis=0) * 100

print("\n📈 MATRIZ DE RETENCIÓN (%) - Primeros 6 meses:")
print(retention_rate.iloc[:, :7].round(1))

print("\n📊 ESTADÍSTICAS DE RETENCIÓN:")
print(f"Retención Mes 1: {retention_rate[1].mean():.1f}%")
print(f"Retención Mes 3: {retention_rate[3].mean():.1f}%")
if 6 in retention_rate.columns:
    print(f"Retención Mes 6: {retention_rate[6].mean():.1f}%")

# %% Análisis de churn
ultima_trans = transactions.groupby('client_id')['date'].max().reset_index()
ultima_trans.columns = ['client_id', 'ultima_transaccion']

churn_analysis = clients.merge(ultima_trans, on='client_id', how='left')
churn_analysis['dias_activo'] = (
    churn_analysis['ultima_transaccion'] - churn_analysis['registration_date']
).dt.days
churn_analysis['dias_inactivo'] = (fecha_corte - churn_analysis['ultima_transaccion']).dt.days

balance_actual = portfolio[portfolio['date'] == portfolio['date'].max()].groupby('client_id')['balance'].sum().reset_index()
balance_actual.columns = ['client_id', 'balance_actual']
churn_analysis = churn_analysis.merge(balance_actual, on='client_id', how='left')
churn_analysis['balance_actual'] = churn_analysis['balance_actual'].fillna(0)

churn_analysis['churned'] = (
    (churn_analysis['balance_actual'] == 0) &
    (churn_analysis['dias_inactivo'] > 90)
).astype(int)

print(f"\n=== ANÁLISIS DE CHURN ===")
print(f"Clientes CHURNED: {churn_analysis['churned'].sum()} ({churn_analysis['churned'].mean()*100:.1f}%)")
print(f"Clientes ACTIVOS: {(1-churn_analysis['churned']).sum()} ({(1-churn_analysis['churned']).mean()*100:.1f}%)")

# %% [markdown]
# ## 4. Segmentación RFM

# %% Calcular métricas RFM
recency = transactions.groupby('client_id')['date'].max().reset_index()
recency.columns = ['client_id', 'ultima_fecha']
recency['recency'] = (fecha_corte - recency['ultima_fecha']).dt.days

frequency = transactions.groupby('client_id').size().reset_index()
frequency.columns = ['client_id', 'frequency']

monetary = transactions[transactions['type'] == 'deposit'].groupby('client_id')['amount'].sum().reset_index()
monetary.columns = ['client_id', 'monetary']

rfm = recency[['client_id', 'recency']].merge(frequency, on='client_id', how='left')
rfm = rfm.merge(monetary, on='client_id', how='left')
rfm = rfm.merge(clients[['client_id', 'segment', 'registration_date']], on='client_id', how='left')
rfm['monetary'] = rfm['monetary'].fillna(0)

print("=== MÉTRICAS RFM ===\n")
print(rfm[['recency', 'frequency', 'monetary']].describe())

# %% Crear scores RFM
rfm['R_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)

rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# %% Segmentar clientes RFM
def segmentar_rfm(df):
    campeones = (df['R_score'] == 5) & (df['F_score'] >= 4) & (df['M_score'] >= 4)
    leales = (df['R_score'] >= 4) & (df['F_score'] >= 4)
    potenciales = (df['R_score'] >= 4) & (df['F_score'] <= 2) & (df['M_score'] >= 4)
    atencion = (df['R_score'] == 3) & (df['F_score'] == 3)
    riesgo = (df['R_score'] <= 2) & (df['F_score'] >= 4)
    hibernando = (df['R_score'] <= 2) & (df['F_score'] <= 2) & (df['M_score'] >= 4)
    perdidos = (df['R_score'] <= 2) & (df['F_score'] <= 2) & (df['M_score'] <= 2)
    
    dias_registro = (fecha_corte - df['registration_date']).dt.days
    nuevos = (dias_registro <= 90) & (df['F_score'] <= 2)
    
    segmento = []
    for i in range(len(df)):
        if campeones.iloc[i] and not nuevos.iloc[i]:
            segmento.append('Campeones')
        elif leales.iloc[i] and not nuevos.iloc[i] and not campeones.iloc[i]:
            segmento.append('Leales')
        elif potenciales.iloc[i] and not nuevos.iloc[i]:
            segmento.append('Potenciales')
        elif riesgo.iloc[i]:
            segmento.append('En Riesgo')
        elif hibernando.iloc[i]:
            segmento.append('Hibernando')
        elif perdidos.iloc[i]:
            segmento.append('Perdidos')
        elif nuevos.iloc[i]:
            segmento.append('Nuevos')
        elif atencion.iloc[i]:
            segmento.append('Necesitan Atención')
        else:
            segmento.append('Promedio')
    
    return segmento

rfm['segmento_rfm'] = segmentar_rfm(rfm)

print("\n📊 DISTRIBUCIÓN DE SEGMENTOS RFM:")
print(rfm['segmento_rfm'].value_counts())
print("\nProporción (%):")
print((rfm['segmento_rfm'].value_counts() / len(rfm) * 100).round(1))

# %% Guardar análisis
rfm.to_csv('rfm_segmentation.csv', index=False)
churn_analysis.to_csv('churn_analysis.csv', index=False)
retention_rate.to_csv('retention_matrix.csv')
print("\n✓ Archivos guardados: rfm_segmentation.csv, churn_analysis.csv, retention_matrix.csv")

# %% [markdown]
# ## 5. Visualizaciones Ejecutivas

# %% Configuración de visualizaciones
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (14, 8)

# %% Figura 1: Evolución de NNM
print("\n📊 Generando visualizaciones...")

fig, ax = plt.subplots(figsize=(14, 6))

transactions['year_month'] = transactions['date'].dt.to_period('M').dt.to_timestamp()
flujo_mensual = transactions.groupby(['year_month', 'type'])['amount'].sum().unstack(fill_value=0)
flujo_mensual['net_flow'] = flujo_mensual.get('deposit', 0) - flujo_mensual.get('withdrawal', 0)

x = range(len(flujo_mensual))
ax.bar(x, flujo_mensual['deposit']/1e9, width=0.8, label='Depósitos', color='#2ecc71', alpha=0.8)
ax.bar(x, -flujo_mensual['withdrawal']/1e9, width=0.8, label='Retiros', color='#e74c3c', alpha=0.8)

ax2 = ax.twinx()
ax2.plot(x, flujo_mensual['net_flow']/1e9, color='#3498db', linewidth=3,
         marker='o', markersize=6, label='Flujo Neto (NNM)')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)

ax.set_xlabel('Mes', fontsize=12, fontweight='bold')
ax.set_ylabel('Flujo de Dinero (Miles de Millones $)', fontsize=12, fontweight='bold')
ax2.set_ylabel('NNM (Miles de Millones $)', fontsize=12, fontweight='bold')
ax.set_title('Evolución del Net New Money (NNM)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x[::2])
ax.set_xticklabels([d.strftime('%Y-%m') for d in flujo_mensual.index[::2]], rotation=45)
ax.grid(True, alpha=0.3)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('fig1_nnm_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

# %% Figura 2: Matriz de Retención
fig, ax = plt.subplots(figsize=(14, 10))

retention_clean = retention_rate.iloc[-12:, :7].copy()
retention_clean.index = [str(idx) for idx in retention_clean.index]

sns.heatmap(retention_clean, annot=True, fmt='.0f', cmap='RdYlGn',
            vmin=0, vmax=100, cbar_kws={'label': 'Tasa de Retención (%)'}, ax=ax,
            linewidths=0.5, linecolor='gray')

ax.set_title('Matriz de Retención por Cohorte', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Meses desde Registro', fontsize=12, fontweight='bold')
ax.set_ylabel('Cohorte (Mes de Registro)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fig2_retention_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# %% Figura 3: Distribución de Segmentos RFM
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

segmento_counts = rfm['segmento_rfm'].value_counts()
colors_segments = {
    'Campeones': '#2ecc71', 'Leales': '#27ae60', 'Potenciales': '#3498db',
    'Promedio': '#95a5a6', 'Necesitan Atención': '#f39c12', 'En Riesgo': '#e67e22',
    'Hibernando': '#9b59b6', 'Perdidos': '#e74c3c'
}
colors = [colors_segments.get(seg, '#95a5a6') for seg in segmento_counts.index]

wedges, texts, autotexts = ax1.pie(segmento_counts, labels=segmento_counts.index, autopct='%1.1f%%',
                                     colors=colors, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax1.set_title('Distribución de Clientes por Segmento RFM', fontsize=14, fontweight='bold')

valor_segmento = rfm.groupby('segmento_rfm')['monetary'].sum().sort_values(ascending=True) / 1e9
colors_valor = [colors_segments.get(seg, '#95a5a6') for seg in valor_segmento.index]

valor_segmento.plot(kind='barh', ax=ax2, color=colors_valor, alpha=0.8)
ax2.set_xlabel('Valor Total (Miles de Millones $)', fontsize=12, fontweight='bold')
ax2.set_title('Valor Total por Segmento RFM', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

for i, v in enumerate(valor_segmento):
    ax2.text(v + 0.1, i, f'${v:.1f}B', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('fig3_rfm_segments.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Visualizaciones guardadas: fig1_nnm_evolution.png, fig2_retention_matrix.png, fig3_rfm_segments.png")

# %% [markdown]
# ## 6. Modelos Predictivos de Churn

# %% Preparación de datos para modelado
print("\n=== MODELOS PREDICTIVOS DE CHURN ===\n")

model_data = clients[['client_id', 'age', 'income_monthly', 'segment', 'risk_score']].copy()

rfm_features = rfm[['client_id', 'recency', 'frequency', 'monetary',
                     'R_score', 'F_score', 'M_score', 'RFM_score']].copy()
model_data = model_data.merge(rfm_features, on='client_id', how='left')

model_data['dias_desde_registro'] = (fecha_corte - clients['registration_date']).dt.days

productos_por_cliente = transactions.groupby('client_id')['product'].nunique().reset_index()
productos_por_cliente.columns = ['client_id', 'num_productos']
model_data = model_data.merge(productos_por_cliente, on='client_id', how='left')

model_data = model_data.merge(balance_actual, on='client_id', how='left')
model_data['balance_actual'] = model_data['balance_actual'].fillna(0)

dep_ret = transactions.groupby(['client_id', 'type'])['amount'].sum().unstack(fill_value=0)
dep_ret['deposit_withdrawal_ratio'] = dep_ret['deposit'] / (dep_ret['withdrawal'] + 1)
model_data = model_data.merge(dep_ret[['deposit_withdrawal_ratio']],
                               left_on='client_id', right_index=True, how='left')

trans_std = transactions.groupby('client_id')['amount'].std().reset_index()
trans_std.columns = ['client_id', 'amount_std']
model_data = model_data.merge(trans_std, on='client_id', how='left')

trans_range = transactions.groupby('client_id')['date'].agg(['min', 'max']).reset_index()
trans_range['dias_actividad'] = (trans_range['max'] - trans_range['min']).dt.days
model_data = model_data.merge(trans_range[['client_id', 'dias_actividad']], on='client_id', how='left')

model_data = model_data.merge(churn_analysis[['client_id', 'churned']], on='client_id', how='left')
model_data = model_data.fillna(0)

print(f"Features creados: {len(model_data.columns)} columnas")
print(f"Churn Rate: {model_data['churned'].mean()*100:.1f}%")

# %% Train-test split
X = model_data.drop(['client_id', 'churned'], axis=1).copy()
y = model_data['churned']

le = LabelEncoder()
X['segment_encoded'] = le.fit_transform(X['segment'])
X = X.drop('segment', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f"\nTrain set: {len(X_train)} | Test set: {len(X_test)}")

# %% Modelo 1: Regresión Logística
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("\n📊 REGRESIÓN LOGÍSTICA:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_lr):.3f}")

# %% Modelo 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42,
                            class_weight='balanced', max_depth=10)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n📊 RANDOM FOREST:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.3f}")

feature_importance_rf = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance_rf.head(10))

# %% Modelo 3: Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42,
                                max_depth=5, learning_rate=0.1)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
y_proba_gb = gb.predict_proba(X_test)[:, 1]

print("\n📊 GRADIENT BOOSTING:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_gb):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_gb):.3f}")

# %% Comparación de modelos
results = pd.DataFrame({
    'Modelo': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_gb)],
    'ROC-AUC': [roc_auc_score(y_test, y_proba_lr), roc_auc_score(y_test, y_proba_rf), roc_auc_score(y_test, y_proba_gb)],
    'F1-Score': [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf), f1_score(y_test, y_pred_gb)]
})

print("\n📊 COMPARACIÓN DE MODELOS:")
print(results)

best_model_idx = results['ROC-AUC'].idxmax()
best_model_name = results.loc[best_model_idx, 'Modelo']
print(f"\n🏆 MEJOR MODELO: {best_model_name} (ROC-AUC: {results.loc[best_model_idx, 'ROC-AUC']:.3f})")

# %% Predicción de clientes en riesgo
X_all = model_data.drop(['client_id', 'churned'], axis=1).copy()
X_all['segment_encoded'] = le.transform(X_all['segment'])
X_all = X_all.drop('segment', axis=1)

churn_probability = rf.predict_proba(X_all)[:, 1]
model_data['churn_probability'] = churn_probability
model_data['churn_predicted'] = (churn_probability > 0.5).astype(int)

high_risk = model_data[model_data['churn_probability'] > 0.7].sort_values('churn_probability', ascending=False)

print(f"\n⚠️ Clientes de ALTO RIESGO (prob > 0.7): {len(high_risk)}")
print("\nTop 10 clientes en mayor riesgo:")
print(high_risk[['client_id', 'segment', 'churn_probability', 'churned', 'recency', 'frequency', 'monetary']].head(10))

predictions_output = model_data[['client_id', 'segment', 'churn_probability', 'churn_predicted', 'churned']].copy()
predictions_output.to_csv('churn_predictions.csv', index=False)
print("\n✓ Predicciones guardadas en: churn_predictions.csv")

# %% Visualización de modelos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_proba_gb)

ax1.plot(fpr_lr, tpr_lr, label=f'Logistic Reg (AUC={roc_auc_score(y_test, y_proba_lr):.3f})', linewidth=2)
ax1.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test, y_proba_rf):.3f})', linewidth=2)
ax1.plot(fpr_gb, tpr_gb, label=f'Gradient Boost (AUC={roc_auc_score(y_test, y_proba_gb):.3f})', linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
ax1.set_xlabel('False Positive Rate', fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontweight='bold')
ax1.set_title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

top_features = feature_importance_rf.head(15)
ax2.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.8)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'])
ax2.set_xlabel('Importancia', fontweight='bold')
ax2.set_title('Top 15 Features - Random Forest', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('fig_model_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Guardado: fig_model_performance.png")

# %% [markdown]
# ## 7. Forecasting con Prophet

# %% Funciones auxiliares
def detectar_outliers_iqr(series, factor=3.0):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    p99 = series.quantile(0.99)
    return series.clip(lower=lower, upper=min(upper, p99))

# %% Preparar datos NNM para Prophet
if PROPHET_DISPONIBLE:
    print("\n=== FORECASTING CON PROPHET ===\n")
    
    transactions['week'] = transactions['date'].dt.to_period('W').dt.to_timestamp()
    
    flujo_semanal = transactions.groupby(['week', 'type'])['amount'].sum().unstack(fill_value=0)
    flujo_semanal['net_flow'] = flujo_semanal.get('deposit', 0) - flujo_semanal.get('withdrawal', 0)
    flujo_semanal['net_flow'] = detectar_outliers_iqr(flujo_semanal['net_flow'])
    
    portfolio['week'] = portfolio['date'].dt.to_period('W').dt.to_timestamp()
    balance_semanal = portfolio.groupby('week')['balance'].mean().reset_index()
    balance_semanal.columns = ['week', 'balance_promedio']
    
    productos_semana = transactions.groupby('week')['product'].nunique().reset_index()
    productos_semana.columns = ['week', 'productos_activos']
    
    clientes_activos = portfolio.groupby('week').apply(
        lambda x: (x['balance'] == 0).sum() / len(x) if len(x) > 0 else 0
    ).reset_index()
    clientes_activos.columns = ['week', 'churn_rate']
    
    df_prophet_nnm = flujo_semanal.reset_index()
    df_prophet_nnm = df_prophet_nnm.rename(columns={'week': 'ds', 'net_flow': 'y'})
    df_prophet_nnm = df_prophet_nnm.merge(balance_semanal, left_on='ds', right_on='week', how='left')
    df_prophet_nnm = df_prophet_nnm.merge(productos_semana, left_on='ds', right_on='week', how='left')
    df_prophet_nnm = df_prophet_nnm.merge(clientes_activos, left_on='ds', right_on='week', how='left')
    df_prophet_nnm = df_prophet_nnm.drop(columns=[c for c in df_prophet_nnm.columns if c.startswith('week')], errors='ignore')
    
    for col in ['balance_promedio', 'productos_activos', 'churn_rate']:
        df_prophet_nnm[col] = df_prophet_nnm[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    balance_std = df_prophet_nnm['balance_promedio'].std()
    productos_std = df_prophet_nnm['productos_activos'].std()
    churn_std = df_prophet_nnm['churn_rate'].std()
    
    df_prophet_nnm['balance_promedio_norm'] = (
        (df_prophet_nnm['balance_promedio'] - df_prophet_nnm['balance_promedio'].mean()) /
        (balance_std if balance_std > 0 else 1)
    )
    df_prophet_nnm['productos_activos_norm'] = (
        (df_prophet_nnm['productos_activos'] - df_prophet_nnm['productos_activos'].mean()) /
        (productos_std if productos_std > 0 else 1)
    )
    df_prophet_nnm['churn_rate_norm'] = (
        (df_prophet_nnm['churn_rate'] - df_prophet_nnm['churn_rate'].mean()) /
        (churn_std if churn_std > 0 else 1)
    )
    
    df_prophet_nnm = df_prophet_nnm.fillna(0)
    
    print(f"Datos NNM preparados: {len(df_prophet_nnm)} semanas")
    print(f"Rango: {df_prophet_nnm['ds'].min()} a {df_prophet_nnm['ds'].max()}")

# %% Modelo Prophet NNM
if PROPHET_DISPONIBLE:
    print("\n🚀 Entrenando modelo Prophet NNM...")
    
    m_nnm = Prophet(
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95
    )
    
    m_nnm.add_regressor('balance_promedio_norm')
    m_nnm.add_regressor('productos_activos_norm')
    m_nnm.add_regressor('churn_rate_norm')
    
    m_nnm.fit(df_prophet_nnm)
    
    future_nnm = m_nnm.make_future_dataframe(periods=52, freq='W')
    ultimas_semanas = df_prophet_nnm.tail(4)
    future_nnm['balance_promedio_norm'] = ultimas_semanas['balance_promedio_norm'].mean()
    future_nnm['productos_activos_norm'] = ultimas_semanas['productos_activos_norm'].mean()
    future_nnm['churn_rate_norm'] = ultimas_semanas['churn_rate_norm'].mean()
    future_nnm = future_nnm.fillna(0)
    
    forecast_nnm = m_nnm.predict(future_nnm)
    
    fecha_max_nnm = df_prophet_nnm['ds'].max()
    forecast_futuro_nnm = forecast_nnm[forecast_nnm['ds'] > fecha_max_nnm]
    
    print(f"\n💰 PRONÓSTICO NNM 52 SEMANAS:")
    print(f"NNM Promedio Semanal: ${forecast_futuro_nnm['yhat'].mean()/1e9:.3f}B")
    print(f"NNM Total: ${forecast_futuro_nnm['yhat'].sum()/1e9:.2f}B")

# %% Preparar datos Nuevos Clientes para Prophet
if PROPHET_DISPONIBLE:
    clients['week'] = clients['registration_date'].dt.to_period('W').dt.to_timestamp()
    registros_semana = clients.groupby('week').size().reset_index()
    registros_semana.columns = ['ds', 'y']
    registros_semana['y'] = detectar_outliers_iqr(registros_semana['y'])
    
    print(f"\nDatos Nuevos Clientes preparados: {len(registros_semana)} semanas")

# %% Modelo Prophet Nuevos Clientes
if PROPHET_DISPONIBLE:
    print("\n🚀 Entrenando modelo Prophet Nuevos Clientes...")
    
    m_clientes = Prophet(
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95
    )
    
    m_clientes.fit(registros_semana)
    
    future_clientes = m_clientes.make_future_dataframe(periods=52, freq='W')
    forecast_clientes = m_clientes.predict(future_clientes)
    
    fecha_max_clientes = registros_semana['ds'].max()
    forecast_futuro_clientes = forecast_clientes[forecast_clientes['ds'] > fecha_max_clientes]
    
    print(f"\n👥 PRONÓSTICO NUEVOS CLIENTES 52 SEMANAS:")
    print(f"Clientes Promedio Semanal: {forecast_futuro_clientes['yhat'].mean():.1f}")
    print(f"Clientes Total: {forecast_futuro_clientes['yhat'].sum():.0f}")

# %% Guardar modelos Prophet
if PROPHET_DISPONIBLE:
    carpeta = 'modelos_prophet_tinvest'
    os.makedirs(carpeta, exist_ok=True)
    
    with open(f'{carpeta}/modelo_nnm.pkl', 'wb') as f:
        pickle.dump(m_nnm, f)
    forecast_nnm.to_csv(f'{carpeta}/forecast_nnm.csv', index=False)
    
    with open(f'{carpeta}/modelo_nuevos_clientes.pkl', 'wb') as f:
        pickle.dump(m_clientes, f)
    forecast_clientes.to_csv(f'{carpeta}/forecast_nuevos_clientes.csv', index=False)
    
    print(f"\n✓ Modelos Prophet guardados en {carpeta}/")

# %% Visualización Prophet NNM
if PROPHET_DISPONIBLE:
    fig, ax = plt.subplots(figsize=(16, 8))
    
    fecha_max = df_prophet_nnm['ds'].max()
    forecast_hist = forecast_nnm[forecast_nnm['ds'] <= fecha_max]
    forecast_futuro = forecast_nnm[forecast_nnm['ds'] > fecha_max]
    
    scale = 1e9
    
    ax.plot(df_prophet_nnm['ds'], df_prophet_nnm['y']/scale, 'o',
            label='Histórico', color='#2E86AB', markersize=4, alpha=0.6)
    ax.plot(forecast_hist['ds'], forecast_hist['yhat']/scale,
            '-', label='Ajuste modelo', color='#06A77D', linewidth=2)
    ax.plot(forecast_futuro['ds'], forecast_futuro['yhat']/scale,
            '-o', label='Pronóstico 52 semanas', color='#FF6B35', linewidth=2.5, markersize=5)
    ax.fill_between(forecast_futuro['ds'],
                     forecast_futuro['yhat_lower']/scale,
                     forecast_futuro['yhat_upper']/scale,
                     alpha=0.2, color='#FF6B35', label='IC 95%')
    ax.axvline(fecha_max, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
    ax.set_ylabel('NNM (Miles de Millones $)', fontsize=12, fontweight='bold')
    ax.set_title('Pronóstico de Net New Money (NNM) - Prophet', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_prophet_nnm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Guardado: fig_prophet_nnm.png")

# %% Visualización Prophet Nuevos Clientes
if PROPHET_DISPONIBLE:
    fig, ax = plt.subplots(figsize=(16, 8))
    
    fecha_max = registros_semana['ds'].max()
    forecast_hist = forecast_clientes[forecast_clientes['ds'] <= fecha_max]
    forecast_futuro = forecast_clientes[forecast_clientes['ds'] > fecha_max]
    
    ax.plot(registros_semana['ds'], registros_semana['y'], 'o',
            label='Histórico', color='#2E86AB', markersize=4, alpha=0.6)
    ax.plot(forecast_hist['ds'], forecast_hist['yhat'],
            '-', label='Ajuste modelo', color='#06A77D', linewidth=2)
    ax.plot(forecast_futuro['ds'], forecast_futuro['yhat'],
            '-o', label='Pronóstico 52 semanas', color='#FF6B35', linewidth=2.5, markersize=5)
    ax.fill_between(forecast_futuro['ds'],
                     forecast_futuro['yhat_lower'],
                     forecast_futuro['yhat_upper'],
                     alpha=0.2, color='#FF6B35', label='IC 95%')
    ax.axvline(fecha_max, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nuevos Clientes', fontsize=12, fontweight='bold')
    ax.set_title('Pronóstico de Nuevos Clientes - Prophet', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_prophet_clientes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Guardado: fig_prophet_clientes.png")

# %% [markdown]
# ## 8. Resumen Ejecutivo

# %% Resumen final
print("\n" + "="*80)
print("RESUMEN EJECUTIVO COMPLETO - TINVEST")
print("="*80)

print(f"""
📊 MÉTRICAS CLAVE:
  • Total Clientes: {len(clients):,}
  • Total Transacciones: {len(transactions):,}
  • Churn Rate: {churn_analysis['churned'].mean()*100:.1f}%
  • Retención Mes 3: {retention_rate[3].mean():.1f}%

💰 FLUJO DE DINERO:
  • Total Depósitos: ${total_deposits/1e9:.2f}B
  • Total Retiros: ${total_withdrawals/1e9:.2f}B
  • NNM Total: ${net_flow/1e9:.2f}B

💎 SEGMENTACIÓN RFM:
  • Campeones: {len(rfm[rfm['segmento_rfm']=='Campeones'])} ({len(rfm[rfm['segmento_rfm']=='Campeones'])/len(rfm)*100:.1f}%)
  • En Riesgo: {len(rfm[rfm['segmento_rfm']=='En Riesgo'])} ({len(rfm[rfm['segmento_rfm']=='En Riesgo'])/len(rfm)*100:.1f}%)
  • Perdidos: {len(rfm[rfm['segmento_rfm']=='Perdidos'])} ({len(rfm[rfm['segmento_rfm']=='Perdidos'])/len(rfm)*100:.1f}%)

🤖 MODELO PREDICTIVO:
  • Mejor modelo: {best_model_name}
  • ROC-AUC: {results.loc[best_model_idx, 'ROC-AUC']:.3f}
  • Clientes alto riesgo: {len(high_risk)}
""")

if PROPHET_DISPONIBLE:
    print(f"""🔮 FORECASTING (52 SEMANAS):
  • NNM Promedio Semanal: ${forecast_futuro_nnm['yhat'].mean()/1e9:.3f}B
  • NNM Total: ${forecast_futuro_nnm['yhat'].sum()/1e9:.2f}B
  • Nuevos Clientes Promedio: {forecast_futuro_clientes['yhat'].mean():.1f}/semana
  • Total Nuevos Clientes: {forecast_futuro_clientes['yhat'].sum():.0f}
""")

print("""
✅ ARCHIVOS GENERADOS:
  • rfm_segmentation.csv
  • churn_analysis.csv
  • retention_matrix.csv
  • churn_predictions.csv
  • fig1_nnm_evolution.png
  • fig2_retention_matrix.png
  • fig3_rfm_segments.png
  • fig_model_performance.png
  • fig_prophet_nnm.png (si Prophet disponible)
  • fig_prophet_clientes.png (si Prophet disponible)
  • modelos_prophet_tinvest/ (si Prophet disponible)

📌 RECOMENDACIONES:
  1. Priorizar intervención en clientes de alto riesgo (prob > 0.7)
  2. Reactivar segmentos 'En Riesgo' y 'Hibernando'
  3. Programas de fidelización para 'Campeones' y 'Leales'
  4. Mejorar onboarding para retención temprana
  5. Monitorear NNM y registros semanalmente
""")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)