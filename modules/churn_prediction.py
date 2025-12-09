# -*- coding: utf-8 -*-
"""
Churn Prediction Module
Modelos predictivos de churn
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score, accuracy_score

@st.cache_data
def prepare_churn_data(clients, transactions, portfolio, fecha_corte):
    """Prepara los datos para el modelo de churn"""
    # Análisis de churn
    ultima_trans = transactions.groupby('client_id')['date'].max().reset_index()
    ultima_trans.columns = ['client_id', 'ultima_transaccion']
    
    churn_analysis = clients.merge(ultima_trans, on='client_id', how='left')
    churn_analysis['dias_inactivo'] = (fecha_corte - churn_analysis['ultima_transaccion']).dt.days
    
    balance_actual = portfolio[portfolio['date'] == portfolio['date'].max()].groupby('client_id')['balance'].sum().reset_index()
    balance_actual.columns = ['client_id', 'balance_actual']
    churn_analysis = churn_analysis.merge(balance_actual, on='client_id', how='left')
    churn_analysis['balance_actual'] = churn_analysis['balance_actual'].fillna(0)
    
    churn_analysis['churned'] = (
        (churn_analysis['balance_actual'] == 0) &
        (churn_analysis['dias_inactivo'] > 90)
    ).astype(int)
    
    return churn_analysis

@st.cache_data
def build_features(clients, transactions, portfolio, fecha_corte):
    """Construye features para el modelo"""
    model_data = clients[['client_id', 'age', 'income_monthly', 'segment', 'risk_score']].copy()
    
    # Recency, Frequency, Monetary
    recency = transactions.groupby('client_id')['date'].max().reset_index()
    recency['recency'] = (fecha_corte - recency['date']).dt.days
    
    frequency = transactions.groupby('client_id').size().reset_index()
    frequency.columns = ['client_id', 'frequency']
    
    monetary = transactions[transactions['type'] == 'deposit'].groupby('client_id')['amount'].sum().reset_index()
    monetary.columns = ['client_id', 'monetary']
    
    model_data = model_data.merge(recency[['client_id', 'recency']], on='client_id', how='left')
    model_data = model_data.merge(frequency, on='client_id', how='left')
    model_data = model_data.merge(monetary, on='client_id', how='left')
    
    # Días desde registro
    model_data['dias_desde_registro'] = (fecha_corte - clients['registration_date']).dt.days
    
    # Productos únicos
    productos = transactions.groupby('client_id')['product'].nunique().reset_index()
    productos.columns = ['client_id', 'num_productos']
    model_data = model_data.merge(productos, on='client_id', how='left')
    
    # Balance actual
    balance_actual = portfolio[portfolio['date'] == portfolio['date'].max()].groupby('client_id')['balance'].sum().reset_index()
    balance_actual.columns = ['client_id', 'balance_actual']
    model_data = model_data.merge(balance_actual, on='client_id', how='left')
    
    # Ratio depósitos/retiros
    dep_ret = transactions.groupby(['client_id', 'type'])['amount'].sum().unstack(fill_value=0)
    if 'deposit' in dep_ret.columns and 'withdrawal' in dep_ret.columns:
        dep_ret['deposit_withdrawal_ratio'] = dep_ret['deposit'] / (dep_ret['withdrawal'] + 1)
        model_data = model_data.merge(dep_ret[['deposit_withdrawal_ratio']], 
                                     left_on='client_id', right_index=True, how='left')
    
    # Variabilidad de transacciones
    trans_std = transactions.groupby('client_id')['amount'].std().reset_index()
    trans_std.columns = ['client_id', 'amount_std']
    model_data = model_data.merge(trans_std, on='client_id', how='left')
    
    # Target
    churn_analysis = prepare_churn_data(clients, transactions, portfolio, fecha_corte)
    model_data = model_data.merge(churn_analysis[['client_id', 'churned']], on='client_id', how='left')
    
    model_data = model_data.fillna(0)
    
    return model_data

@st.cache_resource
def train_models(model_data):
    """Entrena los modelos de churn"""
    X = model_data.drop(['client_id', 'churned', 'segment'], axis=1).copy()
    y = model_data['churned']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos
    models = {}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = {
        'model': lr,
        'predictions': lr.predict(X_test_scaled),
        'probabilities': lr.predict_proba(X_test_scaled)[:, 1],
        'scaled': True
    }
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
    rf.fit(X_train, y_train)
    models['Random Forest'] = {
        'model': rf,
        'predictions': rf.predict(X_test),
        'probabilities': rf.predict_proba(X_test)[:, 1],
        'scaled': False,
        'feature_importance': pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
    }
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = {
        'model': gb,
        'predictions': gb.predict(X_test),
        'probabilities': gb.predict_proba(X_test)[:, 1],
        'scaled': False
    }
    
    return models, X_train, X_test, y_train, y_test, scaler

def render(clients, transactions, portfolio, fecha_corte):
    """Renderiza el módulo de predicción de churn"""
    st.markdown("## ⚠️ Predicción de Churn")
    st.markdown("---")
    
    # Preparar datos
    with st.spinner("Preparando datos y entrenando modelos..."):
        churn_analysis = prepare_churn_data(clients, transactions, portfolio, fecha_corte)
        model_data = build_features(clients, transactions, portfolio, fecha_corte)
        models, X_train, X_test, y_train, y_test, scaler = train_models(model_data)
    
    # Métricas de churn
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churned_count = churn_analysis['churned'].sum()
        st.metric("Clientes Churned", f"{churned_count:,}", 
                 f"{churned_count/len(churn_analysis)*100:.1f}%")
    with col2:
        active_count = (1-churn_analysis['churned']).sum()
        st.metric("Clientes Activos", f"{active_count:,}",
                 f"{active_count/len(churn_analysis)*100:.1f}%")
    with col3:
        avg_inactive = churn_analysis['dias_inactivo'].mean()
        st.metric("Días Inactivos Prom.", f"{avg_inactive:.0f}")
    with col4:
        high_risk = len(model_data[model_data['churned'] == 0])
        st.metric("Total Clientes", f"{high_risk:,}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Modelos", "📊 Performance", "⚠️ Alto Riesgo", "📈 Análisis"])
    
    with tab1:
        render_models(models, y_test)
    
    with tab2:
        render_performance(models, X_test, y_test)
    
    with tab3:
        render_high_risk(model_data, models, X_train.columns)
    
    with tab4:
        render_analysis(churn_analysis, model_data)

def render_models(models, y_test):
    """Muestra comparación de modelos"""
    st.markdown("### 🤖 Comparación de Modelos")
    
    # Calcular métricas
    results = []
    for name, model_info in models.items():
        results.append({
            'Modelo': name,
            'Accuracy': accuracy_score(y_test, model_info['predictions']),
            'ROC-AUC': roc_auc_score(y_test, model_info['probabilities']),
            'F1-Score': f1_score(y_test, model_info['predictions'])
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.round(3)
    
    # Mostrar tabla
    st.dataframe(
        results_df.style.highlight_max(subset=['Accuracy', 'ROC-AUC', 'F1-Score'], axis=0),
        use_container_width=True
    )
    
    # Mejor modelo
    best_model = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Modelo']
    best_score = results_df.loc[results_df['ROC-AUC'].idxmax(), 'ROC-AUC']
    st.success(f"🏆 **Mejor Modelo:** {best_model} (ROC-AUC: {best_score:.3f})")
    
    # Feature importance (Random Forest)
    if 'Random Forest' in models:
        st.markdown("### 📊 Importancia de Variables (Random Forest)")
        
        fi = models['Random Forest']['feature_importance'].head(15)
        fig = px.bar(
            fi,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 15 Features más Importantes",
            labels={'importance': 'Importancia', 'feature': 'Variable'},
            color='importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def render_performance(models, X_test, y_test):
    """Muestra performance detallado"""
    st.markdown("### 📊 Performance Detallado")
    
    # Curvas ROC
    st.markdown("#### Curvas ROC")
    
    fig = go.Figure()
    
    for name, model_info in models.items():
        fpr, tpr, _ = roc_curve(y_test, model_info['probabilities'])
        auc = roc_auc_score(y_test, model_info['probabilities'])
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{name} (AUC={auc:.3f})',
            line=dict(width=2)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title="Curvas ROC - Comparación de Modelos",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribución de probabilidades
    st.markdown("#### Distribución de Probabilidades")
    
    col1, col2 = st.columns(2)
    
    for i, (name, model_info) in enumerate(list(models.items())[:2]):
        with col1 if i == 0 else col2:
            probs_df = pd.DataFrame({
                'probability': model_info['probabilities'],
                'actual': ['Churned' if x == 1 else 'Active' for x in y_test]
            })
            
            fig = px.histogram(
                probs_df,
                x='probability',
                color='actual',
                nbins=50,
                title=f"Distribución - {name}",
                labels={'probability': 'Probabilidad de Churn'},
                barmode='overlay',
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)

def render_high_risk(model_data, models, feature_columns):
    """Muestra clientes de alto riesgo"""
    st.markdown("### ⚠️ Clientes en Alto Riesgo")
    
    # Usar Random Forest para predicciones
    rf_model = models['Random Forest']['model']
    
    X_all = model_data.drop(['client_id', 'churned', 'segment'], axis=1)
    X_all = X_all[feature_columns]
    
    churn_probability = rf_model.predict_proba(X_all)[:, 1]
    model_data['churn_probability'] = churn_probability
    model_data['risk_level'] = pd.cut(
        churn_probability,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    # Métricas de riesgo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alto_riesgo = len(model_data[model_data['risk_level'] == 'Alto'])
        st.metric("Alto Riesgo", f"{alto_riesgo:,}", 
                 f"{alto_riesgo/len(model_data)*100:.1f}%")
    with col2:
        medio_riesgo = len(model_data[model_data['risk_level'] == 'Medio'])
        st.metric("Medio Riesgo", f"{medio_riesgo:,}",
                 f"{medio_riesgo/len(model_data)*100:.1f}%")
    with col3:
        bajo_riesgo = len(model_data[model_data['risk_level'] == 'Bajo'])
        st.metric("Bajo Riesgo", f"{bajo_riesgo:,}",
                 f"{bajo_riesgo/len(model_data)*100:.1f}%")
    
    # Distribución de riesgo
    st.markdown("#### Distribución de Nivel de Riesgo")
    risk_counts = model_data['risk_level'].value_counts()
    
    fig = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        title="Clientes por Nivel de Riesgo",
        labels={'x': 'Nivel de Riesgo', 'y': 'Clientes'},
        color=risk_counts.index,
        color_discrete_map={'Bajo': '#2ecc71', 'Medio': '#f39c12', 'Alto': '#e74c3c'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Lista de alto riesgo
    st.markdown("#### 🚨 Top 50 Clientes en Mayor Riesgo")
    
    high_risk_clients = model_data[model_data['churned'] == 0].sort_values(
        'churn_probability', ascending=False
    ).head(50)
    
    display_cols = ['client_id', 'segment', 'churn_probability', 'recency', 'frequency', 
                    'monetary', 'balance_actual', 'risk_score']
    high_risk_display = high_risk_clients[display_cols].copy()
    high_risk_display['churn_probability'] = high_risk_display['churn_probability'].apply(lambda x: f"{x:.2%}")
    high_risk_display['monetary'] = high_risk_display['monetary'].apply(lambda x: f"${x:,.0f}")
    high_risk_display['balance_actual'] = high_risk_display['balance_actual'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(high_risk_display, use_container_width=True)
    
    # Descarga
    csv = high_risk_clients.to_csv(index=False)
    st.download_button(
        label="📥 Descargar Lista Completa (CSV)",
        data=csv,
        file_name='high_risk_clients.csv',
        mime='text/csv',
        use_container_width=True
    )

def render_analysis(churn_analysis, model_data):
    """Análisis de churn"""
    st.markdown("### 📈 Análisis de Churn")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Churn por Segmento")
        churn_by_segment = churn_analysis.groupby('segment')['churned'].agg(['sum', 'count'])
        churn_by_segment['rate'] = churn_by_segment['sum'] / churn_by_segment['count'] * 100
        
        fig = px.bar(
            x=churn_by_segment.index,
            y=churn_by_segment['rate'],
            title="Tasa de Churn por Segmento",
            labels={'x': 'Segmento', 'y': 'Tasa de Churn (%)'},
            color=churn_by_segment['rate'],
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Días de Inactividad")
        fig = px.histogram(
            churn_analysis,
            x='dias_inactivo',
            color='churned',
            nbins=50,
            title="Distribución de Días de Inactividad",
            labels={'dias_inactivo': 'Días Inactivo', 'churned': 'Estado'},
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Factores de churn
    st.markdown("#### 🔍 Características de Clientes Churned vs Activos")
    
    comparison = pd.DataFrame({
        'Métrica': ['Edad Promedio', 'Ingreso Promedio', 'Risk Score Promedio', 
                   'Recency Promedio', 'Frequency Promedio', 'Monetary Promedio'],
        'Churned': [
            churn_analysis[churn_analysis['churned']==1]['age'].mean(),
            churn_analysis[churn_analysis['churned']==1]['income_monthly'].mean(),
            churn_analysis[churn_analysis['churned']==1]['risk_score'].mean(),
            model_data[model_data['churned']==1]['recency'].mean(),
            model_data[model_data['churned']==1]['frequency'].mean(),
            model_data[model_data['churned']==1]['monetary'].mean()
        ],
        'Activos': [
            churn_analysis[churn_analysis['churned']==0]['age'].mean(),
            churn_analysis[churn_analysis['churned']==0]['income_monthly'].mean(),
            churn_analysis[churn_analysis['churned']==0]['risk_score'].mean(),
            model_data[model_data['churned']==0]['recency'].mean(),
            model_data[model_data['churned']==0]['frequency'].mean(),
            model_data[model_data['churned']==0]['monetary'].mean()
        ]
    })
    
    comparison = comparison.round(2)
    st.dataframe(comparison, use_container_width=True)