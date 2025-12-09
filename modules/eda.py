# -*- coding: utf-8 -*-
"""
EDA Module
Análisis Exploratorio de Datos
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render(clients, transactions, portfolio):
    """Renderiza el análisis exploratorio"""
    st.markdown("## 🔍 Análisis Exploratorio de Datos")
    st.markdown("---")
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Clientes", "💰 Transacciones", "📈 Portfolio", "📉 Calidad de Datos"])
    
    with tab1:
        render_clients_analysis(clients)
    
    with tab2:
        render_transactions_analysis(transactions)
    
    with tab3:
        render_portfolio_analysis(portfolio)
    
    with tab4:
        render_data_quality(clients, transactions, portfolio)

def render_clients_analysis(clients):
    """Análisis de clientes"""
    st.markdown("### 👥 Análisis de Clientes")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Clientes", f"{len(clients):,}")
    with col2:
        avg_age = clients['age'].mean()
        st.metric("Edad Promedio", f"{avg_age:.1f} años")
    with col3:
        avg_income = clients['income_monthly'].mean()
        st.metric("Ingreso Promedio", f"${avg_income:,.0f}")
    with col4:
        avg_risk = clients['risk_score'].mean()
        st.metric("Risk Score Promedio", f"{avg_risk:.1f}")
    
    st.markdown("---")
    
    # Distribuciones
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribución por Segmento")
        segment_counts = clients['segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Clientes por Segmento",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Distribución de Edad")
        fig = px.histogram(
            clients,
            x='age',
            nbins=30,
            title="Distribución de Edad de Clientes",
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(xaxis_title="Edad", yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)
    
    # Ingresos vs Risk Score
    st.markdown("#### Ingresos vs Risk Score")
    fig = px.scatter(
        clients,
        x='income_monthly',
        y='risk_score',
        color='segment',
        title="Relación Ingresos - Risk Score",
        labels={'income_monthly': 'Ingreso Mensual ($)', 'risk_score': 'Risk Score'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Registros por mes
    st.markdown("#### Registros por Mes")
    clients_month = clients.set_index('registration_date').resample('M').size()
    fig = px.line(
        x=clients_month.index,
        y=clients_month.values,
        title="Nuevos Clientes por Mes",
        labels={'x': 'Mes', 'y': 'Nuevos Clientes'}
    )
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)

def render_transactions_analysis(transactions):
    """Análisis de transacciones"""
    st.markdown("### 💰 Análisis de Transacciones")
    
    # Métricas
    total_deposits = transactions[transactions['type'] == 'deposit']['amount'].sum()
    total_withdrawals = transactions[transactions['type'] == 'withdrawal']['amount'].sum()
    net_flow = total_deposits - total_withdrawals
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Depósitos", f"${total_deposits/1e9:.2f}B")
    with col2:
        st.metric("Total Retiros", f"${total_withdrawals/1e9:.2f}B")
    with col3:
        st.metric("NNM", f"${net_flow/1e9:.2f}B", delta=f"{(net_flow/total_deposits)*100:.1f}%")
    
    st.markdown("---")
    
    # Distribuciones
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Transacciones por Tipo")
        type_counts = transactions['type'].value_counts()
        fig = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title="Distribución por Tipo",
            labels={'x': 'Tipo', 'y': 'Cantidad'},
            color=type_counts.index,
            color_discrete_map={'deposit': '#2ecc71', 'withdrawal': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Transacciones por Producto")
        product_counts = transactions['product'].value_counts()
        fig = px.bar(
            x=product_counts.index,
            y=product_counts.values,
            title="Distribución por Producto",
            labels={'x': 'Producto', 'y': 'Cantidad'},
            color_discrete_sequence=['#3498db']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Evolución temporal
    st.markdown("#### Evolución del NNM")
    transactions['year_month'] = transactions['date'].dt.to_period('M').dt.to_timestamp()
    flujo_mensual = transactions.groupby(['year_month', 'type'])['amount'].sum().unstack(fill_value=0)
    
    # Asegurar columnas
    if 'deposit' not in flujo_mensual.columns:
        flujo_mensual['deposit'] = 0
    if 'withdrawal' not in flujo_mensual.columns:
        flujo_mensual['withdrawal'] = 0
    
    flujo_mensual['net_flow'] = flujo_mensual['deposit'] - flujo_mensual['withdrawal']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=flujo_mensual.index,
        y=flujo_mensual['deposit']/1e9,
        name='Depósitos',
        marker_color='#5cb85c',
        opacity=0.85,
        width=20*24*60*60*1000
    ))
    
    fig.add_trace(go.Bar(
        x=flujo_mensual.index,
        y=-flujo_mensual['withdrawal']/1e9,
        name='Retiros',
        marker_color='#d9534f',
        opacity=0.85,
        width=20*24*60*60*1000
    ))
    
    fig.add_trace(go.Scatter(
        x=flujo_mensual.index,
        y=flujo_mensual['net_flow']/1e9,
        name='Flujo Neto (NNM)',
        mode='lines+markers',
        line=dict(color='#2b7bba', width=3),
        marker=dict(size=7),
        yaxis='y2'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(128,128,128,0.3)", line_width=1)
    
    fig.update_layout(
        title="Evolución del Net New Money (NNM)",
        xaxis=dict(
            title="Mes",
            tickformat='%Y-%m',
            tickangle=-45,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title="Flujo (Miles de Millones $)",
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True
        ),
        yaxis2=dict(
            title="NNM (Miles de Millones $)",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        barmode='relative',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def render_portfolio_analysis(portfolio):
    """Análisis de portfolio"""
    st.markdown("### 📈 Análisis de Portfolio")
    
    # Balance promedio por fecha
    balance_time = portfolio.groupby('date')['balance'].agg(['mean', 'sum', 'count']).reset_index()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Balance Promedio", f"${balance_time['mean'].iloc[-1]:,.0f}")
    with col2:
        st.metric("Balance Total", f"${balance_time['sum'].iloc[-1]/1e9:.2f}B")
    with col3:
        st.metric("Clientes con Balance", f"{balance_time['count'].iloc[-1]:,}")
    
    st.markdown("---")
    
    # Evolución del balance
    st.markdown("#### Evolución del Balance Total")
    fig = px.line(
        balance_time,
        x='date',
        y='sum',
        title="Balance Total en el Tiempo",
        labels={'date': 'Fecha', 'sum': 'Balance Total ($)'}
    )
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribución de balances
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribución de Balance (último día)")
        last_date = portfolio['date'].max()
        last_portfolio = portfolio[portfolio['date'] == last_date]
        
        fig = px.histogram(
            last_portfolio[last_portfolio['balance'] > 0],
            x='balance',
            nbins=50,
            title="Distribución de Balances",
            labels={'balance': 'Balance ($)'},
            color_discrete_sequence=['#9b59b6']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Clientes por Rango de Balance")
        last_portfolio_copy = last_portfolio.copy()
        last_portfolio_copy['balance_range'] = pd.cut(
            last_portfolio_copy['balance'],
            bins=[0, 1000, 10000, 50000, 100000, float('inf')],
            labels=['$0-1K', '$1K-10K', '$10K-50K', '$50K-100K', '$100K+']
        )
        range_counts = last_portfolio_copy['balance_range'].value_counts().sort_index()
        
        fig = px.bar(
            x=range_counts.index.astype(str),
            y=range_counts.values,
            title="Clientes por Rango de Balance",
            labels={'x': 'Rango', 'y': 'Clientes'},
            color_discrete_sequence=['#e67e22']
        )
        st.plotly_chart(fig, use_container_width=True)

def render_data_quality(clients, transactions, portfolio):
    """Análisis de calidad de datos"""
    st.markdown("### 📉 Calidad de Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Clientes")
        st.metric("Registros", f"{len(clients):,}")
        st.metric("Duplicados", f"{clients.duplicated().sum():,}")
        nulos_clients = clients.isnull().sum().sum()
        st.metric("Valores Nulos", f"{nulos_clients:,}")
        
        if nulos_clients > 0:
            st.warning("Columnas con nulos:")
            st.write(clients.isnull().sum()[clients.isnull().sum() > 0])
    
    with col2:
        st.markdown("#### Transacciones")
        st.metric("Registros", f"{len(transactions):,}")
        st.metric("Duplicados", f"{transactions.duplicated().sum():,}")
        nulos_trans = transactions.isnull().sum().sum()
        st.metric("Valores Nulos", f"{nulos_trans:,}")
        
        if nulos_trans > 0:
            st.warning("Columnas con nulos:")
            st.write(transactions.isnull().sum()[transactions.isnull().sum() > 0])
    
    with col3:
        st.markdown("#### Portfolio")
        st.metric("Registros", f"{len(portfolio):,}")
        st.metric("Duplicados", f"{portfolio.duplicated().sum():,}")
        nulos_port = portfolio.isnull().sum().sum()
        st.metric("Valores Nulos", f"{nulos_port:,}")
        
        if nulos_port > 0:
            st.warning("Columnas con nulos:")
            st.write(portfolio.isnull().sum()[portfolio.isnull().sum() > 0])
    
    st.markdown("---")
    
    # Rangos temporales
    st.markdown("#### Rangos Temporales")
    
    info_data = {
        'Dataset': ['Clientes', 'Transacciones', 'Portfolio'],
        'Fecha Inicial': [
            clients['registration_date'].min().strftime('%Y-%m-%d'),
            transactions['date'].min().strftime('%Y-%m-%d'),
            portfolio['date'].min().strftime('%Y-%m-%d')
        ],
        'Fecha Final': [
            clients['registration_date'].max().strftime('%Y-%m-%d'),
            transactions['date'].max().strftime('%Y-%m-%d'),
            portfolio['date'].max().strftime('%Y-%m-%d')
        ],
        'Días': [
            (clients['registration_date'].max() - clients['registration_date'].min()).days,
            (transactions['date'].max() - transactions['date'].min()).days,
            (portfolio['date'].max() - portfolio['date'].min()).days
        ]
    }
    
    st.dataframe(pd.DataFrame(info_data), use_container_width=True)