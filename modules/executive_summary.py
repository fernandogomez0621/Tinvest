# -*- coding: utf-8 -*-
"""
Executive Summary Module
Resumen ejecutivo del dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def render(clients, transactions, portfolio, fecha_corte):
    """Renderiza el resumen ejecutivo"""
    st.markdown("## 📊 Resumen Ejecutivo")
    st.markdown("---")
    
    # KPIs principales
    render_kpis(clients, transactions, portfolio, fecha_corte)
    
    st.markdown("---")
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        render_nnm_chart(transactions)
    
    with col2:
        render_client_growth(clients)
    
    st.markdown("---")
    
    # Segunda fila de gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        render_segment_distribution(clients)
    
    with col2:
        render_product_distribution(transactions)
    
    st.markdown("---")
    
    # Insights y recomendaciones
    render_insights(clients, transactions, portfolio, fecha_corte)

def render_kpis(clients, transactions, portfolio, fecha_corte):
    """KPIs principales"""
    st.markdown("### 🎯 KPIs Principales")
    
    # Calcular métricas
    total_clients = len(clients)
    total_transactions = len(transactions)
    
    total_deposits = transactions[transactions['type'] == 'deposit']['amount'].sum()
    total_withdrawals = transactions[transactions['type'] == 'withdrawal']['amount'].sum()
    nnm = total_deposits - total_withdrawals
    
    last_date = portfolio['date'].max()
    total_balance = portfolio[portfolio['date'] == last_date]['balance'].sum()
    
    avg_balance = portfolio[portfolio['date'] == last_date]['balance'].mean()
    
    # Clientes activos (con balance > 0)
    active_clients = len(portfolio[(portfolio['date'] == last_date) & (portfolio['balance'] > 0)])
    
    # Mostrar métricas
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Clientes",
            f"{total_clients:,}",
            delta=f"{active_clients:,} activos"
        )
    
    with col2:
        st.metric(
            "Transacciones",
            f"{total_transactions:,}",
            delta=f"{total_transactions/total_clients:.1f} por cliente"
        )
    
    with col3:
        st.metric(
            "NNM Total",
            f"${nnm/1e9:.2f}B",
            delta=f"{(nnm/total_deposits)*100:.1f}%" if total_deposits > 0 else "0%"
        )
    
    with col4:
        st.metric(
            "Balance Total",
            f"${total_balance/1e9:.2f}B",
            delta=f"${avg_balance:,.0f} promedio"
        )
    
    with col5:
        retention_rate = (active_clients / total_clients * 100) if total_clients > 0 else 0
        st.metric(
            "Tasa Retención",
            f"{retention_rate:.1f}%",
            delta=f"{active_clients:,} activos"
        )

def render_nnm_chart(transactions):
    """Gráfico de evolución del NNM"""
    st.markdown("### 💰 Evolución del Net New Money (NNM)")
    
    transactions_copy = transactions.copy()
    transactions_copy['year_month'] = transactions_copy['date'].dt.to_period('M').dt.to_timestamp()
    
    flujo_mensual = transactions_copy.groupby(['year_month', 'type'])['amount'].sum().unstack(fill_value=0)
    flujo_mensual['net_flow'] = flujo_mensual.get('deposit', 0) - flujo_mensual.get('withdrawal', 0)
    
    fig = go.Figure()
    
    # Barras de depósitos y retiros
    fig.add_trace(go.Bar(
        x=flujo_mensual.index,
        y=flujo_mensual['deposit']/1e9,
        name='Depósitos',
        marker_color='#2ecc71',
        opacity=0.8
    ))
    
    fig.add_trace(go.Bar(
        x=flujo_mensual.index,
        y=-flujo_mensual['withdrawal']/1e9,
        name='Retiros',
        marker_color='#e74c3c',
        opacity=0.8
    ))
    
    # Línea de NNM
    fig.add_trace(go.Scatter(
        x=flujo_mensual.index,
        y=flujo_mensual['net_flow']/1e9,
        name='NNM',
        mode='lines+markers',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Flujo (Miles de Millones $)",
        yaxis2=dict(
            title="NNM (Miles de Millones $)",
            overlaying='y',
            side='right'
        ),
        barmode='relative',
        hovermode='x unified',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_client_growth(clients):
    """Gráfico de crecimiento de clientes"""
    st.markdown("### 👥 Crecimiento de Clientes")
    
    clients_copy = clients.copy()
    clients_copy['year_month'] = clients_copy['registration_date'].dt.to_period('M').dt.to_timestamp()
    
    registros_mensuales = clients_copy.groupby('year_month').size().reset_index()
    registros_mensuales.columns = ['month', 'new_clients']
    registros_mensuales['cumulative'] = registros_mensuales['new_clients'].cumsum()
    
    fig = go.Figure()
    
    # Barras de nuevos clientes
    fig.add_trace(go.Bar(
        x=registros_mensuales['month'],
        y=registros_mensuales['new_clients'],
        name='Nuevos Clientes',
        marker_color='#9b59b6',
        opacity=0.7
    ))
    
    # Línea acumulada
    fig.add_trace(go.Scatter(
        x=registros_mensuales['month'],
        y=registros_mensuales['cumulative'],
        name='Acumulado',
        mode='lines+markers',
        line=dict(color='#e67e22', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        xaxis_title="Mes",
        yaxis_title="Nuevos Clientes",
        yaxis2=dict(
            title="Clientes Acumulados",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_segment_distribution(clients):
    """Distribución por segmento"""
    st.markdown("### 🎯 Distribución por Segmento")
    
    segment_counts = clients['segment'].value_counts()
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    fig = go.Figure(data=[go.Pie(
        labels=segment_counts.index,
        values=segment_counts.values,
        hole=0.4,
        marker_colors=colors,
        textposition='inside',
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_product_distribution(transactions):
    """Distribución por producto"""
    st.markdown("### 📦 Distribución por Producto")
    
    product_counts = transactions['product'].value_counts()
    
    fig = px.bar(
        x=product_counts.index,
        y=product_counts.values,
        title="",
        labels={'x': 'Producto', 'y': 'Transacciones'},
        color=product_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_insights(clients, transactions, portfolio, fecha_corte):
    """Insights y recomendaciones"""
    st.markdown("### 💡 Insights y Recomendaciones")
    
    # Calcular métricas para insights
    total_deposits = transactions[transactions['type'] == 'deposit']['amount'].sum()
    total_withdrawals = transactions[transactions['type'] == 'withdrawal']['amount'].sum()
    nnm = total_deposits - total_withdrawals
    
    last_date = portfolio['date'].max()
    active_clients = len(portfolio[(portfolio['date'] == last_date) & (portfolio['balance'] > 0)])
    total_clients = len(clients)
    
    # Analizar últimos 30 días
    fecha_30_dias = fecha_corte - pd.Timedelta(days=30)
    trans_recientes = transactions[transactions['date'] >= fecha_30_dias]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Fortalezas")
        
        fortalezas = []
        
        # NNM positivo
        if nnm > 0:
            fortalezas.append(f"✅ **NNM positivo** de ${nnm/1e9:.2f}B indica crecimiento saludable")
        
        # Alta retención
        retention_rate = (active_clients / total_clients * 100) if total_clients > 0 else 0
        if retention_rate > 70:
            fortalezas.append(f"✅ **Alta tasa de retención** del {retention_rate:.1f}%")
        
        # Actividad reciente
        if len(trans_recientes) > 0:
            actividad_pct = (len(trans_recientes) / len(transactions) * 100)
            if actividad_pct > 20:
                fortalezas.append(f"✅ **Alta actividad reciente**: {actividad_pct:.1f}% de transacciones en últimos 30 días")
        
        # Diversificación de productos
        productos_unicos = transactions['product'].nunique()
        if productos_unicos >= 4:
            fortalezas.append(f"✅ **Buena diversificación** con {productos_unicos} productos activos")
        
        for fortaleza in fortalezas:
            st.markdown(fortaleza)
        
        if not fortalezas:
            st.info("Generando análisis de fortalezas...")
    
    with col2:
        st.markdown("#### ⚠️ Áreas de Mejora")
        
        mejoras = []
        
        # NNM negativo
        if nnm < 0:
            mejoras.append(f"⚠️ **NNM negativo** de ${nnm/1e9:.2f}B - Revisar estrategia de retención")
        
        # Baja retención
        if retention_rate < 70:
            mejoras.append(f"⚠️ **Baja retención** del {retention_rate:.1f}% - Implementar programas de fidelización")
        
        # Clientes inactivos
        inactive_pct = ((total_clients - active_clients) / total_clients * 100) if total_clients > 0 else 0
        if inactive_pct > 30:
            mejoras.append(f"⚠️ **{inactive_pct:.1f}% de clientes inactivos** - Diseñar campañas de reactivación")
        
        # Concentración en pocos productos
        if productos_unicos < 3:
            mejoras.append(f"⚠️ **Baja diversificación** - Promover productos adicionales")
        
        for mejora in mejoras:
            st.markdown(mejora)
        
        if not mejoras:
            st.success("No se identificaron áreas críticas de mejora")
    
    # Recomendaciones estratégicas
    st.markdown("---")
    st.markdown("#### 🎯 Recomendaciones Estratégicas")
    
    recomendaciones = [
        {
            "título": "🎯 Segmentación Personalizada",
            "descripción": "Implementar estrategias diferenciadas para cada segmento RFM identificado",
            "prioridad": "Alta"
        },
        {
            "título": "⚠️ Prevención de Churn",
            "descripción": "Intervenir proactivamente con clientes de alto riesgo identificados por el modelo predictivo",
            "prioridad": "Alta"
        },
        {
            "título": "💎 Fidelización de Campeones",
            "descripción": "Programas exclusivos para retener a los clientes más valiosos",
            "prioridad": "Media"
        },
        {
            "título": "🔄 Reactivación",
            "descripción": "Campañas específicas para clientes en estado 'Hibernando' y 'En Riesgo'",
            "prioridad": "Media"
        },
        {
            "título": "📊 Monitoreo Continuo",
            "descripción": "Seguimiento semanal de KPIs y ajuste de estrategias según forecasting",
            "prioridad": "Alta"
        }
    ]
    
    for rec in recomendaciones:
        color = "#e74c3c" if rec["prioridad"] == "Alta" else "#f39c12"
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; 
                    border-left: 4px solid {color}; margin-bottom: 1rem;'>
            <h4 style='margin: 0 0 0.5rem 0;'>{rec['título']}</h4>
            <p style='margin: 0; color: #666;'>{rec['descripción']}</p>
            <small style='color: {color}; font-weight: bold;'>Prioridad: {rec['prioridad']}</small>
        </div>
        """, unsafe_allow_html=True)