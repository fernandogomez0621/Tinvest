# -*- coding: utf-8 -*-
"""
Cohort Analysis Module
Análisis de cohortes y retención
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

@st.cache_data
def calculate_cohorts(clients, transactions, fecha_corte):
    """Calcula las métricas de cohortes"""
    # Crear cohortes
    clients_copy = clients.copy()
    clients_copy['cohort'] = clients_copy['registration_date'].dt.to_period('M')
    
    # Unir transacciones con cohortes
    trans_with_cohort = transactions.merge(
        clients_copy[['client_id', 'cohort', 'registration_date']],
        on='client_id'
    )
    
    # Calcular período
    trans_with_cohort['periodo'] = (
        (trans_with_cohort['date'].dt.to_period('M') - trans_with_cohort['cohort'])
        .apply(lambda x: x.n)
    )
    
    # Tabla de retención
    retention_table = trans_with_cohort.groupby(['cohort', 'periodo'])['client_id'].nunique().reset_index()
    retention_table.columns = ['cohort', 'periodo', 'clientes_activos']
    
    # Matriz de retención
    retention_matrix = retention_table.pivot(index='cohort', columns='periodo', values='clientes_activos')
    
    # Tamaño de cohortes
    cohort_sizes = clients_copy.groupby('cohort').size()
    retention_rate = retention_matrix.divide(retention_matrix.index.map(cohort_sizes.to_dict()), axis=0) * 100
    
    return retention_rate, cohort_sizes, trans_with_cohort

def render(clients, transactions, fecha_corte):
    """Renderiza el análisis de cohortes"""
    st.markdown("## 👥 Análisis de Cohortes y Retención")
    st.markdown("---")
    
    # Calcular cohortes
    with st.spinner("Calculando cohortes..."):
        retention_rate, cohort_sizes, trans_with_cohort = calculate_cohorts(clients, transactions, fecha_corte)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cohortes", len(cohort_sizes))
    with col2:
        ret_m1 = retention_rate[1].mean() if 1 in retention_rate.columns else 0
        st.metric("Retención Mes 1", f"{ret_m1:.1f}%")
    with col3:
        ret_m3 = retention_rate[3].mean() if 3 in retention_rate.columns else 0
        st.metric("Retención Mes 3", f"{ret_m3:.1f}%")
    with col4:
        ret_m6 = retention_rate[6].mean() if 6 in retention_rate.columns else 0
        st.metric("Retención Mes 6", f"{ret_m6:.1f}%")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔥 Mapa de Calor", "📊 Tendencias", "📈 Análisis Detallado"])
    
    with tab1:
        render_heatmap(retention_rate)
    
    with tab2:
        render_trends(retention_rate)
    
    with tab3:
        render_detailed_analysis(retention_rate, cohort_sizes, trans_with_cohort)

def render_heatmap(retention_rate):
    """Renderiza el mapa de calor de retención"""
    st.markdown("### 🔥 Matriz de Retención")
    
    # Selector de meses
    max_months = min(12, len(retention_rate.columns))
    months_to_show = st.slider("Meses a mostrar", 3, max_months, min(7, max_months))
    
    # Selector de cohortes
    cohorts_to_show = st.slider("Cohortes a mostrar", 3, len(retention_rate), min(12, len(retention_rate)))
    
    # Preparar datos
    retention_clean = retention_rate.iloc[-cohorts_to_show:, :months_to_show].copy()
    retention_clean.index = [str(idx) for idx in retention_clean.index]
    retention_clean.columns = [f"M{col}" for col in retention_clean.columns]
    
    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=retention_clean.values,
        x=retention_clean.columns,
        y=retention_clean.index,
        colorscale='RdYlGn',
        zmin=0,
        zmax=100,
        text=retention_clean.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Retención (%)")
    ))
    
    fig.update_layout(
        title="Matriz de Retención por Cohorte",
        xaxis_title="Meses desde Registro",
        yaxis_title="Cohorte (Mes de Registro)",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas
    st.markdown("#### 📊 Estadísticas de Retención")
    stats_data = {
        'Mes': [f"M{i}" for i in range(months_to_show)],
        'Promedio': [retention_rate[i].mean() if i in retention_rate.columns else 0 for i in range(months_to_show)],
        'Mínimo': [retention_rate[i].min() if i in retention_rate.columns else 0 for i in range(months_to_show)],
        'Máximo': [retention_rate[i].max() if i in retention_rate.columns else 0 for i in range(months_to_show)]
    }
    stats_df = pd.DataFrame(stats_data)
    stats_df[['Promedio', 'Mínimo', 'Máximo']] = stats_df[['Promedio', 'Mínimo', 'Máximo']].round(1)
    st.dataframe(stats_df, use_container_width=True)

def render_trends(retention_rate):
    """Renderiza las tendencias de retención"""
    st.markdown("### 📊 Tendencias de Retención")
    
    # Retención promedio por mes
    months_to_plot = min(12, len(retention_rate.columns))
    avg_retention = [retention_rate[i].mean() if i in retention_rate.columns else 0 
                     for i in range(months_to_plot)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(months_to_plot)),
        y=avg_retention,
        mode='lines+markers',
        name='Retención Promedio',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Retención Promedio por Mes desde Registro",
        xaxis_title="Meses desde Registro",
        yaxis_title="Tasa de Retención (%)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Retención por cohorte específica
    st.markdown("#### 📈 Retención por Cohorte")
    
    # Selector de cohortes
    cohorts_list = [str(c) for c in retention_rate.index[-10:]]
    selected_cohorts = st.multiselect(
        "Selecciona cohortes para comparar",
        cohorts_list,
        default=cohorts_list[:3] if len(cohorts_list) >= 3 else cohorts_list
    )
    
    if selected_cohorts:
        fig = go.Figure()
        
        for cohort in selected_cohorts:
            cohort_data = retention_rate.loc[cohort, :months_to_plot]
            fig.add_trace(go.Scatter(
                x=list(range(len(cohort_data))),
                y=cohort_data.values,
                mode='lines+markers',
                name=str(cohort)
            ))
        
        fig.update_layout(
            title="Comparación de Retención entre Cohortes",
            xaxis_title="Meses desde Registro",
            yaxis_title="Tasa de Retención (%)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_detailed_analysis(retention_rate, cohort_sizes, trans_with_cohort):
    """Análisis detallado de cohortes"""
    st.markdown("### 📈 Análisis Detallado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tamaño de Cohortes")
        cohort_sizes_df = cohort_sizes.reset_index()
        cohort_sizes_df.columns = ['Cohorte', 'Clientes']
        cohort_sizes_df['Cohorte'] = cohort_sizes_df['Cohorte'].astype(str)
        
        fig = px.bar(
            cohort_sizes_df.tail(12),
            x='Cohorte',
            y='Clientes',
            title="Clientes por Cohorte (últimos 12 meses)",
            color='Clientes',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Drop-off Rate")
        # Calcular drop-off (pérdida entre meses consecutivos)
        months_to_show = min(6, len(retention_rate.columns))
        dropoff_data = []
        
        for i in range(1, months_to_show):
            if i in retention_rate.columns and i-1 in retention_rate.columns:
                dropoff = retention_rate[i-1].mean() - retention_rate[i].mean()
                dropoff_data.append({'Mes': f"M{i-1}→M{i}", 'Drop-off': dropoff})
        
        if dropoff_data:
            dropoff_df = pd.DataFrame(dropoff_data)
            fig = px.bar(
                dropoff_df,
                x='Mes',
                y='Drop-off',
                title="Pérdida de Retención entre Meses",
                color='Drop-off',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de retención
    st.markdown("#### 📋 Tabla de Retención Detallada")
    
    retention_display = retention_rate.iloc[-10:, :7].copy()
    retention_display.index = [str(idx) for idx in retention_display.index]
    retention_display = retention_display.round(1)
    
    st.dataframe(
        retention_display.style.background_gradient(cmap='RdYlGn', vmin=0, vmax=100),
        use_container_width=True
    )