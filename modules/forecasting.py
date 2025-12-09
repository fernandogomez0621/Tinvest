# -*- coding: utf-8 -*-
"""
Forecasting Module
Pronósticos con Prophet
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

@st.cache_resource
def train_prophet_nnm(transactions):
    """Entrena modelo Prophet para NNM"""
    if not PROPHET_AVAILABLE:
        return None, None
    
    # Preparar datos
    transactions_copy = transactions.copy()
    transactions_copy['week'] = transactions_copy['date'].dt.to_period('W').dt.to_timestamp()
    
    flujo_semanal = transactions_copy.groupby(['week', 'type'])['amount'].sum().unstack(fill_value=0)
    flujo_semanal['net_flow'] = flujo_semanal.get('deposit', 0) - flujo_semanal.get('withdrawal', 0)
    
    # Limpiar outliers
    Q1 = flujo_semanal['net_flow'].quantile(0.25)
    Q3 = flujo_semanal['net_flow'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    p99 = flujo_semanal['net_flow'].quantile(0.99)
    flujo_semanal['net_flow'] = flujo_semanal['net_flow'].clip(lower=lower, upper=min(upper, p99))
    
    df_prophet = flujo_semanal.reset_index()
    df_prophet = df_prophet.rename(columns={'week': 'ds', 'net_flow': 'y'})
    
    # Entrenar modelo
    m = Prophet(
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    
    m.fit(df_prophet)
    
    # Forecast
    future = m.make_future_dataframe(periods=52, freq='W')
    forecast = m.predict(future)
    
    return m, forecast

@st.cache_resource
def train_prophet_clients(clients):
    """Entrena modelo Prophet para nuevos clientes"""
    if not PROPHET_AVAILABLE:
        return None, None
    
    clients_copy = clients.copy()
    clients_copy['week'] = clients_copy['registration_date'].dt.to_period('W').dt.to_timestamp()
    
    registros_semana = clients_copy.groupby('week').size().reset_index()
    registros_semana.columns = ['ds', 'y']
    
    # Limpiar outliers
    Q1 = registros_semana['y'].quantile(0.25)
    Q3 = registros_semana['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    p99 = registros_semana['y'].quantile(0.99)
    registros_semana['y'] = registros_semana['y'].clip(lower=lower, upper=min(upper, p99))
    
    # Entrenar modelo
    m = Prophet(
        seasonality_mode='additive',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    
    m.fit(registros_semana)
    
    # Forecast
    future = m.make_future_dataframe(periods=52, freq='W')
    forecast = m.predict(future)
    
    return m, forecast

def render(transactions, clients):
    """Renderiza el módulo de forecasting"""
    st.markdown("## 🔮 Forecasting")
    st.markdown("---")
    
    if not PROPHET_AVAILABLE:
        st.error("⚠️ Prophet no está instalado. Ejecuta: `pip install prophet`")
        st.info("Este módulo requiere la librería Prophet de Facebook para realizar pronósticos de series temporales.")
        return
    
    # Tabs
    tab1, tab2 = st.tabs(["💰 NNM Forecast", "👥 Nuevos Clientes Forecast"])
    
    with tab1:
        render_nnm_forecast(transactions)
    
    with tab2:
        render_clients_forecast(clients)

def render_nnm_forecast(transactions):
    """Pronóstico de NNM"""
    st.markdown("### 💰 Pronóstico de Net New Money (NNM)")
    
    with st.spinner("Entrenando modelo Prophet para NNM..."):
        model, forecast = train_prophet_nnm(transactions)
    
    if model is None or forecast is None:
        st.error("Error al entrenar el modelo")
        return
    
    # Métricas
    transactions_copy = transactions.copy()
    transactions_copy['week'] = transactions_copy['date'].dt.to_period('W').dt.to_timestamp()
    fecha_max = pd.Timestamp(transactions_copy['week'].max())
    
    forecast_futuro = forecast[forecast['ds'] > fecha_max]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_weekly = forecast_futuro['yhat'].mean()
        st.metric("NNM Semanal Promedio", f"${avg_weekly/1e6:.2f}M")
    with col2:
        total_yearly = forecast_futuro['yhat'].sum()
        st.metric("NNM Total (52 semanas)", f"${total_yearly/1e9:.2f}B")
    with col3:
        current_nnm = forecast[forecast['ds'] <= fecha_max]['yhat'].tail(4).mean()
        growth = ((avg_weekly - current_nnm) / current_nnm * 100) if current_nnm != 0 else 0
        st.metric("Crecimiento vs Actual", f"{growth:+.1f}%")
    
    st.markdown("---")
    
    # Gráfico de pronóstico
    st.markdown("#### 📈 Pronóstico NNM - 52 Semanas")
    
    # Preparar datos históricos
    flujo_semanal = transactions_copy.groupby(['week', 'type'])['amount'].sum().unstack(fill_value=0)
    flujo_semanal['net_flow'] = flujo_semanal.get('deposit', 0) - flujo_semanal.get('withdrawal', 0)
    
    forecast_hist = forecast[forecast['ds'] <= fecha_max]
    forecast_fut = forecast[forecast['ds'] > fecha_max]
    
    fig = go.Figure()
    
    # Histórico real
    fig.add_trace(go.Scatter(
        x=flujo_semanal.index,
        y=flujo_semanal['net_flow']/1e9,
        mode='markers',
        name='Histórico Real',
        marker=dict(size=6, color='#2E86AB')
    ))
    
    # Ajuste del modelo
    fig.add_trace(go.Scatter(
        x=forecast_hist['ds'],
        y=forecast_hist['yhat']/1e9,
        mode='lines',
        name='Ajuste Modelo',
        line=dict(color='#06A77D', width=2)
    ))
    
    # Pronóstico
    fig.add_trace(go.Scatter(
        x=forecast_fut['ds'],
        y=forecast_fut['yhat']/1e9,
        mode='lines+markers',
        name='Pronóstico',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=5)
    ))
    
    # Intervalo de confianza
    fig.add_trace(go.Scatter(
        x=forecast_fut['ds'],
        y=forecast_fut['yhat_upper']/1e9,
        mode='lines',
        name='IC Superior',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_fut['ds'],
        y=forecast_fut['yhat_lower']/1e9,
        mode='lines',
        name='IC 95%',
        fill='tonexty',
        fillcolor='rgba(255, 107, 53, 0.2)',
        line=dict(width=0)
    ))
    
    # Línea vertical - usar add_shape en lugar de add_vline
    fig.add_shape(
        type="line",
        x0=fecha_max,
        x1=fecha_max,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=fecha_max,
        y=1,
        yref="paper",
        text="Hoy",
        showarrow=False,
        yanchor="bottom"
    )
    
    fig.update_layout(
        title="Pronóstico de Net New Money (NNM) - Próximas 52 Semanas",
        xaxis_title="Fecha",
        yaxis_title="NNM (Miles de Millones $)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Componentes
    st.markdown("#### 📊 Componentes de la Serie Temporal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tendencia
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['trend']/1e9,
            mode='lines',
            name='Tendencia',
            line=dict(color='#3498db', width=2)
        ))
        fig_trend.update_layout(
            title="Tendencia",
            xaxis_title="Fecha",
            yaxis_title="NNM (Miles de Millones $)",
            height=300
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Estacionalidad
        if 'yearly' in forecast.columns:
            fig_season = go.Figure()
            fig_season.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yearly']/1e6,
                mode='lines',
                name='Estacionalidad Anual',
                line=dict(color='#e74c3c', width=2)
            ))
            fig_season.update_layout(
                title="Estacionalidad Anual",
                xaxis_title="Fecha",
                yaxis_title="Efecto (Millones $)",
                height=300
            )
            st.plotly_chart(fig_season, use_container_width=True)
    
    # Tabla de pronóstico
    st.markdown("#### 📋 Tabla de Pronóstico (Próximos 3 Meses)")
    forecast_table = forecast_fut.head(12)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_table.columns = ['Fecha', 'Pronóstico', 'Límite Inferior', 'Límite Superior']
    forecast_table['Fecha'] = forecast_table['Fecha'].dt.strftime('%Y-%m-%d')
    forecast_table[['Pronóstico', 'Límite Inferior', 'Límite Superior']] = \
        forecast_table[['Pronóstico', 'Límite Inferior', 'Límite Superior']].applymap(lambda x: f"${x/1e6:.2f}M")
    
    st.dataframe(forecast_table, use_container_width=True)

def render_clients_forecast(clients):
    """Pronóstico de nuevos clientes"""
    st.markdown("### 👥 Pronóstico de Nuevos Clientes")
    
    with st.spinner("Entrenando modelo Prophet para nuevos clientes..."):
        model, forecast = train_prophet_clients(clients)
    
    if model is None or forecast is None:
        st.error("Error al entrenar el modelo")
        return
    
    # Métricas
    clients_copy = clients.copy()
    clients_copy['week'] = clients_copy['registration_date'].dt.to_period('W').dt.to_timestamp()
    fecha_max = pd.Timestamp(clients_copy['week'].max())
    
    forecast_futuro = forecast[forecast['ds'] > fecha_max]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_weekly = forecast_futuro['yhat'].mean()
        st.metric("Clientes Semanales Promedio", f"{avg_weekly:.0f}")
    with col2:
        total_yearly = forecast_futuro['yhat'].sum()
        st.metric("Total Nuevos Clientes (52 sem)", f"{total_yearly:.0f}")
    with col3:
        current_clients = forecast[forecast['ds'] <= fecha_max]['yhat'].tail(4).mean()
        growth = ((avg_weekly - current_clients) / current_clients * 100) if current_clients != 0 else 0
        st.metric("Crecimiento vs Actual", f"{growth:+.1f}%")
    
    st.markdown("---")
    
    # Gráfico de pronóstico
    st.markdown("#### 📈 Pronóstico Nuevos Clientes - 52 Semanas")
    
    registros_semana = clients_copy.groupby('week').size().reset_index()
    registros_semana.columns = ['ds', 'y']
    
    forecast_hist = forecast[forecast['ds'] <= fecha_max]
    forecast_fut = forecast[forecast['ds'] > fecha_max]
    
    fig = go.Figure()
    
    # Histórico real
    fig.add_trace(go.Scatter(
        x=registros_semana['ds'],
        y=registros_semana['y'],
        mode='markers',
        name='Histórico Real',
        marker=dict(size=6, color='#2E86AB')
    ))
    
    # Ajuste del modelo
    fig.add_trace(go.Scatter(
        x=forecast_hist['ds'],
        y=forecast_hist['yhat'],
        mode='lines',
        name='Ajuste Modelo',
        line=dict(color='#06A77D', width=2)
    ))
    
    # Pronóstico
    fig.add_trace(go.Scatter(
        x=forecast_fut['ds'],
        y=forecast_fut['yhat'],
        mode='lines+markers',
        name='Pronóstico',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=5)
    ))
    
    # Intervalo de confianza
    fig.add_trace(go.Scatter(
        x=forecast_fut['ds'],
        y=forecast_fut['yhat_upper'],
        mode='lines',
        name='IC Superior',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_fut['ds'],
        y=forecast_fut['yhat_lower'],
        mode='lines',
        name='IC 95%',
        fill='tonexty',
        fillcolor='rgba(255, 107, 53, 0.2)',
        line=dict(width=0)
    ))
    
    # Línea vertical - usar add_shape en lugar de add_vline
    fig.add_shape(
        type="line",
        x0=fecha_max,
        x1=fecha_max,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=fecha_max,
        y=1,
        yref="paper",
        text="Hoy",
        showarrow=False,
        yanchor="bottom"
    )
    
    fig.update_layout(
        title="Pronóstico de Nuevos Clientes - Próximas 52 Semanas",
        xaxis_title="Fecha",
        yaxis_title="Nuevos Clientes",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de pronóstico
    st.markdown("#### 📋 Tabla de Pronóstico (Próximos 3 Meses)")
    forecast_table = forecast_fut.head(12)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_table.columns = ['Fecha', 'Pronóstico', 'Límite Inferior', 'Límite Superior']
    forecast_table['Fecha'] = forecast_table['Fecha'].dt.strftime('%Y-%m-%d')
    forecast_table[['Pronóstico', 'Límite Inferior', 'Límite Superior']] = \
        forecast_table[['Pronóstico', 'Límite Inferior', 'Límite Superior']].round(0).astype(int)
    
    st.dataframe(forecast_table, use_container_width=True)