# -*- coding: utf-8 -*-
"""
RFM Segmentation Module
Segmentación RFM de clientes
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

@st.cache_data
def calculate_rfm(clients, transactions, fecha_corte):
    """Calcula las métricas RFM"""
    # Recency
    recency = transactions.groupby('client_id')['date'].max().reset_index()
    recency.columns = ['client_id', 'ultima_fecha']
    recency['recency'] = (fecha_corte - recency['ultima_fecha']).dt.days
    
    # Frequency
    frequency = transactions.groupby('client_id').size().reset_index()
    frequency.columns = ['client_id', 'frequency']
    
    # Monetary
    monetary = transactions[transactions['type'] == 'deposit'].groupby('client_id')['amount'].sum().reset_index()
    monetary.columns = ['client_id', 'monetary']
    
    # Combinar
    rfm = recency[['client_id', 'recency']].merge(frequency, on='client_id', how='left')
    rfm = rfm.merge(monetary, on='client_id', how='left')
    rfm = rfm.merge(clients[['client_id', 'segment', 'registration_date']], on='client_id', how='left')
    rfm['monetary'] = rfm['monetary'].fillna(0)
    
    # Scores
    rfm['R_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
    rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
    rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    
    # Segmentación
    rfm['segmento_rfm'] = segmentar_rfm(rfm, fecha_corte)
    
    return rfm

def segmentar_rfm(df, fecha_corte):
    """Segmenta clientes según RFM"""
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

def render(clients, transactions, fecha_corte):
    """Renderiza el análisis RFM"""
    st.markdown("## 💎 Segmentación RFM")
    st.markdown("---")
    
    # Calcular RFM
    with st.spinner("Calculando segmentación RFM..."):
        rfm = calculate_rfm(clients, transactions, fecha_corte)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        campeones = len(rfm[rfm['segmento_rfm'] == 'Campeones'])
        st.metric("Campeones", f"{campeones:,}", f"{campeones/len(rfm)*100:.1f}%")
    with col2:
        en_riesgo = len(rfm[rfm['segmento_rfm'] == 'En Riesgo'])
        st.metric("En Riesgo", f"{en_riesgo:,}", f"{en_riesgo/len(rfm)*100:.1f}%")
    with col3:
        perdidos = len(rfm[rfm['segmento_rfm'] == 'Perdidos'])
        st.metric("Perdidos", f"{perdidos:,}", f"{perdidos/len(rfm)*100:.1f}%")
    with col4:
        valor_total = rfm['monetary'].sum()
        st.metric("Valor Total", f"${valor_total/1e9:.2f}B")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribución", "💰 Valor", "🔍 Análisis Detallado", "👥 Clientes"])
    
    with tab1:
        render_distribution(rfm)
    
    with tab2:
        render_value_analysis(rfm)
    
    with tab3:
        render_detailed_rfm(rfm)
    
    with tab4:
        render_client_list(rfm)

def render_distribution(rfm):
    """Distribución de segmentos"""
    st.markdown("### 📊 Distribución de Segmentos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segmento_counts = rfm['segmento_rfm'].value_counts()
        colors_dict = {
            'Campeones': '#2ecc71', 'Leales': '#27ae60', 'Potenciales': '#3498db',
            'Promedio': '#95a5a6', 'Necesitan Atención': '#f39c12', 'En Riesgo': '#e67e22',
            'Hibernando': '#9b59b6', 'Perdidos': '#e74c3c', 'Nuevos': '#1abc9c'
        }
        colors = [colors_dict.get(seg, '#95a5a6') for seg in segmento_counts.index]
        
        fig = px.pie(
            values=segmento_counts.values,
            names=segmento_counts.index,
            title="Distribución de Clientes por Segmento",
            color_discrete_sequence=colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=segmento_counts.index,
            y=segmento_counts.values,
            title="Cantidad de Clientes por Segmento",
            labels={'x': 'Segmento', 'y': 'Clientes'},
            color=segmento_counts.index,
            color_discrete_map=colors_dict
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribución RFM
    st.markdown("#### Distribución de Scores RFM")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(rfm, x='R_score', title="Recency Score", 
                          color_discrete_sequence=['#e74c3c'], nbins=5)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(rfm, x='F_score', title="Frequency Score",
                          color_discrete_sequence=['#3498db'], nbins=5)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(rfm, x='M_score', title="Monetary Score",
                          color_discrete_sequence=['#2ecc71'], nbins=5)
        st.plotly_chart(fig, use_container_width=True)

def render_value_analysis(rfm):
    """Análisis de valor por segmento"""
    st.markdown("### 💰 Análisis de Valor")
    
    # Valor por segmento
    valor_segmento = rfm.groupby('segmento_rfm').agg({
        'monetary': ['sum', 'mean', 'count']
    }).reset_index()
    valor_segmento.columns = ['Segmento', 'Valor Total', 'Valor Promedio', 'Clientes']
    valor_segmento = valor_segmento.sort_values('Valor Total', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            valor_segmento,
            x='Segmento',
            y='Valor Total',
            title="Valor Total por Segmento",
            labels={'Valor Total': 'Valor Total ($)'},
            color='Valor Total',
            color_continuous_scale='Greens'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            valor_segmento,
            x='Segmento',
            y='Valor Promedio',
            title="Valor Promedio por Cliente",
            labels={'Valor Promedio': 'Valor Promedio ($)'},
            color='Valor Promedio',
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla de valor
    st.markdown("#### 📋 Tabla de Valor por Segmento")
    valor_segmento['Valor Total'] = valor_segmento['Valor Total'].apply(lambda x: f"${x/1e6:.2f}M")
    valor_segmento['Valor Promedio'] = valor_segmento['Valor Promedio'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(valor_segmento, use_container_width=True)
    
    # Pareto
    st.markdown("#### 📊 Análisis Pareto (80/20)")
    rfm_sorted = rfm.sort_values('monetary', ascending=False).reset_index(drop=True)
    rfm_sorted['cumulative_value'] = rfm_sorted['monetary'].cumsum()
    rfm_sorted['cumulative_pct'] = rfm_sorted['cumulative_value'] / rfm_sorted['monetary'].sum() * 100
    rfm_sorted['client_pct'] = (rfm_sorted.index + 1) / len(rfm_sorted) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rfm_sorted['client_pct'],
        y=rfm_sorted['cumulative_pct'],
        mode='lines',
        name='Valor Acumulado',
        line=dict(color='#2ecc71', width=3)
    ))
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="80% del valor")
    fig.update_layout(
        title="Curva de Pareto - Distribución del Valor",
        xaxis_title="% de Clientes",
        yaxis_title="% Valor Acumulado",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Encontrar el punto 80/20
    idx_80 = (rfm_sorted['cumulative_pct'] >= 80).idxmax()
    pct_clientes_80 = rfm_sorted.loc[idx_80, 'client_pct']
    st.info(f"El **{pct_clientes_80:.1f}%** de los clientes generan el **80%** del valor total")

def render_detailed_rfm(rfm):
    """Análisis detallado RFM"""
    st.markdown("### 🔍 Análisis Detallado")
    
    # Scatter 3D
    st.markdown("#### 📈 Visualización 3D: R-F-M")
    
    fig = px.scatter_3d(
        rfm.sample(min(1000, len(rfm))),
        x='recency',
        y='frequency',
        z='monetary',
        color='segmento_rfm',
        title="Distribución RFM en 3D (muestra de 1000 clientes)",
        labels={'recency': 'Recency (días)', 'frequency': 'Frequency', 'monetary': 'Monetary ($)'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlaciones
    st.markdown("#### 📊 Correlaciones RFM")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            rfm,
            x='frequency',
            y='monetary',
            color='segmento_rfm',
            title="Frequency vs Monetary",
            labels={'frequency': 'Frequency', 'monetary': 'Monetary ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            rfm,
            x='recency',
            y='monetary',
            color='segmento_rfm',
            title="Recency vs Monetary",
            labels={'recency': 'Recency (días)', 'monetary': 'Monetary ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Estadísticas por segmento
    st.markdown("#### 📋 Estadísticas por Segmento")
    stats = rfm.groupby('segmento_rfm')[['recency', 'frequency', 'monetary']].agg(['mean', 'median']).round(2)
    st.dataframe(stats, use_container_width=True)

def render_client_list(rfm):
    """Lista de clientes por segmento"""
    st.markdown("### 👥 Explorador de Clientes")
    
    # Selector de segmento
    segmentos = ['Todos'] + list(rfm['segmento_rfm'].unique())
    selected_segment = st.selectbox("Selecciona un segmento", segmentos)
    
    # Filtrar
    if selected_segment != 'Todos':
        rfm_filtered = rfm[rfm['segmento_rfm'] == selected_segment]
    else:
        rfm_filtered = rfm
    
    st.info(f"Mostrando {len(rfm_filtered):,} clientes")
    
    # Ordenar por
    sort_by = st.selectbox("Ordenar por", ['RFM_score', 'monetary', 'frequency', 'recency'])
    ascending = st.checkbox("Orden ascendente", value=False)
    
    rfm_display = rfm_filtered.sort_values(sort_by, ascending=ascending)[
        ['client_id', 'segment', 'segmento_rfm', 'recency', 'frequency', 'monetary', 
         'R_score', 'F_score', 'M_score', 'RFM_score']
    ].head(100)
    
    # Formatear
    rfm_display_formatted = rfm_display.copy()
    rfm_display_formatted['monetary'] = rfm_display_formatted['monetary'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(rfm_display_formatted, use_container_width=True)
    
    # Botón de descarga
    csv = rfm_filtered.to_csv(index=False)
    st.download_button(
        label="📥 Descargar segmento completo (CSV)",
        data=csv,
        file_name=f'rfm_{selected_segment.lower().replace(" ", "_")}.csv',
        mime='text/csv',
        use_container_width=True
    )