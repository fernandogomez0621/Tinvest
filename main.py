# -*- coding: utf-8 -*-
"""
Tinvest Analytics Dashboard
Main Application File
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Tinvest Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar módulos
try:
    from modules import data_loader
    from modules import eda
    from modules import cohort_analysis
    from modules import rfm_segmentation
    from modules import churn_prediction
    from modules import forecasting
    from modules import executive_summary
except ImportError as e:
    st.error(f"Error al importar módulos: {e}")
    st.info("Asegúrate de que la carpeta 'modules' existe con todos los archivos .py")
    st.stop()

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">💰 Tinvest Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=TINVEST", use_container_width=True)
    st.markdown("## 📊 Panel de Control")
    st.markdown("---")
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        try:
            clients, transactions, portfolio, fecha_corte = data_loader.load_data()
            st.success("✅ Datos cargados correctamente")
            
            # Métricas rápidas
            st.markdown("### 📈 Métricas Rápidas")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Clientes", f"{len(clients):,}")
            with col2:
                st.metric("Transacciones", f"{len(transactions):,}")
        except Exception as e:
            st.error(f"❌ Error al cargar datos: {e}")
            st.info("Asegúrate de tener los archivos: clients.csv, transactions.csv, portfolio_balance.csv")
            st.stop()
    
    st.markdown("---")
    st.markdown("### ℹ️ Información")
    st.info(f"Fecha de corte: {fecha_corte.strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    st.markdown("### 🔄 Actualizar")
    if st.button("🔄 Recargar Datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Tabs principales
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Resumen Ejecutivo",
    "🔍 EDA",
    "👥 Cohortes",
    "💎 Segmentación RFM",
    "⚠️ Predicción Churn",
    "🔮 Forecasting",
    "📥 Exportar"
])

# Tab 1: Resumen Ejecutivo
with tab1:
    executive_summary.render(clients, transactions, portfolio, fecha_corte)

# Tab 2: EDA
with tab2:
    eda.render(clients, transactions, portfolio)

# Tab 3: Análisis de Cohortes
with tab3:
    cohort_analysis.render(clients, transactions, fecha_corte)

# Tab 4: Segmentación RFM
with tab4:
    rfm_segmentation.render(clients, transactions, fecha_corte)

# Tab 5: Predicción de Churn
with tab5:
    churn_prediction.render(clients, transactions, portfolio, fecha_corte)

# Tab 6: Forecasting
with tab6:
    forecasting.render(transactions, clients)

# Tab 7: Exportar
with tab7:
    st.markdown("## 📥 Exportar Resultados")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Datos Procesados")
        if st.button("📥 Descargar RFM Segmentation", use_container_width=True):
            try:
                rfm_data = pd.read_csv('rfm_segmentation.csv')
                csv = rfm_data.to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name='rfm_segmentation.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            except:
                st.warning("Archivo no encontrado. Ejecuta primero el análisis RFM.")
        
        if st.button("📥 Descargar Churn Predictions", use_container_width=True):
            try:
                churn_data = pd.read_csv('churn_predictions.csv')
                csv = churn_data.to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name='churn_predictions.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            except:
                st.warning("Archivo no encontrado. Ejecuta primero el modelo de churn.")
    
    with col2:
        st.markdown("### 📈 Reportes")
        if st.button("📄 Generar Reporte PDF", use_container_width=True):
            st.info("Funcionalidad en desarrollo...")
        
        if st.button("📊 Generar Reporte Excel", use_container_width=True):
            st.info("Funcionalidad en desarrollo...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>Tinvest Analytics Dashboard v1.0 | Powered by Streamlit 🚀</p>
    <p>© 2024 Tinvest Financial Analytics</p>
</div>
""", unsafe_allow_html=True)