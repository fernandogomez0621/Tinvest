# -*- coding: utf-8 -*-
"""
Data Loader Module
Carga y valida los datos de Tinvest
"""

import streamlit as st
import pandas as pd
from datetime import datetime

@st.cache_data
def load_data():
    """Carga los datos principales de Tinvest"""
    try:
        # Cargar CSVs
        clients = pd.read_csv('clients.csv')
        transactions = pd.read_csv('transactions.csv')
        portfolio = pd.read_csv('portfolio_balance.csv')
        
        # Convertir fechas
        clients['registration_date'] = pd.to_datetime(clients['registration_date'])
        transactions['date'] = pd.to_datetime(transactions['date'])
        portfolio['date'] = pd.to_datetime(portfolio['date'])
        
        # Fecha de corte
        fecha_corte = transactions['date'].max()
        
        return clients, transactions, portfolio, fecha_corte
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Archivo no encontrado: {e}")
    except Exception as e:
        raise Exception(f"Error al cargar datos: {e}")

def get_data_summary(clients, transactions, portfolio):
    """Retorna un resumen de los datos cargados"""
    summary = {
        'clients': {
            'count': len(clients),
            'columns': list(clients.columns),
            'date_range': (clients['registration_date'].min(), clients['registration_date'].max())
        },
        'transactions': {
            'count': len(transactions),
            'columns': list(transactions.columns),
            'date_range': (transactions['date'].min(), transactions['date'].max())
        },
        'portfolio': {
            'count': len(portfolio),
            'columns': list(portfolio.columns),
            'date_range': (portfolio['date'].min(), portfolio['date'].max())
        }
    }
    return summary

def validate_data(clients, transactions, portfolio):
    """Valida la calidad de los datos"""
    issues = []
    
    # Validar duplicados
    if clients.duplicated().sum() > 0:
        issues.append(f"⚠️ {clients.duplicated().sum()} clientes duplicados")
    if transactions.duplicated().sum() > 0:
        issues.append(f"⚠️ {transactions.duplicated().sum()} transacciones duplicadas")
    
    # Validar nulos
    for name, df in [("Clients", clients), ("Transactions", transactions), ("Portfolio", portfolio)]:
        nulos = df.isnull().sum()
        if nulos.sum() > 0:
            issues.append(f"⚠️ {name}: {nulos.sum()} valores nulos")
    
    # Validar integridad referencial
    client_ids_trans = set(transactions['client_id'].unique())
    client_ids_clients = set(clients['client_id'].unique())
    missing_clients = client_ids_trans - client_ids_clients
    if missing_clients:
        issues.append(f"⚠️ {len(missing_clients)} client_ids en transactions no existen en clients")
    
    return issues if issues else ["✅ Datos validados correctamente"]