"""
Reusable metrics card components for SeeSense Dashboard
"""
import streamlit as st
from typing import Optional, Union


def metric_card_html(title: str, value: str, delta: Optional[str] = None, 
                    delta_color: str = "normal", help_text: Optional[str] = None) -> str:
    """
    Create HTML for a custom metric card
    
    Args:
        title: Card title
        value: Main metric value
        delta: Change indicator (e.g., "+5%" or "-2%")
        delta_color: Color of delta ("positive", "negative", "normal")
        help_text: Optional help text
        
    Returns:
        HTML string for the metric card
    """
    delta_colors = {
        "positive": "#28a745",
        "negative": "#dc3545", 
        "normal": "#6c757d"
    }
    
    delta_html = ""
    if delta:
        color = delta_colors.get(delta_color, "#6c757d")
        delta_html = f'<div style="color: {color}; font-size: 14px; font-weight: 500; margin-top: 4px;">{delta}</div>'
    
    help_html = ""
    if help_text:
        help_html = f'<div style="color: #6c757d; font-size: 12px; margin-top: 4px;">{help_text}</div>'
    
    return f"""
    <div style="
        background-color: #ffffff;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 16px;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    ">
        <div style="
            color: #6c757d;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 8px;
        ">{title}</div>
        <div style="
            color: #333;
            font-size: 28px;
            font-weight: 700;
            line-height: 1.2;
        ">{value}</div>
        {delta_html}
        {help_html}
    </div>
    """


def render_metric_grid(metrics: list, columns: int = 4):
    """
    Render a grid of metrics cards
    
    Args:
        metrics: List of metric dictionaries with keys: title, value, delta, delta_color, help_text
        columns: Number of columns in the grid
    """
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        col_idx = i % columns
        with cols[col_idx]:
            delta_color = "normal"
            if metric.get('delta'):
                # Auto-detect delta color from delta string
                if metric['delta'].startswith('+'):
                    delta_color = "positive"
                elif metric['delta'].startswith('-'):
                    delta_color = "negative"
                else:
                    delta_color = "normal"
            
            card_html = metric_card_html(
                title=metric['title'],
                value=metric['value'],
                delta=metric.get('delta'),
                delta_color=delta_color,
                help_text=metric.get('help_text')
            )
            st.markdown(card_html, unsafe_allow_html=True)


def status_indicator(status: str, label: str) -> str:
    """
    Create a status indicator
    
    Args:
        status: Status level ("success", "warning", "error", "info")
        label: Status label text
        
    Returns:
        HTML string for status indicator
    """
    colors = {
        "success": "#28a745",
        "warning": "#ffc107", 
        "error": "#dc3545",
        "info": "#17a2b8"
    }
    
    icons = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌", 
        "info": "ℹ️"
    }
    
    color = colors.get(status, "#6c757d")
    icon = icons.get(status, "•")
    
    return f"""
    <div style="
        display: inline-flex;
        align-items: center;
        background-color: {color}20;
        border: 1px solid {color}50;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 14px;
        font-weight: 500;
        color: {color};
        margin: 2px;
    ">
        <span style="margin-right: 6px;">{icon}</span>
        {label}
    </div>
    """


def kpi_summary_card(title: str, primary_kpi: dict, secondary_kpis: list = None) -> str:
    """
    Create a KPI summary card with primary and secondary metrics
    
    Args:
        title: Card title
        primary_kpi: Dict with 'value', 'label', 'delta' keys
        secondary_kpis: List of secondary KPI dicts
        
    Returns:
        HTML string for KPI summary card
    """
    secondary_html = ""
    if secondary_kpis:
        secondary_items = []
        for kpi in secondary_kpis:
            delta_html = ""
            if kpi.get('delta'):
                delta_color = "#28a745" if kpi['delta'].startswith('+') else "#dc3545"
                delta_html = f'<span style="color: {delta_color}; font-size: 12px; margin-left: 4px;">{kpi["delta"]}</span>'
            
            secondary_items.append(f"""
                <div style="text-align: center; padding: 8px;">
                    <div style="font-size: 18px; font-weight: 600; color: #333;">{kpi['value']}</div>
                    <div style="font-size: 12px; color: #6c757d;">{kpi['label']}{delta_html}</div>
                </div>
            """)
        
        secondary_html = f"""
            <div style="
                display: flex;
                justify-content: space-around;
                border-top: 1px solid #e0e0e0;
                margin-top: 12px;
                padding-top: 12px;
            ">
                {''.join(secondary_items)}
            </div>
        """
    
    primary_delta = ""
    if primary_kpi.get('delta'):
        delta_color = "#28a745" if primary_kpi['delta'].startswith('+') else "#dc3545"
        primary_delta = f'<div style="color: {delta_color}; font-size: 16px; font-weight: 500; margin-top: 4px;">{primary_kpi["delta"]}</div>'
    
    return f"""
    <div style="
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 16px;
    ">
        <div style="
            color: #6c757d;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
        ">{title}</div>
        
        <div style="text-align: center; margin-bottom: 12px;">
            <div style="font-size: 36px; font-weight: 700; color: #333;">{primary_kpi['value']}</div>
            <div style="font-size: 14px; color: #6c757d;">{primary_kpi['label']}</div>
            {primary_delta}
        </div>
        
        {secondary_html}
    </div>
    """


def trend_indicator(value: float, trend: str = "neutral") -> str:
    """
    Create a trend indicator with arrow
    
    Args:
        value: Numeric value
        trend: Trend direction ("up", "down", "neutral")
        
    Returns:
        HTML string for trend indicator
    """
    arrows = {
        "up": "↗️",
        "down": "↘️", 
        "neutral": "➡️"
    }
    
    colors = {
        "up": "#28a745",
        "down": "#dc3545",
        "neutral": "#6c757d"
    }
    
    arrow = arrows.get(trend, "➡️")
    color = colors.get(trend, "#6c757d")
    
    return f"""
    <span style="
        display: inline-flex;
        align-items: center;
        color: {color};
        font-weight: 600;
        font-size: 14px;
    ">
        <span style="margin-right: 4px;">{arrow}</span>
        {value}
    </span>
    """
