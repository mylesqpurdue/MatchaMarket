"""
Responsive design utilities for the Stock Market Dashboard.
This module provides functions for creating responsive layouts.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

def create_responsive_container(children, fluid=True, className=""):
    """
    Create a responsive container for dashboard content.
    
    Args:
        children: Child components to include in the container
        fluid (bool, optional): Whether to use a fluid container
        className (str, optional): Additional CSS classes
        
    Returns:
        dash component: Responsive container
    """
    container_class = f"container-fluid {className}" if fluid else f"container {className}"
    return html.Div(children, className=container_class)

def create_responsive_row(children, className=""):
    """
    Create a responsive row for dashboard content.
    
    Args:
        children: Child components to include in the row
        className (str, optional): Additional CSS classes
        
    Returns:
        dash component: Responsive row
    """
    return dbc.Row(children, className=f"row {className}")

def create_responsive_col(children, xs=12, sm=None, md=None, lg=None, xl=None, className=""):
    """
    Create a responsive column for dashboard content.
    
    Args:
        children: Child components to include in the column
        xs, sm, md, lg, xl (int, optional): Column widths at different breakpoints
        className (str, optional): Additional CSS classes
        
    Returns:
        dash component: Responsive column
    """
    return dbc.Col(
        children,
        xs=xs,
        sm=sm,
        md=md,
        lg=lg,
        xl=xl,
        className=className
    )

def create_card_deck(cards, deck_className="", card_className=""):
    """
    Create a responsive card deck.
    
    Args:
        cards (list): List of card components
        deck_className (str, optional): Additional CSS classes for the deck
        card_className (str, optional): Additional CSS classes for each card wrapper
        
    Returns:
        dash component: Card deck
    """
    # Wrap each card in a column for responsive behavior
    card_columns = [
        create_responsive_col(card, xs=12, sm=6, lg=4, className=card_className)
        for card in cards
    ]
    
    return create_responsive_row(card_columns, className=f"card-deck {deck_className}")

def create_collapsible_section(title, content, id_prefix, open=True):
    """
    Create a collapsible section for the dashboard.
    
    Args:
        title (str): Section title
        content: Content to display in the section
        id_prefix (str): Prefix for component IDs
        open (bool, optional): Whether the section is initially open
        
    Returns:
        dash component: Collapsible section
    """
    return html.Div([
        html.Div([
            html.H5(title, className="mb-0"),
            html.Button(
                "▼" if open else "▶",
                id=f"{id_prefix}-collapse-button",
                className="btn btn-link collapse-button"
            )
        ], className="d-flex justify-content-between align-items-center"),
        dbc.Collapse(
            content,
            id=f"{id_prefix}-collapse",
            is_open=open
        )
    ], className="collapsible-section")

def create_tab_layout(tabs, id_prefix):
    """
    Create a responsive tab layout.
    
    Args:
        tabs (list): List of (tab_label, tab_content) tuples
        id_prefix (str): Prefix for component IDs
        
    Returns:
        dash component: Tab layout
    """
    return html.Div([
        dbc.Tabs(
            [
                dbc.Tab(content, label=label, tab_id=f"{id_prefix}-tab-{i}")
                for i, (label, content) in enumerate(tabs)
            ],
            id=f"{id_prefix}-tabs",
            active_tab=f"{id_prefix}-tab-0"
        )
    ], className="responsive-tabs")

def create_mobile_menu(items):
    """
    Create a mobile-friendly menu.
    
    Args:
        items (list): List of (label, callback_id) tuples
        
    Returns:
        dash component: Mobile menu
    """
    return html.Div([
        dbc.Button(
            html.I(className="fas fa-bars"),
            id="mobile-menu-button",
            color="primary",
            className="d-md-none"  # Only visible on small screens
        ),
        dbc.Collapse(
            dbc.Card([
                dbc.CardBody([
                    html.A(
                        label,
                        id=callback_id,
                        className="mobile-menu-item"
                    )
                    for label, callback_id in items
                ])
            ]),
            id="mobile-menu-collapse",
            is_open=False
        )
    ], className="mobile-menu d-md-none")  # Only visible on small screens
