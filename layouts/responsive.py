import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

def create_responsive_container(children, fluid=True, className=""):
    container_class = f"container-fluid {className}" if fluid else f"container {className}"
    return html.Div(children, className=container_class)

def create_responsive_row(children, className=""):
    return dbc.Row(children, className=f"row {className}")

def create_responsive_col(children, xs=12, sm=None, md=None, lg=None, xl=None, className=""):
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
    # Wrap each card in a column for responsive behavior
    card_columns = [
        create_responsive_col(card, xs=12, sm=6, lg=4, className=card_className)
        for card in cards
    ]
    
    return create_responsive_row(card_columns, className=f"card-deck {deck_className}")

def create_collapsible_section(title, content, id_prefix, open=True):
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
