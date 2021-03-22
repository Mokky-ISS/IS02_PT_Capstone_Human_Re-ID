import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from demo_dash import blocks

external_stylesheets = [
    dbc.themes.COSMO,
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    "https://use.fontawesome.com/releases/v5.7.2/css/all.css",
]

app = dash.Dash(
    __name__, title="RE-ID Dash Demo",
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    meta_tags=[{
        'name': 'viewport',
        'content': 'width=device-width, initial-scale=1.0'
    }])

app.layout = dbc.Container(
    id='app-layout',
    children=[
        dcc.Location(id='url', refresh=False),
        blocks.title_block,
        html.Hr(style={'margin-top': 0, 'margin-bottom': 0}),
        html.Div(id='page-content', children=[])
    ],
    fluid=True
)