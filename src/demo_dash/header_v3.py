import dash_html_components as html
import dash_bootstrap_components as dbc


sp_logo = dbc.Col(
    id='logo-col',
    children=[
        dbc.NavLink(
            id='sp-group-link',
            children=html.Img(
                src="https://www.spgroup.com.sg/www/theme/logo.png",
                title='SP Group',
                style={'background-color': '#0f71a1'}),
            href="https://www.spgroup.com.sg/home",
        ),
    ],
    width='auto',
    align='start',
)

title_text = 'Re-Identification'

title = dbc.Col(
    id='title-col',
    children=[
        dbc.Col([
            html.H1(
                id='re-id-title',
                children=title_text,
                style={
                    'text-align': 'center',
                    'font-weight': 'bold',
                }
            ),
        ]),
    ],
    width=True,
    align='center',
)

navbar_views = dbc.Col(
    id='navbar-views-col',
    children=dbc.Navbar(
        id='navbar-views',
        children=[
            dbc.Row([
                dbc.Col([
                    dbc.NavLink(id='home-link',
                                children='Home', href='/'),
                    dbc.Tooltip(
                        children='Select image for query',
                        target='home-link',
                        placement='auto-start',
                        style={'font-size': '150%'},
                    ),
                ], width='auto'),
            ], align='stretch', justify='between'),
        ],
        color="light",
        light=True,
    ),
    width='auto',
    align='start'
)

title_block = dbc.Row(
    id='title-block-row',
    children=[
        sp_logo,
        title,
        navbar_views,
    ],
    no_gutters=True,
)

subtitle = dbc.Col(
    id='subtitle-col',
    children=[
        dbc.Col([
            html.H4(
                id='page-subtitle',
                children='',
                style={
                    'text-align': 'center',
                }
            ),
        ]),
    ],
    width=True,
    align='center',
)




