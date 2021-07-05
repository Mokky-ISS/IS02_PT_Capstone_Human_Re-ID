import dash_html_components as html
import dash_bootstrap_components as dbc

title_text = 'Merging Phase 2'

title = dbc.Col(
    id='title-col',
    children=[
        dbc.Col([
            html.H1(
                id='label-title',
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
                    dbc.NavLink(id='view1-link',
                                children='View 1', href='/view1'),
                    dbc.Tooltip(
                        children='Detailed view of any individual (Search by human ID)',
                        target='view1-link',
                        placement='auto-start',
                        style={'font-size': '150%'},
                    ),
                ], width='auto'),
                dbc.Col([
                    dbc.NavLink(id='view2-link', children='View 2',
                                href='/view2'),
                    dbc.Tooltip(
                        children='General view of all individual/identities (Search by channel and/or timestamp)',
                        target='view2-link',
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
        #sp_logo,
        title,
        #navbar_views,
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
