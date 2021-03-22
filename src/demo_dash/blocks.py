import dash_html_components as html
import dash_bootstrap_components as dbc

sp_logo = dbc.Col(
    id='ai-singapore-col',
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

title_text = 'Re-Identification Demo'

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

title_block = dbc.Row(
    id='title-block-row',
    children=[
        sp_logo,
        title,
        #navbar_real_batch,
    ],
    no_gutters=True,
)

