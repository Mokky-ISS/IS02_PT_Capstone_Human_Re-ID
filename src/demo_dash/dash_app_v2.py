import logging
import os
import dash
from dash_bootstrap_components._components.Row import Row
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from demo_dash import header_v2 as header
from database import query_image, query_re_id
from urllib.parse import parse_qs
import base64
import cv2
import mediapipe as mp

dbimage = query_image.DbQuery()
dbreid = query_re_id.DbQuery()

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
        header.title_block,
        header.subtitle,
        html.Hr(),
        dbc.Container(id='page-content', fluid=True),
    ],
    fluid=True,
)

@app.callback(
    Output(component_id='page-content', component_property='children'),
    Input(component_id='url', component_property='pathname'),
    Input(component_id='url', component_property='search')
)
def display_page(pathname, search):
    if search is not None:
        queries = parse_qs(search[1:])
        logging.info(queries)
    else:
        queries = None
    print(pathname)

    layout_page = []
    try:
        if pathname is not None and 'view2' in pathname:
            layout_page.append(
                view2_page_content())
        else:
            layout_page.append(
                view1_page_content())
    except Exception as ex:
        logging.error(ex)
    return layout_page  # , title



SIDEBAR_STYLE = {
    "position": "static",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    'height': '100%',
}

CAMERA_LOCATION_OPTIONS = [
    {'label': 'Camera 1', 'value': 1},
    {'label': 'Camera 2', 'value': 2},
    {'label': 'Camera 3', 'value': 3},
    {'label': 'Camera 4', 'value': 4},
    {'label': 'Camera 5', 'value': 5},
]

def view1_page_content():
    sidebar = dbc.Col(
        id='view1-page-sidebar',
        children=[
            html.P('Human ID:'),
            dbc.Spinner(dcc.Dropdown(
                id='view1-human-id',
                options=dbreid.get_human_id_options(),
            )),
            html.Br(),
            dbc.Row([
                dbc.Col(html.P('Time:'), width=2),
                dbc.Col(dcc.DatePickerRange(id='view1-datetime'), width='auto'),
                ]),
            html.Br(),
            html.P('Camera/location:'),
            dbc.Spinner(dcc.Dropdown(id='view1-cam-loc-dropdown', options=CAMERA_LOCATION_OPTIONS)),
            html.Br(),
            dbc.Button(children='Refresh', id='view1-btn-refresh',
                       color='primary', block=True, size="lg"),
        ],
        width=2,
        style=SIDEBAR_STYLE,
    )

    return dbc.Row(children=[
        sidebar,
        dbc.Col(
            id='display-col',
            children=[
                html.H5('Images:'),
                dbc.Spinner(dbc.Row(
                    id='view1-images-row',
                    form=True,
                    style={'flex-wrap': 'nowrap', 'overflow':'auto'})),
                html.Br(),
                html.H5('Logs:'),
                dbc.Spinner(dbc.Row(
                    id='logs_row',
                    children=
                        dcc.Textarea(
                            id='view1-logs-text',
                            disabled=True,
                            style={'width': '100%','height': '200px'}
                            ),
                        )),
                ],
            width=10,
        )
    ])


def view2_page_content():
    sidebar = dbc.Col(
        id='view2-page-sidebar',
        children=[
            html.P('Camera/location:'),
            dbc.Spinner(dcc.Dropdown(id='view2-cam-loc-dropdown',
                                     options=CAMERA_LOCATION_OPTIONS)),
            html.Br(),
            dbc.Row([
                dbc.Col(html.P('Time:'), width=2),
                dbc.Col(dcc.DatePickerRange(
                    id='view2-datetime'), width='auto'),
            ]),
            html.Br(),
            dbc.Button(children='Refresh', id='view2-btn-refresh',
                       color='primary', block=True, size="lg"),
        ],
        width=2,
        style=SIDEBAR_STYLE,
    )

    return dbc.Row(children=[
        sidebar,
        dbc.Col(
            id='display-col',
            children=[
                html.H5('Images:'),
                dbc.Spinner(dbc.Row(
                    id='view2-images-row',
                    form=True,
                    style={'flex-wrap': 'nowrap', 'overflow': 'auto'})),
                html.Br(),
                html.H5('Logs:'),
                dbc.Spinner(dbc.Row(
                    id='view2-logs-row',
                    children=dcc.Textarea(
                            id='view2-logs-text',
                            disabled=True,
                            style={'width': '100%', 'height': '200px'}
                            ),
                )),
            ],
            width=10,
        )
    ])


@app.callback(
    Output(component_id='view1-images-row', component_property='children'),
    Output(component_id='view1-logs-text', component_property='value'),
    Input(component_id='view1-btn-refresh', component_property='n_clicks'),
    State(component_id='view1-human-id', component_property='value'),
    State(component_id='view1-datetime', component_property='start_date'),
    State(component_id='view1-datetime', component_property='end_date'),
    State(component_id='view1-cam-loc-dropdown', component_property='value'),
)
def update_view1_images(btn_clicks, human_id, start_date, end_date, cam_loc):
    #print(human_id, start_date, end_date, cam_loc)
    df_images = dbreid.get_images(human_id=human_id, start_datetime=start_date, end_datetime=end_date)
    #print(df_images.columns)
    images_col = []
    for i in range(len(df_images)):
        encoded_image = base64.b64encode(df_images['img'].iloc[i])
        images_col.append(
            dbc.Col(html.Img(
                src='data:image/png;base64,{}'.format(
                    encoded_image.decode()),
                title=df_images['img_id'].iloc[i],
            )))
    #print(df_images[['img_id', 'human_id', 'inference_datetime']])
    logs_text = df_images[['img_id', 'human_id',
                           'inference_datetime']].to_string(justify='center')
    return images_col, logs_text


@app.callback(
    Output(component_id='view2-images-row', component_property='children'),
    Output(component_id='view2-logs-text', component_property='value'),
    Input(component_id='view2-btn-refresh', component_property='n_clicks'),
    State(component_id='view2-cam-loc-dropdown', component_property='value'),
    State(component_id='view2-datetime', component_property='start_date'),
    State(component_id='view2-datetime', component_property='end_date'),
)
def update_view2_images(btn_clicks, cam_loc, start_date, end_date):
    df_images = dbreid.get_images(
        human_id=None, start_datetime=start_date, end_datetime=end_date)
    df_images = df_images[(df_images['human_id'] != '0000') \
                          & (df_images['human_id'] != '-1')]

    images_col = []
    for i in range(len(df_images)):
        encoded_image = base64.b64encode(df_images['img'].iloc[i])
        images_col.append(
            dbc.Col(html.Img(
                src='data:image/png;base64,{}'.format(
                    encoded_image.decode()),
                title=df_images['img_id'].iloc[i],
            )))

    logs_text = df_images[['img_id', 'human_id',
                           'inference_datetime']].to_string(justify='center')
    return images_col, logs_text
