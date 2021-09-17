import io
from PIL import Image
import logging
import os
import dash
from dash_bootstrap_components._components.Button import Button
from dash_bootstrap_components._components.CardBody import CardBody
from dash_bootstrap_components._components.CardHeader import CardHeader
from dash_bootstrap_components._components.DropdownMenu import DropdownMenu
from dash_bootstrap_components._components.Row import Row
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
#from dash_html_components.H5 import H5
from demo_dash import header_v3 as header
from database import query_database, query_reid
from urllib.parse import parse_qs, urlencode
import base64
import cv2
import mediapipe as mp
import dash_datetimepicker
import sys
import pandas as pd
from datetime import datetime, timedelta
sys.path.append('../')
sys.path.append(f'../reid')
#import inference
from inference import reid_inference
from utils import to_sqlite

_reid_db_path = None
_reid = None

external_stylesheets = [
    dbc.themes.COSMO,
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    "https://use.fontawesome.com/releases/v5.7.2/css/all.css",
]

app = dash.Dash(
    __name__, title="RE-ID Dash",
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
    Input(component_id='url', component_property='search'),
)
def display_page(pathname, search):
    params = extract_results_search_params(pathname, search)
    layout_page = []
    try:
        if params is not None and pathname[1:] == 'results':
            layout_page.append(results_page_content(params))
        else:
            layout_page.append(home_page_content())
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


def home_page_content():
    global _reid_db_path
    _reid_db_path = None
    headerColWidth=2
    content = dbc.Col(
        id='home-page',
        children=[
            # Select Database for use
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.P('Select Database', style={
                            'font-weight': 'bold'}), width=headerColWidth),
                    dbc.Col(dcc.Dropdown(id='database-id',
                            options=get_database_options()), width=True),
                ]),
            ])),
            html.Br(),
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.P('Start Date & Time', style={
                            'font-weight': 'bold'}), width=headerColWidth),
                    dbc.Col([
                            dbc.Row([
                                dbc.Col(html.P('Date:', style={'font-weight': 'bold'}), width='auto', align='center'),
                                dbc.Col(dcc.DatePickerSingle(id='db-date-start-id',display_format='DD-MM-YYYY'), width='auto'),
                                dbc.Col(html.P('Hour (0 ~ 23):', style={'font-weight': 'bold'}), width='auto', align='center'),
                                dbc.Col(dbc.Input(id='db-time-start-hr-id', type='number', min=0, max=23), width=1),
                                dbc.Col(html.P('Minute (0 ~ 59):', style={'font-weight': 'bold'}), width='auto', align='center'),
                                dbc.Col(dbc.Input(id='db-time-start-min-id', type='number', min=0, max=59), width=1),
                            ]),
                            #dash_datetimepicker.DashDatetimepicker(id='datetime-range-id'),
                        ],
                        width=True
                    ),
                ]),
                dbc.Row([
                    dbc.Col(html.P('End Date & Time', style={
                            'font-weight': 'bold'}), width=headerColWidth),
                    dbc.Col([
                            dbc.Row([
                                dbc.Col(html.P('Date:', style={'font-weight': 'bold'}), width='auto', align='center'),
                                dbc.Col(dcc.DatePickerSingle(id='db-date-end-id',display_format='DD-MM-YYYY'), width='auto'),
                                dbc.Col(html.P('Hour (0 ~ 23):', style={'font-weight': 'bold'}), width='auto', align='center'),
                                dbc.Col(dbc.Input(id='db-time-end-hr-id', type='number', min=0, max=23), width=1),
                                dbc.Col(html.P('Minute (0 ~ 59):', style={'font-weight': 'bold'}), width='auto', align='center'),
                                dbc.Col(dbc.Input(id='db-time-end-min-id', type='number', min=0, max=59), width=1),
                            ]),
                        ],
                        width=True
                    ),
                ]),
                html.Br(),
                dbc.Row([
                    dbc.Col(html.P('Select Camera ID', style={
                            'font-weight': 'bold'}), width=headerColWidth),
                    dbc.Col(dcc.Dropdown(id='camera-id'), width=True),
                ]),

                # Line separator
                dbc.Row([
                    dbc.Col(html.Hr(), align='center'),
                    dbc.Col(
                        html.P("or", style={'font-weight': 'bold'}), align='center', width='auto'),
                    dbc.Col(html.Hr(), align='center'),
                ],
                    align='start',
                ),

                dbc.Row([
                    dbc.Col(html.P('Upload an image', style={
                            'font-weight': 'bold'}), width=headerColWidth),
                    dbc.Col(
                        dcc.Upload(
                            id='upload-image',
                            accept='image/*',
                            multiple=False,
                            children=[
                                dbc.Button('Click to upload',
                                           id='upload-image-button',
                                           color='primary',
                                           block=True,
                                           size="lg",
                                           style={'word-wrap': 'normal'})
                            ],
                        ),
                        width='auto',
                    ),
                ]),
                html.P('Picture Requirement:', style={'font-size': 'small'}),
                html.P('• Best with aspect ratio of 1:2 i.e. 128W, 256H',
                       style={'font-size': 'small'}),
                html.P('• Full body image from head to toe',
                       style={'font-size': 'small'}),
            ])),
            html.Br(),
            # Upload image
            dbc.Card(dbc.CardBody([

                dbc.Row([
                    dbc.Col([
                        html.P('Select Human Image', style={'font-weight': 'bold'}),
                        html.P('(Narrow the search by date & time)', style={'font-size': 'small', 'font-style': 'italic'}),
                    ],
                    width=headerColWidth),
                    dbc.Col(
                        id='display-col',
                        children=[
                            dbc.Spinner(dbc.Row(
                                id='view-db-images',
                                form=True,
                                style={
                                    'display': 'flex',
                                    'flex-wrap': 'wrap',
                                    'overflow': 'auto',
                                },
                                #no_gutters=True,
                                #fluid=True,
                            )),
                        ],
                        width=True,
                        align='stretch'
                    ),
                ]),
            ])),
        ],
        width=True,
    )

    return dbc.Row(children=[
        content,
    ])


def results_page_content(params):
    if 'database' in params:
        path_db = params['database']
    else:
        path_db = None
    if 'image_id' in params:
        img_id = params['image_id']
    else:
        img_id = None
    if 'image' in params:
        img = params['image']
    else:
        img = None
    if 'image_filename' in params:
        img_name = params['image_filename']
    else:
        img_name = None

    sidebar_contents = []

    # Show selected image
    if path_db is not None and os.path.exists(path_db):
        dbquery = query_database.DbQuery(path_db)
        minDate, maxDate = dbquery.get_date_range()
        details_row = []
        image=None
        if img_id is not None:
            df = dbquery.get_images(img_id=img_id)
            row = df.iloc[0]
            encoded_image = base64.b64encode(row.img)
            image = 'data:image/png;base64,{}'.format(encoded_image.decode())
            if row.img_id is not None:
                details_row.append(dbc.Row(
                    [
                        html.B('Image ID:', style={'margin-right': '5px'}),
                        html.P(row.img_id),
                    ],
                    #className="card-text",
                ))
            if row.timestamp is not None:
                details_row.append(dbc.Row(
                    [
                        html.B('Date/Time:', style={'margin-right': '5px'}),
                        html.P(row.timestamp),
                    ],
                    #className="card-text",
                ))
            if row.cam_id is not None:
                details_row.append(dbc.Row(
                    [
                        html.B('Camera ID:', style={'margin-right': '5px'}),
                        html.P(row.cam_id),
                    ],
                    #className="card-text",
                ))
            if "loc" in df.columns and row["loc"] is not None:
                details_row.append(dbc.Row(
                    [
                        html.B('Location:', style={'margin-right': '5px'}),
                        html.P(row["loc"]),
                    ],
                    #className="card-text",
                ))
        elif img is not None:
            image = img
            if img_name is not None:
                details_row.append(dbc.Row(
                    [
                        html.B('File Name:', style={'margin-right': '5px'}),
                        html.P(img_name),
                    ],
                ))
        if image is not None:
            sidebar_contents.append(
                dbc.Card(
                    children=[
                        dbc.CardImg(
                            id='results-sidebar-image',
                            src=image,
                            style={
                                'width': '8vw',
                                'object-fit': 'contain',
                            },
                        ),
                        dbc.CardBody(details_row),
                    ],
                    style={
                        'padding': '5%',
                    },
                )
            )

    # filter
    sidebar_contents.append(
        dbc.Card([
            dbc.CardBody([
                html.H6('Search Filter', style={
                    'font-weight': 'bold', 'color': '#007fcf',}),
                #html.Br(),
                dbc.Col([
                        html.P('Select Start Date & Time', style={'font-weight': 'bold'}),
                        dbc.Col([
                            dbc.Row([
                                dbc.Col(html.P('Date:'), width='auto', align='center'),
                                dbc.Col(
                                    dcc.DatePickerSingle(
                                        id='results-filter-date-start-id',
                                        display_format='DD-MM-YYYY',
                                        min_date_allowed=minDate,
                                        max_date_allowed=maxDate),
                                    width=True),
                            ]),
                            dbc.Row([
                                dbc.Col(html.P('Hour (0 ~ 23):'), width='auto', align='center'),
                                dbc.Col(dbc.Input(id='results-filter-time-start-hr-id', type='number', min=0, max=23), width=True),
                            ]),
                            dbc.Row([
                                dbc.Col(html.P('Minute (0 ~ 59):'), width='auto', align='center'),
                                dbc.Col(dbc.Input(id='results-filter-time-start-min-id', type='number', min=0, max=59), width=True),
                            ]),
                        ]),
                        #dash_datetimepicker.DashDatetimepicker(id='results-filter-datetime'),
                        ], style={'padding': '1%'}),
                dbc.Col([
                        html.P('Select End Date & Time', style={'font-weight': 'bold'}),
                        dbc.Col([
                            dbc.Row([
                                dbc.Col(html.P('Date:'), width='auto', align='center'),
                                dbc.Col(
                                    dcc.DatePickerSingle(
                                        id='results-filter-date-end-id',
                                        display_format='DD-MM-YYYY',
                                        min_date_allowed=minDate,
                                        max_date_allowed=maxDate),
                                    width=True),
                            ]),
                            dbc.Row([
                                dbc.Col(html.P('Hour (0 ~ 23):'), width='auto', align='center'),
                                dbc.Col(dbc.Input(id='results-filter-time-end-hr-id', type='number', min=0, max=23), width=True),
                            ]),
                            dbc.Row([
                                dbc.Col(html.P('Minute (0 ~ 59):'), width='auto', align='center'),
                                dbc.Col(dbc.Input(id='results-filter-time-end-min-id', type='number', min=0, max=59), width=True),
                            ]),
                        ]),
                        #dash_datetimepicker.DashDatetimepicker(id='results-filter-datetime'),
                        ], style={'padding': '1%'}),
                dbc.Col([
                        html.P('Camera ID', style={'font-weight': 'bold'}),
                        dcc.Dropdown(id='results-filter-cam-id',
                                   options=dbquery.get_cam_id_options()),
                        ], style={'padding': '1%'}),
                dbc.Col([
                        html.P(children='Threshold (Default is 0.6)',
                            style={'font-weight': 'bold'}),
                        dbc.Input(id='results-filter-threshold',type='number', step=0.1, value=0.6),
                    ],
                    style={'padding': '1%'}),
                html.Br(),
                dbc.Button(children="Filter", id='results-filter-button', color="primary",
                           block=True, size='lg'),
            ]),
        ])
    )

    return dbc.Row(children=[
        dbc.Col(
            id='results-page-sidebar',
            children=sidebar_contents,
            width=3,
            style=SIDEBAR_STYLE,
        ),
        dbc.Col(dbc.Spinner(
            id='display-results-col',
            #width=True,
        ),width=True,),
    ])

@app.callback(
    Output(component_id='camera-id', component_property='options'),
    Input(component_id='database-id', component_property='value'),
)
def update_camera_ids(path_db):
    if path_db is not None:
        dbquery = query_database.DbQuery(path_db)
        return dbquery.get_cam_id_options()
    else:
        return []


@app.callback(
    Output(component_id='upload-image-button', component_property='disabled'),
    Input(component_id='database-id', component_property='value'),
)
def update_camera_ids(path_db):
    return path_db is None

@app.callback(
    Output(component_id='db-date-start-id', component_property='min_date_allowed'),
    Output(component_id='db-date-end-id', component_property='max_date_allowed'),
    Input(component_id='database-id', component_property='value'),
)
def update_db_start_date_min_end_date_max(path_db):
    if path_db is not None:
        dbquery = query_database.DbQuery(path_db)
        minDate, maxDate = dbquery.get_date_range()
        return minDate, maxDate
    else:
        return None, None

@app.callback(
    Output(component_id='db-date-start-id', component_property='max_date_allowed'),
    Input(component_id='db-date-end-id', component_property='date'),
    Input(component_id='db-date-end-id', component_property='max_date_allowed'),
)
def update_db_start_date_max(end_date, end_max_date):
    if end_date is not None:
        return end_date
    else:
        return end_max_date


@app.callback(
    Output(component_id='db-date-end-id', component_property='min_date_allowed'),
    Input(component_id='db-date-start-id', component_property='date'),
    Input(component_id='db-date-start-id', component_property='min_date_allowed'),
)
def update_db_end_date_min(start_date, start_min_date):
    if start_date is not None:
        return start_date
    else:
        return start_min_date


@app.callback(
    Output(component_id='results-filter-date-start-id',component_property='max_date_allowed'),
    Input(component_id='results-filter-date-end-id', component_property='date'),
    Input(component_id='results-filter-date-end-id', component_property='max_date_allowed'),
)
def update_results_start_date_max(end_date, end_max_date):
    if end_date is not None:
        return end_date
    else:
        return end_max_date


@app.callback(
    Output(component_id='results-filter-date-end-id',component_property='min_date_allowed'),
    Input(component_id='results-filter-date-start-id', component_property='date'),
    Input(component_id='results-filter-date-start-id',component_property='min_date_allowed'),
)
def update_results_end_date_min(start_date, start_min_date):
    if start_date is not None:
        return start_date
    else:
        return start_min_date

@app.callback(
    Output(component_id='view-db-images', component_property='children'),
    Input(component_id='database-id', component_property='value'),
    Input(component_id='db-date-start-id', component_property='date'),
    Input(component_id='db-time-start-hr-id', component_property='value'),
    Input(component_id='db-time-start-min-id', component_property='value'),
    Input(component_id='db-date-end-id', component_property='date'),
    Input(component_id='db-time-end-hr-id', component_property='value'),
    Input(component_id='db-time-end-min-id', component_property='value'),
    Input(component_id='camera-id', component_property='value'),
    Input(component_id='upload-image', component_property='contents'),
    State(component_id='upload-image', component_property='filename'),
)
def show_database_images(path_db, start_date, start_hour, start_minute, end_date, end_hour, end_minute, cam_id, upload_img, upload_filename):
    dict_trig = get_callback_trigger()
    if 'upload-image' in dict_trig:
        tooltip_msg = f"File name: {upload_filename}"
        return [
            dbc.Card([
                dbc.CardLink(
                    dbc.CardImg(
                        src=upload_img,
                        title=tooltip_msg.strip(),
                        style={
                            'width': '8vw',
                            'object-fit': 'contain'
                        },
                    ),
                    key=upload_filename,
                    # f'{urlResults}?{urlencode(url_dict)}'
                    href=get_results_href(
                        path_db, img=upload_img, img_filename=upload_filename)
                ),
            ])
        ]
    elif path_db is not None and cam_id is not None:
        dbimage = query_database.DbQuery(path_db)
        start_datetime = compile_start_datetime(start_date, start_hour, start_minute)
        end_datetime = compile_end_datetime(end_date, end_hour, end_minute)
        df_images = dbimage.get_images(
            cam_id=cam_id, start_datetime=start_datetime, end_datetime=end_datetime)
        images_col = []
        for _, row in df_images.iterrows():
            encoded_image = base64.b64encode(row.img)
            components = [
                #html.P(f'Camera {row.cam_id}', style={'text-overflow': 'ellipsis', 'width': '8vw', 'margin': '0'})
            ]

            timestamp = row.timestamp
            if timestamp is not None:
                if type(timestamp) == str:
                    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                components.extend([
                    html.P(timestamp.date(),
                                         style={'text-overflow': 'ellipsis', 'width': '8vw', 'margin': '0'}),
                    html.P(timestamp.strftime("%X"),
                           style={'text-overflow': 'ellipsis', 'width': '8vw', 'margin': '0'}),
                    ])

            components.append(
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_image.decode()),
                    title=row.img_id,
                    style={
                        'width': '8vw',
                        'object-fit': 'contain'
                    }
                ))

            tooltip_msg = ""
            if (row.img_id is not None):
                tooltip_msg += f"Image ID: {row.img_id}\r\n"
            if (timestamp is not None):
                tooltip_msg += f"Datetime: {timestamp}\r\n"
            if (row.cam_id is not None):
                tooltip_msg += f"Camera ID: {row.cam_id}\r\n"
            if "loc" in df_images.columns:
                tooltip_msg += f'Location: {row["loc"]}\r\n'
            images_col.append(
                dbc.Card([
                    dbc.CardLink(
                        dbc.CardImg(
                            src='data:image/png;base64,{}'.format(encoded_image.decode()),
                            title=tooltip_msg.strip(),
                            style={
                                'width': '8vw',
                                'object-fit': 'contain'
                            },
                        ),
                        key=row.img_id,
                        href=get_results_href(path_db, img_id=row.img_id)#f'{urlResults}?{urlencode(url_dict)}'
                    ),
                ])
            )
        return images_col
    else:
        return None


@app.callback(
    Output(component_id='display-results-col', component_property='children'),
    Input(component_id='url', component_property='pathname'),
    Input(component_id='url', component_property='search'),
    Input(component_id='results-filter-button', component_property='n_clicks'),
    State(component_id='results-filter-date-start-id', component_property='date'),
    State(component_id='results-filter-time-start-hr-id', component_property='value'),
    State(component_id='results-filter-time-start-min-id', component_property='value'),
    State(component_id='results-filter-date-end-id', component_property='date'),
    State(component_id='results-filter-time-end-hr-id', component_property='value'),
    State(component_id='results-filter-time-end-min-id', component_property='value'),
    State(component_id='results-filter-cam-id', component_property='value'),
    State(component_id='results-filter-threshold', component_property='value'),
)
def show_results_images(pathname, search, n_clicks, start_date, start_hour, start_minute, end_date, end_hour, end_minute, cam_id, threshold):
    dict_trig = get_callback_trigger()

    if 'url' in dict_trig or 'results-filter-button' not in dict_trig:
        start_date = None
        end_date = None
        cam_id = None
        threshold = None

    params = extract_results_search_params(pathname, search)
    if params is not None:
        if 'database' in params:
            path_db = params['database']
        else:
            path_db = None
        if 'image_id' in params:
            img_id = params['image_id']
        else:
            img_id = None
        if 'image' in params:
            img = params['image']
        else:
            img = None
        if 'image_filename' in params:
            img_name = params['image_filename']
        else:
            img_name = None
    else:
        return

    if threshold is None:
        threshold = 0.6

    row_images = []
    if path_db is not None and (img_id is not None or img is not None) and os.path.exists(path_db):
        dbquery = query_database.DbQuery(path_db)
        if img_id is not None:
            df = dbquery.get_images(img_id=img_id)
            row = df.iloc[0]
            image = row.img
        else:
            image = img

        result = run_reid(image, path_db, threshold)

        dict_result = {}
        for item in result:
            for key in item:
                if key in dict_result:
                    dict_result[key].append(item[key])
                else:
                    dict_result[key] = [item[key]]

        df = pd.DataFrame.from_dict(dict_result)
        list_cams = sorted(df.cam_id.unique().tolist())
        if cam_id is not None:
            if cam_id in list_cams:
                list_cams = [cam_id]
            else:
                list_cams = None

        start_datetime = compile_start_datetime(start_date, start_hour, start_minute)
        end_datetime = compile_end_datetime(end_date, end_hour, end_minute)
        if list_cams is not None and len(list_cams) > 0:
            for cam_id in list_cams:
                cam_images=[]
                header = f'Camera {cam_id}'
                for idx_cam, row_cam in df[df.cam_id == cam_id].iterrows():
                    db_reid = query_reid.DbQuery(path_db)
                    df_query = db_reid.get_images(row_cam.img_id)
                    df_query.timestamp = pd.to_datetime(df_query.timestamp)
                    if start_datetime is not None:
                        df_query = df_query[df_query.timestamp >= start_datetime]
                    if end_datetime is not None:
                        df_query = df_query[df_query.timestamp < end_datetime]
                    if len(df_query) > 0:
                        if "loc" in df_query.columns:
                            header = f'Camera {cam_id} {df_query[df_query.cam_id == cam_id]["loc"].iloc[0]}'
                        for idx_query, row_query in df_query.iterrows():
                            encoded_image = base64.b64encode(row_query.img)
                            id_tag = f'result-img-id-{row_query.img_id}'
                            tooltip = []
                            if row_query.img_id is not None:
                                tooltip.extend([
                                    html.B('Image ID:'),
                                    html.Br(),
                                    html.Span(row_query.img_id),
                                    html.Br(),
                                ])
                            if row_query.timestamp is not None:
                                tooltip.extend([
                                    html.B('Date time detected:'),
                                    html.Br(),
                                    html.Span(row_query.timestamp),
                                    html.Br(),
                                ])
                            if row_cam.dist is not None:
                                tooltip.extend([
                                    html.B('Similarity: '),
                                    #html.Br(),
                                    html.Span(round(row_cam.dist,4)),
                                ])
                            cam_images.append(
                                dbc.Card(
                                    children=[
                                        dbc.CardImg(
                                            src='data:image/png;base64,{}'.format(encoded_image.decode()),
                                            id=id_tag,
                                            #title=tooltip_msg.strip(),
                                            style={
                                                'width': '8vw',
                                                'object-fit': 'contain',
                                                #'margin':'5%',
                                            },
                                        ),
                                        dbc.Tooltip([
                                            html.P(tooltip, style={'text-align': 'left'}),
                                            dbc.Button(
                                                html.B('Query this'),
                                                id=f'query-img-id-{row_query.img_id}',
                                                size="md",
                                                href=get_results_href(path_db, img_id=row_query.img_id),
                                                ),
                                            ],
                                            target=id_tag,
                                            autohide=False,
                                            style={'font-size': 'small'},
                                        )
                                    ]
                                ))


                if len(cam_images) > 0:
                    row_images.append(
                        dbc.Card([
                            dbc.CardHeader(header, style={'font-weight': 'bold'}),
                            dbc.CardBody(dbc.Row(cam_images),style={'margin': '1%'},),
                        ]))
            if len(row_images) <= 0:
                row_images.append(html.P('No results found!'))
        else:
            row_images.append(html.P('No results found!'))
    return row_images


@app.callback(
    Output(component_id='results-filter-datetime', component_property='startDate'),
    Input(component_id='url', component_property='pathname'),
    Input(component_id='url', component_property='search')
)
def UpdateResultsFilter(pathname, search):
    params = extract_results_search_params(pathname, search)
    layout_page = []
    try:
        if params is not None and pathname[1:] == 'results':
            layout_page.append(results_page_content(params))
        else:
            layout_page.append(home_page_content())
    except Exception as ex:
        logging.error(ex)
    return None  # , title


def get_database_options():
    path_folder = f'../reid/archive'
    options = []
    for file in sorted(os.listdir(path_folder)):
        filePath = os.path.join(path_folder, file)
        if is_database_valid(filePath):
            fileName, fileExt = os.path.splitext(file)
            if os.path.isfile(filePath) and fileExt.lower() == '.db':
                options.append({'label': fileName, 'value': filePath})

    return options


def is_database_valid(path_db):
    try:
        dbTemp = query_database.DbQuery(path_db)
        tableName = 'vectorkb_table'
        if tableName in dbTemp.get_table_list():
            return True#'cam_id' in dbTemp.get_columns_list(tableName)
        else:
            return False
    except:
        return False


def extract_results_search_params(pathname, search):
    if search is not None:
        queries = parse_qs(search[1:])
        logging.info(queries)
    else:
        queries = None

    try:
        if pathname is not None:
            if queries is not None and pathname[1:] == 'results':
                params={}
                for name in queries:
                    params[name] = queries[name][0]
                return params
    except Exception as ex:
        logging.error(ex)
    return None


def get_callback_trigger():
    ctx = dash.callback_context
    dictTrigger={}
    for trig in ctx.triggered:
        splitTxt = trig['prop_id'].split('.')
        if len(splitTxt) == 2 and len(splitTxt[0]) > 0:
            if splitTxt[0] in dictTrigger:
                dictTrigger[splitTxt[0]].append(splitTxt[1])
            else:
                dictTrigger[splitTxt[0]]=[splitTxt[1]]

    return dictTrigger


# reference /human_tracker/reid_inference.py
def run_reid(img, db_path, threshold=0.6):
    init_reid(db_path)
    global _reid
    #to_sqlite.db_path = db_path
    if isinstance(img, bytes):
        pil_img = to_sqlite.convertBlobtoIMG(img)
    elif isinstance(img,str):
        encoded_image = img.split(",")[1]
        decoded_image = base64.b64decode(encoded_image)
        bytes_image = io.BytesIO(decoded_image)
        pil_img = Image.open(bytes_image).convert('RGB')

    query_feat = _reid.to_query_feat(pil_img)
    return _reid.infer(query_feat, thres=threshold)


def init_reid(db_path):
    global _reid, _reid_db_path
    if _reid is None or _reid_db_path is None or _reid_db_path != db_path:
        _reid = reid_inference(db_path)
        _reid_db_path = db_path



def get_results_href(path_db, img_id=None, img=None, img_filename=None, start_date=None, end_date=None, cam_id=None, threshold=None):
    urlResults = '/results'
    url_dict = {'database': path_db}
    if img_id is not None:
        url_dict['image_id'] = img_id
    if img is not None:
        url_dict['image'] = img
    if img_filename is not None:
        url_dict['image_filename'] = img_filename
    if start_date is not None:
        url_dict['start'] = start_date
    if end_date is not None:
        url_dict['end'] = end_date
    if cam_id is not None:
        url_dict['camera'] = cam_id
    if threshold is not None:
        url_dict['threshold'] = threshold

    return f'{urlResults}?{urlencode(url_dict)}'

def compile_start_datetime(start_date, start_hour, start_minute):
    if start_date is None:
        return None
    if start_hour is None:
        start_hour = 0
    if start_minute is None:
        start_minute = 0
    return compile_datetime(start_date, start_hour, start_minute)

def compile_end_datetime(end_date, end_hour, end_minute):
    if end_date is None:
        return None
    if end_hour is None or end_hour > 23:
        return compile_datetime(end_date, 0, 0) + timedelta(days=1)
    elif end_minute is None or end_minute > 59:
        return compile_datetime(end_date, end_hour, 0) + timedelta(hours=1)
    else:
        return compile_datetime(end_date, end_hour, end_minute) + timedelta(minutes=1)

def compile_datetime(date, hour, minute):
    date_string = f"{date} {hour}:{minute}"
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M")
