import logging
import os
import dash
from dash_bootstrap_components._components.CardHeader import CardHeader
from dash_bootstrap_components._components.DropdownMenu import DropdownMenu
from dash_bootstrap_components._components.Row import Row
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash_html_components.H5 import H5
from demo_dash import header_v2 as header
from database import query_database
from urllib.parse import parse_qs, urlencode
import base64
import cv2
import mediapipe as mp
import dash_datetimepicker

_dbquery = None


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
    params = extract_search_params(pathname, search)
    #print(params)
    layout_page = []
    try:
        if params is not None and len(params) ==2 and pathname[1:] == 'results':
            path_db, img_id =params
            layout_page.append(results_page_content(path_db, img_id))
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

CAMERA_LOCATION_OPTIONS = [
    {'label': 'Camera 1', 'value': 1},
    {'label': 'Camera 2', 'value': 2},
    {'label': 'Camera 3', 'value': 3},
    {'label': 'Camera 4', 'value': 4},
    {'label': 'Camera 5', 'value': 5},
]

def home_page_content():
    headerColWidth=2
    content = dbc.Col(
        id='home-page',
        children=[
            # Upload image
            dbc.Row([
                dbc.Col(html.P('Upload an image', style={'font-weight': 'bold'}), width=headerColWidth),
                dbc.Col(
                    dcc.Upload(
                        id='upload-image',
                        accept='image/*',
                        multiple=False,
                        children=[
                            dbc.Button('Click to upload',
                                       color='primary',
                                       block=True,
                                       size="lg",
                                       style={'word-wrap':'normal'})
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

            # Line separator
            dbc.Row([
                dbc.Col(html.Hr(), align='center'),
                dbc.Col(html.P("or", style={'font-weight': 'bold'}), align='center',width='auto'),
                dbc.Col(html.Hr(), align='center'),
                ],
                align='start',
            ),

            # Extract from Database
            dbc.Row([
                dbc.Col(html.P('Select Database', style={'font-weight': 'bold'}), width=headerColWidth),
                dbc.Col(dcc.Dropdown(id='database-id', options=get_database_options()),width=True),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.P('Select Date & Time', style={'font-weight': 'bold'}), width=headerColWidth),
                dbc.Col(dash_datetimepicker.DashDatetimepicker(id='datetime-range-id'),width=True),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(html.P('Select Camera ID', style={'font-weight': 'bold'}), width=headerColWidth),
                dbc.Col(dcc.Dropdown(id='camera-id'), width=True),
            ]),
            html.Br(),
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
            ])
        ],
        width=True,
    )

    return dbc.Row(children=[
        content,
    ])


def results_page_content(path_db=None, img_id=None):
    sidebar_contents = []

    # Show selected image
    if path_db is not None and img_id is not None and os.path.exists(path_db):
        dbquery = query_database.DbQuery(path_db)
        df = dbquery.get_images(img_id=img_id)
        row = df.iloc[0]
        print(row)
        encoded_image = base64.b64encode(row.img)
        details_row = []
        if row.img_id is not None:
            details_row.append(dbc.Row(
                [
                    html.B('Image ID:', style={'margin-right':'5px'}),
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
                    html.B('Camera ID:', style={'margin-right':'5px'}),
                    html.P(row.cam_id),
                ],
                #className="card-text",
            ))
        sidebar_contents.append(
            dbc.Card(
                children=[
                    dbc.CardImg(
                        id='results-sidebar-image',
                        src='data:image/png;base64,{}'.format(
                            encoded_image.decode()),
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
                html.Br(),
                dbc.Col([
                        html.P('Select Date & Time', style={
                               'font-weight': 'bold'}),
                        dash_datetimepicker.DashDatetimepicker(
                            id='results-filter-datetime'),
                        ], style={'padding': '1%'}),
                dbc.Col([
                        html.P('Camera ID', style={'font-weight': 'bold'}),
                        dbc.Input(id='results-filter-cam-id',type='number', step=1),
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
        dbc.Col(
            id='display-results-col',
            width=True,
        ),
    ])


@app.callback(
    Output(component_id='camera-id', component_property='options'),
    Input(component_id='database-id', component_property='value'),
)
def update_camera_ids(path_db):
    if path_db is not None:
        #print(path_db)
        dbquery = query_database.DbQuery(path_db)
        return dbquery.get_cam_id_options()
    else:
        return []


@app.callback(
    Output(component_id='view-db-images', component_property='children'),
    Input(component_id='database-id', component_property='value'),
    Input(component_id='datetime-range-id', component_property='startDate'),
    Input(component_id='datetime-range-id', component_property='endDate'),
    Input(component_id='camera-id', component_property='value'),
)
def show_database_images(path_db, start_date, end_date, cam_id):
    if path_db is not None and \
        start_date is not None and \
            end_date is not None and \
                cam_id is not None:
        #print(path_db)
        #print(start_date, end_date)
        #print(cam_id)
        dbimage = query_database.DbQuery(path_db)
        df_images = dbimage.get_images(
            cam_id=cam_id, start_datetime=None, end_datetime=None)
        #print(df_images.info())
        images_col = []
        images_col1 = []
        urlResults = '/results'
        for _, row in df_images.iterrows():
            encoded_image = base64.b64encode(row.img)
            components = [
                #html.P(f'Camera {row.cam_id}', style={'text-overflow': 'ellipsis', 'width': '8vw', 'margin': '0'})
            ]
            #url_dict = {'database': os.path.basename(path_db), 'image_id': row.img_id}
            url_dict = {'database': path_db, 'image_id': row.img_id}

            if row.timestamp is not None:
                components.extend([
                    html.P(row.timestamp.date(),
                                         style={'text-overflow': 'ellipsis', 'width': '8vw', 'margin': '0'}),
                    html.P(row.timestamp.strftime("%X"),
                           style={'text-overflow': 'ellipsis', 'width': '8vw', 'margin': '0'}),
                    ])

            components.append(
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_image.decode()),
                    title=row.img_id,
                    #id=img_id,
                    style={
                        'width': '8vw',
                        'object-fit': 'contain'
                    }
                ))

            images_col1.append(
                dbc.Col(
                    children=[
                        dbc.Spinner(html.A(
                            id={'index': row.img_id, 'type': 'image'},
                            key=row.img_id,
                            children=components,
                            href=f'{urlResults}?{urlencode(url_dict)}'
                        ))
                    ],
                    width='auto',
                    align='start',
                ))
            tooltip_msg = ""
            if (row.img_id is not None):
                tooltip_msg += f"Image ID: {row.img_id}\r\n"
            if (row.timestamp is not None):
                tooltip_msg = f"Datetime: {row.timestamp}\r\n"
            if (row.cam_id is not None):
                tooltip_msg += f"Camera ID: {row.cam_id}\r\n"
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
                        href=f'{urlResults}?{urlencode(url_dict)}'
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
    State(component_id='results-filter-datetime', component_property='startDate'),
    State(component_id='results-filter-datetime', component_property='endDate'),
    State(component_id='results-filter-cam-id', component_property='value'),
    State(component_id='results-filter-threshold', component_property='value'),
)
def show_results_images(pathname, search, n_clicks, startDate, endDate, cam_id, threshold):
    params = extract_search_params(pathname, search)
    if params is not None and len(params) == 2:
        path_db, img_id = params
    else:
        return

    dictTrig = get_callback_trigger()
    print(path_db, img_id, n_clicks, startDate, endDate, cam_id, threshold)
    return None


def get_database_options():
    path_folder = f'{os.getcwd()}\\reid'
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
        return 'vectorkb_table' in dbTemp.get_table_list()
    except:
        return False


def extract_search_params(pathname, search):
    if search is not None:
        queries = parse_qs(search[1:])
        logging.info(queries)
    else:
        queries = None
    #print(pathname, queries)

    try:
        if pathname is not None:
            if queries is not None and pathname[1:] == 'results':
                return queries['database'][0], queries['image_id'][0]
    except Exception as ex:
        logging.error(ex)
    return None


def get_callback_trigger():
    ctx = dash.callback_context
    dictTrigger={}
    for trig in ctx.triggered:
        splitTxt = trig['prop_id'].split('.')
        if len(splitTxt) == 2 and len(splitTxt[0]) > 0:
            dictTrigger[splitTxt[0]]=splitTxt[1]

    return dictTrigger
