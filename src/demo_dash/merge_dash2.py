import logging
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from demo_dash import header_merge2 as header
from database import query_merge
from urllib.parse import parse_qs
import base64
import cv2
import numpy as np
import pandas as pd


def process_images(df):
    decoded_column = []
    for _, row in df.iterrows():
        decoded_column.append(
            f'data:image/png;base64,{base64.b64encode(row.img).decode()}')
    df['img_decoded'] = decoded_column
    return df


def prepare_images(df, colNum):
    dictRows = {}
    human_ids = df.human_id.unique()
    n_images = 4
    for human_id in human_ids:
        dfTemp = df[df.human_id == human_id]
        listImages = []
        for i in range(n_images):
            if i < len(dfTemp.index):
                listImages.append(
                    dbc.Col(
                        children=[
                            html.Img(
                                src=dfTemp.iloc[i].img_decoded,
                                title=dfTemp.iloc[i].img_id,
                                style={'width': '5vw', 'object-fit': 'contain'}
                            )
                        ],
                    )
                )
            else:
                break

        row = dbc.Card(
            id=f'col-{colNum}-human_id-{human_id}',
            children=[
                html.A(
                    children=listImages,
                    id={'index': f'col-{colNum}-human_id-{human_id}',
                        'type': f'col-{colNum}'},
                    style={
                        'display': 'flex',
                        'overflow': 'auto',
                    },
                )
            ])
        dictRows[human_id] = row
    return dictRows


db_merge = query_merge.MergeDbQuery()
df_merge = process_images(db_merge.get_all_correctlabel())
listCols = []
for i in range(2):
    listCols.append(prepare_images(df_merge, i))

external_stylesheets = [
    dbc.themes.COSMO,
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    "https://use.fontawesome.com/releases/v5.7.2/css/all.css",
]

app = dash.Dash(
    __name__, title="Label Re-ID 1",
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

    layout_page = []
    try:
        layout_page.append(view_page_content())
    except Exception as ex:
        logging.error(ex)
    return layout_page


LEFT_SIDEBAR_STYLE = {
    "position": "sticky",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    'height': '100%',
}

RIGHT_SIDEBAR_STYLE = {
    "position": "sticky",
    "top": 0,
    "right": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "1rem 2rem",
    "background-color": "#faf9f8",
    'height': '100%',
}


def view_page_content():
    compare_col1 = dbc.Col(
        id='compare-col1',
        children=generate_images_col(0),
        width=5,
        align='stretch',
        style={'overflowY': 'scroll', 'height': '80vh'}
    )

    compare_col2 = dbc.Col(
        id='display-col',
        children=generate_images_col(1),
        width=5,
        align='stretch',
        style={'overflowY': 'scroll', 'height': '80vh'}
    )

    right_sidebar = dbc.Col(
        id='right-sidebar',
        children=[],
        width=2,
        style=RIGHT_SIDEBAR_STYLE,
    )
    return dbc.Row(children=[
        compare_col1,
        compare_col2,
        right_sidebar
        ],
        )

def generate_images_col(col_num):
    rows = [html.H4(f'Column {col_num + 1}')]
    for human_id in listCols[col_num]:
        rows.append(listCols[col_num][human_id])
    return rows
    human_ids = df_merge.human_id.unique()
    rows=[html.H4(f'Column {col_num + 1}')]
    for human_id in human_ids:
        rows.append(generate_images_row(human_id, col_num))
    return rows


def generate_images_row(human_id, col_num):
    df = df_merge[df_merge.human_id == human_id]
    n_images = 4
    listImages = []
    #print(human_id, df.iloc[0].human_id)
    for i in range(n_images):
        if i < len(df.index):
            listImages.append(
                dbc.Col(
                    children=[
                        #html.P(df.iloc[i].img_id,
                        #       style={'white-space': 'nowrap', 'overflow': 'hidden',
                        #              'text-overflow': 'ellipsis', 'width': '4vw', 'margin': '0'}),
                        html.Img(
                            src=df.iloc[i].img_decoded,
                            title=df.iloc[i].img_id,
                            #id=img_id,
                            style={'width': '5vw', 'object-fit': 'contain'}
                        )
                    ],
                )
            )
        else:
            break

    return dbc.Card(
        id=f'col-{col_num}-human_id-{human_id}',
        children=[
            html.A(
                children=listImages,
                id={'index': f'col-{col_num}-human_id-{human_id}', 'type': f'col-{col_num}'},
                #form=True,
                style={
                    'display': 'flex',
                    #'flex-wrap': 'wrap',
                    'overflow': 'auto',
                    #'align-content': 'flex-start'
                },
                #no_gutters=True,
                )
            ])


@app.callback(
    Output({'type': 'col-0', 'index': ALL}, 'style'),
    Output({'type': 'col-0', 'index': ALL}, 'n_clicks'),
    Input({'type': 'col-0', 'index': ALL}, 'n_clicks'),
    State({'type': 'col-0', 'index': ALL}, 'style'),
)
def display_col0_output(n_clicks, style):
    #print(n_clicks)
    for idx, n_click in enumerate(n_clicks):
        #print(idx, n_click)
        if n_click is not None:
            n_clicks[idx]=None
            style[idx]['background-color'] = 'yellow'
        elif 'background-color' in style[idx]:
            del style[idx]['background-color']
    #print(n_clicks)
    return style, n_clicks


@app.callback(
    Output({'type': 'col-1', 'index': MATCH}, 'style'),
    Output({'type': 'col-1', 'index': MATCH}, 'n_clicks'),
    Input({'type': 'col-1', 'index': MATCH}, 'n_clicks'),
    State({'type': 'col-1', 'index': MATCH}, 'style'),
)
def display_col1_output(n_clicks, style):
    if n_clicks is not None:
        if 'background-color' in style:
            del style['background-color']
        else:
            style['background-color'] = 'yellow'

    return style, None
