import logging
import dash
from dash_bootstrap_components._components.Spinner import Spinner
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL, ALLSMALLER
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

def get_human_id_options(df):
    human_ids = sorted(df.human_id.unique())
    return [{'label': id, 'value': id} for id in human_ids]


def build_image_row(df, human_id):
    dfTemp = df[df.human_id == human_id]
    listImages = []
    n_images = 6
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
    return listImages


def compare_images(df):
    dictRows = {}
    human_ids = sorted(df.human_id.unique())
    for human_id in human_ids:
        row = dbc.Card(
            id=f'row-human_id-{human_id}',
            children=[
                html.P(f'Human ID: {human_id}'),
                html.A(
                    children=build_image_row(df, human_id),
                    id={'index': f'compare-human_id-{human_id}',
                        'type': f'compare'},
                    key=human_id,
                    style={
                        'display': 'flex',
                        'overflow': 'auto',
                    },
                )
            ])
        dictRows[human_id] = row
    return dictRows


def reference_images(df):
    dictRows = {}
    human_ids = sorted(df.human_id.unique())
    for human_id in human_ids:
        row = dbc.Card(
            id=f'row-human_id-{human_id}',
            children=[
                html.P(f'Human ID: {human_id}'),
                dbc.Row(
                    children=build_image_row(df, human_id),
                    id={'index': f'reference-human_id-{human_id}',
                        'type': f'reference'},
                    key=human_id,
                    style={
                        'display': 'flex',
                        'overflow': 'auto',
                    },
                )
            ])
        dictRows[human_id] = row
    return dictRows

dbMerge = query_merge.MergeDbQuery()
dfMerge = process_images(dbMerge.get_all_correctlabel())
humanIdOptions = get_human_id_options(dfMerge)
dictCompareRows = compare_images(dfMerge)
dictReferenceRows = reference_images(dfMerge)
listMerge = dbMerge.get_merge()
table_header = [html.Thead(html.Tr([html.Th("Merge Human IDs")]))]

external_stylesheets = [
    dbc.themes.COSMO,
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    "https://use.fontawesome.com/releases/v5.7.2/css/all.css",
]

app = dash.Dash(
    __name__, title="Merge Re-ID 2",
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
    global humanIdOptions
    left_sidebar = dbc.Col(
        id='view-page-sidebar',
        children=[
            html.P('Human ID:'),
            dbc.Spinner(dcc.Dropdown(
                id='view-human-id',
                options=humanIdOptions,
            )),
        ],
        width='auto',
        style=LEFT_SIDEBAR_STYLE,
    )

    center_content = dbc.Col(
        id='display-col',
        children=[
            dbc.Spinner(
                children=[
                    dbc.Row(
                        id='reference-images-row',
                        form=True,
                        style={'display': 'flex',
                            'flex-wrap': 'wrap', 'overflow': 'auto'},
                        no_gutters=True,
                    )]
            ),
            html.Hr(),
            dbc.Spinner(dbc.Row(
                id='compare-images-row',
                form=True,
                style={'display': 'flex',
                       'flex-wrap': 'wrap', 'overflow': 'auto'},
                no_gutters=True,
            )),
        ],
        width=True,
        align='stretch'
    )

    save_row = dbc.Row(
        children=[
            dbc.Button(children='Save Merges', id='btn-save',
                       disabled=True, color='primary', size="lg"),
            html.Br(),
        ],
        no_gutters=True,
        justify='end'
    )
    global table_header
    table = dbc.Row(
        id='table-merge-row',
        children=dbc.Table(id='table-merge', striped=True,
                           bordered=True, hover=True),
        form=True, no_gutters=True, justify='stretch')
    reset_row = dbc.Row(
        children=[
            html.Br(),
            dbc.Button(children='Reset Merges', id='btn-reset-db',
                            disabled=False, color='secondary', size="lg")
        ],
        no_gutters=True,
        justify='end'
    )
    right_sidebar = dbc.Col(
        id='right-sidebar',
        children=[save_row, table, reset_row],
        width=2,
        style=RIGHT_SIDEBAR_STYLE,
    )
    return dbc.Row(children=[
        left_sidebar,
        center_content,
        right_sidebar
    ])

@app.callback(
    Output(component_id='reference-images-row', component_property='children'),
    Input(component_id='view-human-id', component_property='value'),
)
def update_reference_images(human_id):
    if human_id is not None:
        global dictReferenceRows
        return dictReferenceRows[human_id]
    else:
        return None


@app.callback(
    Output(component_id='compare-images-row', component_property='children'),
    Input(component_id='view-human-id', component_property='value'),
)
def update_compare_images(human_id):
    if human_id is not None:
        global dictCompareRows
        return dbc.Col(
            id='display-col',
            children=[dictCompareRows[id]
                      for id in dictCompareRows if id != human_id],
            width=True,
            #align='stretch',
            style={'overflowY': 'scroll', 'height': '60vh'}
        )
    else:
        return None


@app.callback(
    Output({'type': 'compare', 'index': MATCH}, 'style'),
    Output({'type': 'compare', 'index': MATCH}, 'n_clicks'),
    Input({'type': 'compare', 'index': MATCH}, 'n_clicks'),
    State({'type': 'compare', 'index': MATCH}, 'style'),
    State({'type': 'compare', 'index': MATCH}, 'key'),
    State('view-human-id', 'value'),
)
def display_compare_output(n_clicks, style, key, ref_id):
    #print(f'display_compare_output {ref_id} {key}')
    merge = get_selected_ids(ref_id)
    #cmp_id = id['index'].replace('compare-human_id-', '')
    if merge is not None and key in merge:
        style['background-color'] = 'yellow'
    if n_clicks is not None:
        if 'background-color' in style:
            del style['background-color']
            del_merge_list(ref_id, key)
        else:
            style['background-color'] = 'yellow'
            add_merge_list(ref_id, key)
    return style, n_clicks


@app.callback(
    Output('table-merge', 'children'),
    Output('btn-reset-db', 'n_clicks'),
    Input({'type': 'compare', 'index': ALL}, 'style'),
    Input('btn-reset-db', 'n_clicks'),
)
def update_merge_table(styles, n_clicks):
    global listMerge
    #print(listMerge)
    if n_clicks is not None:
        dbMerge.reset_merge()
        listMerge = dbMerge.get_merge()

    table_rows = [html.Tr(str(merge)) for merge in listMerge]
    return table_header + [html.Tbody(table_rows)], None


@app.callback(
    Output('btn-save', 'disabled'),
    Input('table-merge', 'children'),
)
def enable_save(children):
    return len(listMerge) <= 0


@app.callback(
    Output('btn-save', 'n_clicks'),
    Input('btn-save', 'n_clicks'),
)
def save_table(n_clicks):
    if n_clicks is not None:
        dbMerge.save_merge(listMerge)
    return None

def get_selected_ids(ref_id):
    global listMerge
    for merge in listMerge:
        if ref_id in merge:
            return merge
    return None

def update_merge_list(list_ids):
    if list_ids is not None:
        global listMerge
        for idx, merge in enumerate(listMerge):
            for id in list_ids:
                if id in merge:
                    listMerge[idx] = list(set(merge + list_ids))
                    return
        listMerge.append(list_ids)

def add_merge_list(ref_id, new_id):
    #print(f'add_merge_list({ref_id}, {new_id})')
    new_list = [ref_id, new_id]
    if ref_id is not None and new_id is not None:
        global listMerge
        for idx in range(len(listMerge)):
            for id in new_list:
                if id in listMerge[idx]:
                    #print(listMerge[idx])
                    listMerge[idx] = sorted(
                        list(set(listMerge[idx] + new_list)))
                    #print(listMerge[idx])
                    #print(listMerge)
                    return
        listMerge.append([ref_id, new_id])
        #print(listMerge)


def del_merge_list(ref_id, del_id):
    #print(f'del_merge_list({ref_id}, {del_id})')
    if ref_id is not None and del_id is not None:
        global listMerge
        for idx in range(len(listMerge)):
            if ref_id in listMerge[idx]:
                listMerge[idx].remove(del_id)
                if len(listMerge[idx]) <= 1:
                    del listMerge[idx]
                #print(listMerge)
                return
        #print(listMerge)
