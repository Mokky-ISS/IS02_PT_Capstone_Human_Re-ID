import logging
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from demo_dash import header_label1 as header
from database import query_re_id, query_mislabelled
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

db_reid = query_re_id.DbQuery()
df_reid = process_images(db_reid.get_images())
db_mislabelled = query_mislabelled.DbQuery()
df_mislabelled = db_mislabelled.get_mislabelled()
df_mislabelled_orig = df_mislabelled.copy()

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
    left_sidebar = dbc.Col(
        id='view-page-sidebar',
        children=[
            html.P('Human ID:'),
            dbc.Spinner(dcc.Dropdown(
                id='view-human-id',
                options=db_reid.get_human_id_options(),
            )),
            html.Br(),
            dbc.Button(children='Refresh', id='view-btn-refresh',
                       color='primary', block=True, size="lg"),
        ],
        width=1,
        style=LEFT_SIDEBAR_STYLE,
    )

    center_content = dbc.Col(
        id='display-col',
        children=[
            dbc.Spinner(dbc.Row(
                id='view-images-row',
                form=True,
                style={'display': 'flex',
                       'flex-wrap': 'wrap', 'overflow': 'auto'},
                no_gutters=True,
                #fluid=True,
            )),
        ],
        width=9,
        align='stretch'
    )


    global df_mislabelled
    table = dbc.Row(
        id='table-mislabelled', form=True, no_gutters=True,
        children=dbc.Table.from_dataframe(df_mislabelled[['img_id']], striped=True, bordered=True, hover=True))
    save_row = dbc.Row(
        children=[
            dbc.Col(html.H6(html.U('Changes'))),
            dbc.Button(children='Save', id='btn-save', disabled=True, color='primary', size="lg")
        ],
        no_gutters=True,
        justify='end'
    )

    right_sidebar = dbc.Col(
        id='right-sidebar',
        children=[save_row, table],
        width=2,
        style=RIGHT_SIDEBAR_STYLE,
    )
    return dbc.Row(children=[
        left_sidebar,
        center_content,
        right_sidebar
    ])


@app.callback(
    Output(component_id='view-images-row', component_property='children'),
    Input(component_id='view-btn-refresh', component_property='n_clicks'),
    State(component_id='view-human-id', component_property='value'),
)
def update_view_images(btn_clicks, human_id):
    # db_reid.get_images(human_id=human_id)
    df_images = df_reid[df_reid.human_id == human_id]
    df_images = df_images.sort_values(by=['face_score'], ascending= False)
    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    images_col = []
    global df_mislabelled
    for index, row in df_images.iterrows():
        #encoded_image = cv2.imdecode(np.frombuffer(df_images['img'].iloc[i], np.uint8), cv2.IMREAD_ANYCOLOR)
        #lab = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2LAB)
        #l, a, b = cv2.split(lab)
        #cl = clahe.apply(l)
        #limg = cv2.merge((cl, a, b))
        #encoded_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        #_, buffer = cv2.imencode('.png', encoded_image)
        #encoded_image = base64.b64encode(buffer)
        #encoded_image = base64.b64encode(row.img)
        style = {}
        if not df_mislabelled.empty and \
            not df_mislabelled[df_mislabelled.img_id == row.img_id].empty and \
            df_mislabelled[df_mislabelled.img_id == row.img_id].iloc[0]['is_mislabelled']:
            style['opacity'] = '0.3'
        images_col.append(
            dbc.Col(
                children=[
                    dbc.Spinner(html.A(
                        id={'index': row.img_id, 'type': 'image'},
                        key=row.img_id,
                        style=style,
                        children = [
                            html.P(row.img_id,
                                   style={'white-space': 'nowrap', 'overflow': 'hidden',
                                          'text-overflow': 'ellipsis', 'width': '8vw', 'margin': '0'}),
                            #html.P(df_images.face_score.iloc[i],
                            #       style={'white-space': 'nowrap', 'overflow': 'hidden',
                            #              'text-overflow': 'ellipsis', 'width': '100px', 'margin': '0'}),
                            html.Img(
                                src=row.img_decoded,
                                title=row.img_id,
                                #id=img_id,
                                style={'width': '8vw', 'object-fit':'contain'}
                                )
                            ]
                    ))
                ],
                width='auto',
                align='start',
            ))

    logs_text = df_images[['img_id', 'human_id',
                           'inference_datetime']].to_string(justify='center')
    arrClicks = None

    return images_col


@app.callback(
    Output({'type': 'image', 'index': MATCH}, 'style'),
    Input({'type': 'image', 'index': MATCH}, 'n_clicks'),
    Input({'type': 'image', 'index': MATCH}, 'style'),
)
def display_output(n_clicks, style):
    if n_clicks:
        if 'opacity' in style:
            del style['opacity']
        else:
            style['opacity'] = '0.3'
    return style


@app.callback(
    Output('table-mislabelled', 'children'),
    #Output('btn-save', 'disabled'),
    Input({'type': 'image', 'index': ALL}, 'style'),
    State({'type': 'image', 'index': ALL}, 'key'),
    State('table-mislabelled', 'children'),
)
def display_selected(style, key, table):
    global df_mislabelled, df_reid, df_mislabelled_orig
    rows = []
    for idx, item in enumerate(style):
        isMislabelled = False
        if 'opacity' in item and item['opacity']=='0.3':
            isMislabelled = True
        human_id = df_reid[df_reid.img_id == key[idx]].iloc[0]['human_id']
        rows.append({'img_id': key[idx], 'is_mislabelled': isMislabelled, 'human_id':human_id})

    if rows:
        df_mislabelled = df_mislabelled.append(rows, ignore_index=True)
        df_mislabelled = df_mislabelled.drop_duplicates('img_id', keep='last')
        df_mislabelled = df_mislabelled.sort_values('img_id')
    if not df_mislabelled.empty:
        difference = pd.concat([df_mislabelled_orig, df_mislabelled]).drop_duplicates(keep=False).drop_duplicates('img_id', keep='last')
        table = dbc.Table.from_dataframe(df_mislabelled[df_mislabelled.img_id.isin(
            key) & df_mislabelled.img_id.isin(difference.img_id)][['img_id']], striped=True, bordered=True, hover=True)
    disabled = difference.empty

    return table, disabled


@app.callback(
    Output('btn-save', 'disabled'),
    Output('btn-save', 'n_clicks'),
    Input('table-mislabelled', 'children'),
    Input('btn-save', 'n_clicks'),
)
def save_mislabelled(children, n_clicks):
    global db_mislabelled, df_mislabelled, df_mislabelled_orig
    if n_clicks is not None:
        db_mislabelled.save_mislabelled(df_mislabelled)
        df_mislabelled = db_mislabelled.get_mislabelled()
        df_mislabelled_orig = df_mislabelled.copy()
    difference = pd.concat([df_mislabelled_orig, df_mislabelled]).drop_duplicates(keep=False)
    disabled = difference.empty

    return disabled, None
