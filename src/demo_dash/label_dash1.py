import logging
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH, ALL
from demo_dash import header_label1 as header
from database import query_re_id, query_correctlabel
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
db_correctlabel = query_correctlabel.DbQuery()
df_correctlabel = db_correctlabel.get_correctlabel()
df_correctlabel_orig = df_correctlabel.copy()

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
            dbc.Spinner(dcc.Input(
                id='view-human-id',
                value=0,
                type="number",
                min = 0,
                step = 1,
                max = 9999,
            )),
            html.Br(),
            dbc.Button(children='Refresh', id='view-btn-refresh',
                       color='primary', block=True, size="lg"),
            html.Hr(),
            dbc.Button(children='Clear Selection',
                       id='btn-clear-selection',
                               color='secondary', block=True, size="lg",),


        ],
        width='auto',
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
        width=True,
        align='stretch'
    )

    global df_correctlabel, df_correctlabel_orig
    df_difference = pd.concat([df_correctlabel_orig, df_correctlabel], ignore_index=True).drop_duplicates(
        keep=False).drop_duplicates('img_id', keep='last')
    table = dbc.Row(
        id='table-correctlabel',
        children=dbc.Table.from_dataframe(df_difference[['img_id', 'is_correct']].astype(str), striped=True, bordered=True, hover=True),
        form=True, no_gutters=True, justify='stretch')
    save_row = dbc.Row(
        children=[
            dbc.Col(html.H6(html.U('Changes'))),
            dbc.Button(children='Save', id='btn-save', disabled=True, color='primary', size="lg")
        ],
        no_gutters=True,
        justify='end'
    )
    reset_row = dbc.Row(
        children=dbc.Button(children='Reset Database', id='btn-reset-db',
                            disabled=df_correctlabel_orig.empty, color='secondary', size="lg"),
        no_gutters=True,
        justify='end'
    )

    right_sidebar = dbc.Col(
        id='right-sidebar',
        children=[save_row, table, reset_row],
        width=3,
        style=RIGHT_SIDEBAR_STYLE,
    )
    return dbc.Row(children=[
        left_sidebar,
        center_content,
        right_sidebar
    ])

clear_clicks = None
@app.callback(
    Output(component_id='view-images-row', component_property='children'),
    Input(component_id='view-btn-refresh', component_property='n_clicks'),
    Input(component_id='btn-clear-selection', component_property='n_clicks'),
    State(component_id='view-human-id', component_property='value'),
)
def update_view_images(refresh_clicks, clear_n_clicks, human_id):
    global df_correctlabel, df_correctlabel_orig
    global clear_clicks

    human_id = str(human_id)

    if clear_n_clicks != clear_clicks:
        print(clear_n_clicks, clear_clicks)
        clear_clicks = clear_n_clicks
        df_correctlabel.loc[df_correctlabel.human_id ==
                            human_id, 'is_correct'] = False

    # db_reid.get_images(human_id=human_id)
    df_images = df_reid[df_reid.human_id == human_id]
    df_images = df_images.sort_values(by=['face_score'], ascending= False)
    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    images_col = []
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
        if not df_correctlabel.empty and \
                row.img_id in df_correctlabel.img_id.unique() and \
                df_correctlabel.loc[df_correctlabel.img_id == row.img_id, 'is_correct'].values[0]:
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

    #df_correctlabel[df_correctlabel.img_id != human_id] = df_correctlabel_orig[df_correctlabel_orig.img_id != human_id]
    df_correctlabel = pd.concat([df_correctlabel[df_correctlabel.img_id == human_id],
                         df_correctlabel_orig[df_correctlabel_orig.img_id != human_id]],
                         ignore_index=True).drop_duplicates(keep=False).drop_duplicates('img_id', keep='last')
    print('Length:',len(df_correctlabel))
    print(df_correctlabel.nunique())
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

save_clicks = None
reset_clicks = None
@app.callback(
    Output('table-correctlabel', 'children'),
    Output('btn-save', 'disabled'),
    Output('btn-reset-db', 'disabled'),
    Input({'type': 'image', 'index': ALL}, 'style'),
    Input('btn-save', 'n_clicks'),
    Input('btn-reset-db', 'n_clicks'),
    State({'type': 'image', 'index': ALL}, 'key'),
    State('table-correctlabel', 'children'),
)
def display_selected(style, save_n_clicks, reset_n_clicks, key, table):
    global db_correctlabel, df_correctlabel, df_reid, df_correctlabel_orig
    global save_clicks, reset_clicks

    if reset_n_clicks != reset_clicks:
        reset_clicks = reset_n_clicks
        db_correctlabel.reset_correctlabel()
        df_correctlabel = db_correctlabel.get_correctlabel()
        df_correctlabel_orig = df_correctlabel.copy()

    if save_n_clicks != save_clicks:
        save_clicks = save_n_clicks
        db_correctlabel.save_correctlabel(df_correctlabel)
        df_correctlabel = db_correctlabel.get_correctlabel()
        df_correctlabel_orig = df_correctlabel.copy()

    rows = []
    for idx, item in enumerate(style):
        isCorrectLabel = False
        if 'opacity' in item and item['opacity']=='0.3':
            isCorrectLabel = True
        human_id = df_reid[df_reid.img_id == key[idx]].iloc[0]['human_id']
        rows.append(
            {'img_id': key[idx], 'is_correct': isCorrectLabel, 'human_id': human_id})

    if rows:
        df_correctlabel = df_correctlabel.append(rows, ignore_index=True)
        df_correctlabel = df_correctlabel.drop_duplicates(
            'img_id', keep='last')
        df_correctlabel = df_correctlabel.sort_values('img_id')

    df_difference = pd.concat([df_correctlabel_orig, df_correctlabel], ignore_index=True).drop_duplicates(
        keep=False).drop_duplicates('img_id', keep='last')
    df_difference = df_difference[(df_difference.is_correct == True) | (
        df_difference.img_id.isin(df_correctlabel_orig.img_id))]
    disabled = df_difference.empty
    df_difference = df_difference[df_difference.img_id.isin(key)]

    table = dbc.Table.from_dataframe(df_difference[['img_id', 'is_correct']].astype(str), striped=True, bordered=True, hover=True)

    return table, disabled, df_correctlabel_orig.empty


#@app.callback(
#    Output('btn-save', 'disabled'),
#    Output('btn-save', 'n_clicks'),
#    Input('table-correctlabel', 'children'),
#    Input('btn-save', 'n_clicks'),
#)
#def save_correctlabel(children, n_clicks):
#   global db_correctlabel, df_correctlabel, df_correctlabel_orig
#    if n_clicks is not None:
#        db_correctlabel.save_correctlabel(df_correctlabel)
#        df_correctlabel = db_correctlabel.get_correctlabel()
#       df_correctlabel_orig = df_correctlabel.copy()
#    difference = pd.concat([df_correctlabel_orig, df_correctlabel], ignore_index=True).drop_duplicates(keep=False)
#    disabled = difference.empty

#    return disabled, None
