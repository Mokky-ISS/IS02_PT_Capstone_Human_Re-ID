import logging
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from demo_dash import header
from urllib.parse import parse_qs
import base64
import cv2
import mediapipe as mp

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


    layout_page = []
    try:
        layout_page.append(
            v1_page_content())
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

DETECT_OPTIONS = [
    {'label': 'Face Detection', 'value': 'face'},
    {'label': 'Face Mesh', 'value': 'face_mesh'},
    {'label': 'Hands', 'value': 'hands'},
    {'label': 'Pose', 'value': 'pose'},
    {'label': 'Holistic', 'value': 'holistic'},
]

def v1_page_content():
    sidebar = dbc.Col(
        id='page-sidebar',
        children=[
            html.P('Images directory:'),
            dbc.Input(
                id='images-directory',
                type='text',
            ),
            html.Br(),
            html.P('Name:'),
            dbc.Spinner(dcc.Dropdown(id='names-dropdown',)),
            html.Br(),
            html.P('Detection:'),
            dbc.Spinner(dcc.Dropdown(id='detect-dropdown', options=DETECT_OPTIONS)),
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
                    id='images_row',
                    form=True,
                    style={'flex-wrap': 'nowrap', 'overflow':'auto'})),
                html.Br(),
                html.H5('Images filename:'),
                dbc.Spinner(dbc.Row(
                    id='logs_row',
                    children=
                        dcc.Textarea(
                            id='logs_text',
                            disabled=True,
                            style={'width': '100%','height': '200px'}
                            ),
                        )),
                ],
            width=10,
        )
    ])


@app.callback(
    Output(component_id='names-dropdown', component_property='options'),
    Input(component_id='images-directory', component_property='value'),)
def update_names(images_directory):
    options = []
    try:
        for root, directories, _ in os.walk(images_directory, topdown=False):
            for name in directories:
                options.append({'label': name, 'value': os.path.join(root, name)})
        return options
    except Exception as ex:
        logging.error(f'update_names: {ex}')
        return []


@app.callback(
    Output(component_id='images_row', component_property='children'),
    Output(component_id='logs_text', component_property='value'),
    Input(component_id='names-dropdown', component_property='value'),
    Input(component_id='detect-dropdown', component_property='value'),
)
def update_images(image_path, detect):
    if image_path is not None:
        images_col = []
        logs_text = ''
        try:
            for root, _, files in os.walk(image_path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        img = cv2.imread(file_path)
                        if detect == 'face':
                            img = detect_face(img)
                        elif detect == 'face_mesh':
                            img = detect_face_mesh(img)
                        elif detect == 'hands':
                            img = detect_hands(img)
                        elif detect == 'pose':
                            img = detect_pose(img)
                        elif detect == 'holistic':
                            img = detect_holistic(img)
                        _, img_buffer = cv2.imencode('.png', img)
                        encoded_image = base64.b64encode(img_buffer)# open(file_path, 'rb').read())
                        images_col.append(
                            dbc.Col(html.Img(
                                src='data:image/png;base64,{}'.format(encoded_image.decode()),
                                title=file,
                                )))
                        if len(logs_text) > 0:
                            logs_text += '\n'
                        logs_text += file
                    except:
                        pass
            return images_col, logs_text
        except Exception as ex:
            logging.error(f'update_images: {ex}')
    return [],[]

def detect_face(img):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        annotated_image = img.copy()

        try:
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            for detection in results.detections:
                #print('Nose tip:')
                #print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawing.draw_detection(annotated_image, detection)
        except Exception as ex:
            logging.error(f'detect_face: {ex}')

        return annotated_image

def detect_face_mesh(img):
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        annotated_image = img.copy()

        try:
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACE_CONNECTIONS)
        except Exception as ex:
            logging.error(f'detect_face_mesh: {ex}')

        return annotated_image

def detect_hands(img):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        annotated_image = img.copy()

        try:
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        except Exception as ex:
            logging.error(f'detect_hands: {ex}')

        return annotated_image

def detect_pose(img):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        annotated_image = img.copy()

        try:
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Draw pose landmarks on the image.
            # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
            # upper_body_only is set to True.
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        except Exception as ex:
            logging.error(f'detect_pose: {ex}')

        return annotated_image

def detect_holistic(img):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        annotated_image = img.copy()

        try:
            # Convert the BGR image to RGB before processing.
            results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Draw pose, left and right hands, and face landmarks on the image.

            mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
            mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # Use mp_holistic.UPPER_BODY_POSE_CONNECTIONS for drawing below when
            # upper_body_only is set to True.
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        except Exception as ex:
            logging.error(f'detect_holistic: {ex}')

        return annotated_image
