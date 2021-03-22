import os
import logging
from demo_dash import dash_app


if __name__ == '__main__':
    dash_app.app.run_server(host="0.0.0.0", debug=True)
