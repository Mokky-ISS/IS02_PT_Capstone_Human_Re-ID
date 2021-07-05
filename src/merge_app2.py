import os
import logging
from demo_dash import merge_dash2a as dash_app


if __name__ == '__main__':
    #print(os.getcwd())
    dash_app.app.run_server(host="0.0.0.0", debug=True)
