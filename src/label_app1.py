import os
import logging
from demo_dash import label_dash1 as dash_app


if __name__ == '__main__':
    #print(os.getcwd())
    dash_app.app.run_server(host="0.0.0.0", debug=True)
