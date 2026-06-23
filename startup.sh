#!/bin/bash
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python-headless
gunicorn --bind=0.0.0.0 --timeout 600 web:app
