#!/bin/bash
pip uninstall -y opencv-python
gunicorn --bind=0.0.0.0 --timeout 600 web:app
