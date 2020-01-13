#!/usr/bin/env bash

python ExtractFeature.py > py_log/extractFeature.log

python Mosaic.py > py_log/mosaic.log

python Z_evalute.py > py_log/evalute.log