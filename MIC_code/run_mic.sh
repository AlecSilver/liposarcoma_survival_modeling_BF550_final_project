#!/bin/bash

# docker pull minepy/mictools

MSYS_NO_PATHCONV=1 docker run -v C:/Users/sherh/Documents/bf550/BF550_final_project/mic_calulations.py:/app/mic_calulations.py\
           -v C:/Users/sherh/Documents/bf550/BF550_final_project/data/combined_metadata.csv:/app/data/combined_metadata.csv \
           -v C:/Users/sherh/Documents/bf550/BF550_final_project/data/combined_expression_data_scaled.csv:/app/data/combined_expression_data_scaled.csv \
           -v C:/Users/sherh/Documents/bf550/BF550_final_project/data/mic:/app/data/mic \
           sha256:c5b919e0f46126fbd8851195cd3495833162c229ede699b21577979896285852 python3 /app/mic_calulations.py