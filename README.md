# Multi-modal Student Classroom Behavior Assisted Teaching System

This is a implementation of Multi-modal Student Classroom Behavior Assisted Teaching System, and the code includes the following modules:

- Data(student classroom behavior datasets from https://universe.roboflow.com/mywork-lkwz4/student-behaviour-detection-neazg/dataset/6)
- Classroom Behavior Recognition using YOLOv8
- Prompt-based Behavior Understanding using LLM

## Main Requirements

- ultralytics 8.3.79
- ultralytics-thop 2.0.14
- PyQt5 5.15.11
- spark-ai-python 0.4.5

## Description

- Vison.py
  - Main script that handles the YOLOv8 model training and evaluation process.
- Cognition.py
  - Behavior event logs detected by YOLOv8 are formatted into prompt templates and fed into LLM to interpret classroom dynamics.

## Running the code

1. Install the required dependency packages
2. To reproduce the results, please use the command `python Vison.py & python Cognition.py`