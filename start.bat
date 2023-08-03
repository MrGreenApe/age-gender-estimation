@echo off

:: Set path to project
cd [Path to project]

:: Activate virtual environment
conda activate age-gender-estimation

:: Run flask app
python .\serve.py