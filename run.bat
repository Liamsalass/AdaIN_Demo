@echo off

call conda env list | findstr /C:"ELEC475_Demo" 1>nul
if errorlevel 1 (
    echo "Creating new environment"
    call conda env create -f env.yml
) else (
    echo "Activating existing environment"
)

call conda activate ELEC475_Demo

python demo.py