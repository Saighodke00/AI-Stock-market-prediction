@echo off
echo Starting Stock Market App...

:: Check if venv exists
if not exist "venv" (
    echo Virtual environment not found! Please create it first.
    pause
    exit /b
)

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Install dependencies to ensure everything is ready
echo Checking dependencies...
pip install -r requirements.txt

:: Run the Streamlit app
echo Launching Streamlit...
streamlit run app.py

pause
