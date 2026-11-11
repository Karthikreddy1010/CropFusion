# Start the CornSight API (port 8010) and web app (port 5180) in two new windows.
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\backend'; .\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8010"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"
Start-Sleep -Seconds 4
Start-Process "http://localhost:5180"
