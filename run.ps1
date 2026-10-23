Set-Location 
Write-Host "Starting Propjini Face Verification Service..." -ForegroundColor Cyan
& .venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
