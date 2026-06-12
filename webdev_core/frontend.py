import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Aero-Twin: Turbofan Digital Twin UI")

# Get path of static directory relative to this file
base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")

# Ensure static folder exists
os.makedirs(static_dir, exist_ok=True)

# Mount static assets
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def get_dashboard():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            "<h3>Dashboard Loading... Static resources not yet created.</h3>", 
            status_code=503
        )
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("Starting Aero-Twin Dashboard Server on http://localhost:8080...")
    uvicorn.run("frontend:app", host="0.0.0.0", port=8080, reload=True)
