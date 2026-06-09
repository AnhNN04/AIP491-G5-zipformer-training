from fastapi import FastAPI

app = FastAPI(title="VietASR Model Serving API")

@app.get("/health")
def health_check():
    return {"status": "ok"}
