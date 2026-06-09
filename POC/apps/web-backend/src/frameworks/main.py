from fastapi import FastAPI

app = FastAPI(title="VietASR Web API Gateway")

@app.get("/health")
def health_check():
    return {"status": "ok"}
