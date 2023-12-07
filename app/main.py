import uvicorn
from fastapi import FastAPI
from router.router import router

app = FastAPI()

app.include_router(router)

def start_uvicorn():
    uvicorn.run("main:app", host="localhost", port=8001, reload=True)

if __name__ == "__main__":
    start_uvicorn()
    
# uvicorn main:app --host localhost --port 8001 --reload