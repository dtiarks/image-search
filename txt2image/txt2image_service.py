from typing import Optional

from fastapi import FastAPI
from pydantic_settings import BaseSettings

from txt2image_searcher import Txt2ImageSearcher


class Settings(BaseSettings):
    collection: str
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    device: str = "cuda"

settings = Settings()
app = FastAPI()

searcher = Txt2ImageSearcher(settings.collection, qdrant_url=settings.url, api_key=settings.api_key, device=settings.device)


@app.get("/api/search")
def search_image(q: str):
    return {"image": searcher.search_image(q)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
