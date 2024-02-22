import uuid
import base64
from io import BytesIO

from PIL import Image

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from transformers import CLIPProcessor, CLIPModel


class Txt2ImageSearcher:
    def __init__(self, collection_name, qdrant_url='http://127.0.0.1:6333', model_name='openai/clip-vit-large-patch14',
                 device='cuda', api_key=None):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device=self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.client = QdrantClient(qdrant_url, api_key=api_key)

        self.vector_size = self.model.config.projection_dim

        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _embed_image(self, image_base64):
        image = Image.open(BytesIO(base64.b64decode(image_base64)))

        inputs = self.processor(images=image, return_tensors="pt",
                                padding=True).to(device=self.device)

        image_embeddings = self.model.get_image_features(**inputs)

        return image_embeddings.squeeze().cpu().tolist()

    def _embed_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt",
                                padding=True).to(device=self.device)

        image_embeddings = self.model.get_text_features(**inputs)

        return image_embeddings.squeeze().cpu().tolist()

    def _upsert_image_point(self, image, embedding):
        image_base64 = image
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"image": image_base64}
                )
            ]
        )

    def ingest_image(self, image):
        embeds = self._embed_image(image)

        self._upsert_image_point(image, embeds)

    def search_image(self, query_text):
        vector = self._embed_text(query_text)

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            limit=1,
        )

        payloads = [hit.payload['image'] for hit in search_result]
        return payloads

    def retrieve_image(self, image_id):
        payload = self.client.retrieve(collection_name=self.collection_name, ids=[image_id])
        if len(payload) == 0:
            return None
        return payload[0]['image']


if __name__ == "__main__":
    searcher = Txt2ImageSearcher("test_collection")

    dog_results = searcher.search_image("A dog")
    image = Image.open(BytesIO(base64.b64decode(dog_results[0])))
    image.show()
    pass
