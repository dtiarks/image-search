import base64
from io import BytesIO
import requests

import gradio as gr
from PIL import Image

TXT2IMAGE_URL = "http://0.0.0.0:8000"


def search_image(query):
    url_search = f"{TXT2IMAGE_URL}/api/search?q={query}"

    response = requests.request("GET", url_search)

    results = response.json()

    return results['image']


def handle_search(image_state, text):
    images_base64 = search_image(text)
    if len(images_base64) == 0:
        return (image_state)

    image = Image.open(BytesIO(base64.b64decode(images_base64[0])))

    return (image)


with gr.Blocks(title="Qdrant image search demo") as demo:
    state = gr.State()

    gr.Markdown("# Finding images by text similarity")
    with gr.Row():
        with gr.Column(scale=2):
            imagebox = gr.Image(type="pil", height=900)
            textbox = gr.Textbox(label="Search text")
            btn = gr.Button("Search")

    btn.click(
        handle_search,
        [imagebox, textbox],
        [imagebox]
    )

demo.launch()
