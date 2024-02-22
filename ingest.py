import argparse
import base64
from io import BytesIO

from PIL import Image

from pathlib import Path
from tqdm import tqdm

from txt2image.txt2image_searcher import Txt2ImageSearcher


def read_image(image_path):
    image = Image.open(image_path)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return img_str


def ingest(path, searcher):
    pathlist = Path(path).glob('**/*')
    for path in tqdm(pathlist):
        if path.suffix not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        path_in_str = str(path)
        searcher.ingest_image(read_image(path_in_str))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--collection', type=str, help='Collection name', required=True)
    parser.add_argument('-u', '--url', type=str, help='HTTP Url of the Qdrant instance (default: http://localhost:6333)', default="http://localhost:6333")
    parser.add_argument('-a', '--api', type=str, help='Qdrant API key (default: none)',
                        default=None)
    parser.add_argument('-i', '--images', type=str, help='Directory with images', required=True)
    parser.add_argument('-d', '--device', type=str, help='Device to run to model on (e.g. a GPU) (default: cuda)',
                        default='cuda')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    searcher = Txt2ImageSearcher(args.collection, qdrant_url=args.url, device=args.device, api_key=args.api)

    ingest(args.images, searcher)
