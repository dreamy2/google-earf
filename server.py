from flask import Flask, jsonify, send_file
from PIL import Image
import numpy as np
import requests
import io
import math
import os

app = Flask(__name__)

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN")

def lat_lon_to_tile(lat, lon, zoom):
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y

def fetch_tile(tileset, z, x, y):
    url = f"https://api.mapbox.com/v4/{tileset}/{z}/{x}/{y}.pngraw?access_token={MAPBOX_TOKEN}"
    r = requests.get(url)
    return Image.open(io.BytesIO(r.content))

def decode_terrain_rgb(img, resolution=64):
    img = img.resize((resolution, resolution)).convert("RGB")
    pixels = np.array(img, dtype=np.float32)
    R, G, B = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
    heights = -10000 + ((R * 65536 + G * 256 + B) * 0.1)
    h_min, h_max = heights.min(), heights.max()
    if h_max == h_min:
        return np.zeros((resolution, resolution)).tolist()
    normalized = (heights - h_min) / (h_max - h_min) * 500
    return normalized.tolist()

@app.route('/terrain/<float:lat>/<float:lon>/<int:zoom>')
def get_terrain(lat, lon, zoom):
    x, y = lat_lon_to_tile(lat, lon, zoom)
    img = fetch_tile("mapbox.terrain-rgb", zoom, x, y)
    heights = decode_terrain_rgb(img, resolution=64)
    return jsonify({
        "heights": heights,
        "size": 64,
        "tile": {"z": zoom, "x": x, "y": y}
    })

@app.route('/satellite/<float:lat>/<float:lon>/<int:zoom>')
def get_satellite(lat, lon, zoom):
    x, y = lat_lon_to_tile(lat, lon, zoom)
    img = fetch_tile("mapbox.satellite", zoom, x, y)
    img = img.resize((256, 256)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)