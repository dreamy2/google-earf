import os
import base64
import math
import io

from flask import Flask, jsonify
from PIL import Image
import numpy as np
import requests

app = Flask(__name__)

SATELLITE_RES = 512

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN")

def lat_lon_to_tile(lat, lon, zoom):
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y

def fetch_tile(tileset, z, x, y):
    url = f"https://api.mapbox.com/v4/{tileset}/{z}/{x}/{y}.png?access_token={MAPBOX_TOKEN}"
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Mapbox error {r.status_code}: {r.text}")
    return Image.open(io.BytesIO(r.content))

def decode_terrain_rgb(img, resolution=64):
    img = img.resize((resolution, resolution)).convert("RGB")
    pixels = np.array(img, dtype=np.float32)
    R, G, B = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
    heights = -10000 + ((R * 65536 + G * 256 + B) * 0.1)
    return heights.tolist()

@app.route('/terrain/<lat>/<lon>/<zoom>')
def get_terrain(lat, lon, zoom):
    lat, lon, zoom = float(lat), float(lon), int(zoom)
    x, y = lat_lon_to_tile(lat, lon, zoom)
    img = fetch_tile("mapbox.terrain-rgb", zoom, x, y)
    heights = decode_terrain_rgb(img, resolution=64)
    return jsonify({
        "heights": heights,
        "size": 64,
        "tile": {"z": zoom, "x": x, "y": y}
    })

@app.route('/satellite/<lat>/<lon>/<zoom>')
def get_satellite(lat, lon, zoom):
    lat, lon, zoom = float(lat), float(lon), int(zoom)
    x, y = lat_lon_to_tile(lat, lon, zoom)
    img = fetch_tile("mapbox.satellite", zoom, x, y)
    img = img.resize((SATELLITE_RES, SATELLITE_RES)).convert("RGBA")
    flat = [v for px in img.getdata() for v in px]
    return jsonify({"pixels": flat, "size": SATELLITE_RES})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)