import os
import math
import io

from flask import Flask, jsonify
from PIL import Image
import numpy as np
import requests
from scipy.ndimage import median_filter
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN")
SATELLITE_RES = 1024
TERRAIN_RES = 128
POI_SIZERANK_MAX = 8  # 0=biggest, 16=tiny. 8 = landmarks + medium pois

# shared thread pool for parallel mapbox calls
tile_pool = ThreadPoolExecutor(max_workers=16)

# shared session reuses tcp connections
http_session = requests.Session()

def lat_lon_to_tile(lat, lon, zoom):
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y

def fetch_tile(tileset, z, x, y):
    url = f"https://api.mapbox.com/v4/{tileset}/{z}/{x}/{y}.png?access_token={MAPBOX_TOKEN}"
    r = http_session.get(url, timeout=10)
    if r.status_code != 200:
        raise Exception(f"Mapbox error {r.status_code}: {r.text}")
    return Image.open(io.BytesIO(r.content))

def fetch_tiles_parallel(tileset, zoom, base_x, base_y, grid):
    jobs = []
    for row in range(grid):
        for col in range(grid):
            jobs.append((row, col, tile_pool.submit(fetch_tile, tileset, zoom, base_x + col, base_y + row)))
    return [(row, col, f.result()) for (row, col, f) in jobs]

def fetch_and_stitch_satellite(terrain_zoom, terrain_x, terrain_y):
    zoom_offset = int(math.log2(SATELLITE_RES // 256))
    grid = 2 ** zoom_offset
    sat_zoom = terrain_zoom + zoom_offset
    x = terrain_x * grid
    y = terrain_y * grid
    stitched = Image.new("RGB", (grid * 256, grid * 256))
    for row, col, tile in fetch_tiles_parallel("mapbox.satellite", sat_zoom, x, y, grid):
        stitched.paste(tile, (col * 256, row * 256))
    return stitched

def decode_terrain_rgb(img):
    img = img.convert("RGB")
    pixels = np.array(img, dtype=np.float32)
    R, G, B = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
    return -10000 + ((R * 65536 + G * 256 + B) * 0.1)

def fetch_and_stitch_terrain(zoom, x, y):
    zoom_offset = int(math.log2(max(TERRAIN_RES // 64, 1)))
    grid = 2 ** zoom_offset
    sat_zoom = zoom + zoom_offset
    tx = x * grid
    ty = y * grid
    tile_size = 64
    stitched = np.zeros((grid * tile_size, grid * tile_size), dtype=np.float32)
    for row, col, tile in fetch_tiles_parallel("mapbox.terrain-rgb", sat_zoom, tx, ty, grid):
        tile = tile.resize((tile_size, tile_size))
        h = decode_terrain_rgb(tile)
        stitched[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size] = h
    return stitched

def smooth_heights(heights, passes=3):
    for _ in range(passes):
        heights = median_filter(heights, size=5)
    return heights

# single tilequery call - simple http, no heavy deps
def tilequery_at(qlat, qlon):
    url = f"https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/tilequery/{qlon},{qlat}.json"
    try:
        r = http_session.get(url, params={
            "access_token": MAPBOX_TOKEN,
            "radius": 1500,
            "limit": 30,
            "layers": "poi_label,airport_label,natural_label",
        }, timeout=5)
        if r.status_code != 200:
            return []
        return r.json().get("features", [])
    except Exception:
        return []

@app.route('/terrain/<lat>/<lon>/<zoom>')
def get_terrain(lat, lon, zoom):
    lat, lon, zoom = float(lat), float(lon), int(zoom)
    x, y = lat_lon_to_tile(lat, lon, zoom)
    heights = fetch_and_stitch_terrain(zoom, x, y)
    heights = smooth_heights(heights, passes=3)
    h_img = Image.fromarray(heights)
    h_img = h_img.resize((TERRAIN_RES, TERRAIN_RES), Image.BILINEAR)
    heights = np.array(h_img)
    return jsonify({
        "heights": heights.tolist(),
        "size": TERRAIN_RES,
    })

@app.route('/satellite/<lat>/<lon>/<zoom>')
def get_satellite(lat, lon, zoom):
    lat, lon, zoom = float(lat), float(lon), int(zoom)
    x, y = lat_lon_to_tile(lat, lon, zoom)
    img = fetch_and_stitch_satellite(zoom, x, y)
    img = img.resize((SATELLITE_RES, SATELLITE_RES)).convert("RGBA")
    flat = list(img.tobytes())
    return jsonify({"pixels": flat, "size": SATELLITE_RES})

@app.route('/pois/<lat>/<lon>/<zoom>')
def get_pois(lat, lon, zoom):
    lat, lon, zoom = float(lat), float(lon), int(zoom)

    # tile span in degrees (approx, good enough for grid placement)
    n = 2 ** zoom
    tile_deg = 360 / n

    # fire off a 3x3 grid of tilequery requests to cover the whole tile
    # each query covers ~1.5km radius, 3x3 grid covers a zoom-12 tile well
    jobs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            qlat = lat + dy * tile_deg / 3
            qlon = lon + dx * tile_deg / 3
            jobs.append(tile_pool.submit(tilequery_at, qlat, qlon))

    # dedupe by feature id
    seen_ids = set()
    pois = []
    for job in jobs:
        for f in job.result():
            fid = f.get("id")
            if fid is not None and fid in seen_ids:
                continue
            if fid is not None:
                seen_ids.add(fid)

            props = f.get("properties", {})
            sizerank = props.get("sizerank", 16)
            if sizerank > POI_SIZERANK_MAX:
                continue

            name = props.get("name") or props.get("name_en") or props.get("ref")
            if not name:
                continue

            geom = f.get("geometry", {})
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates", [0, 0])

            # category comes from either the class field or the source layer
            layer = props.get("tilequery", {}).get("layer", "")
            if layer == "airport_label":
                category = "airport"
            elif layer == "natural_label":
                category = "natural"
            else:
                category = props.get("class", "general")

            pois.append({
                "name": name,
                "lat": coords[1],
                "lon": coords[0],
                "category": category,
                "sizerank": sizerank,
            })

    pois.sort(key=lambda p: p["sizerank"])
    return jsonify({"pois": pois})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)