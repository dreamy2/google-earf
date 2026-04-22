import os
import math
import io
import gc

from flask import Flask, jsonify
from PIL import Image
import numpy as np
import requests
from scipy.ndimage import median_filter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

app = Flask(__name__)

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN")
SATELLITE_RES = 1024
TERRAIN_RES = 128
POI_SIZERANK_MAX = 10  # 1-16 scale, lower = more prominent; 10 keeps major + significant only

# small pool so we dont blow railway memory
tile_pool = ThreadPoolExecutor(max_workers=4)
http_session = requests.Session()

# cache raw tile bytes so neighbor tiles reuse each others fetches
# 256 tiles ~= 25mb, safe for railway free tier
@lru_cache(maxsize=256)
def _cached_tile_bytes(url):
    r = http_session.get(url, timeout=10)
    if r.status_code != 200:
        raise Exception(f"Mapbox error {r.status_code}")
    return r.content

def fetch_tile_bytes(tileset, z, x, y):
    url = f"https://api.mapbox.com/v4/{tileset}/{z}/{x}/{y}.png?access_token={MAPBOX_TOKEN}"
    return _cached_tile_bytes(url)

def lat_lon_to_tile(lat, lon, zoom):
    lat_r = math.radians(lat)
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y

def tile_to_lat_lon(x, y, zoom):
    # inverse of lat_lon_to_tile, returns top-left corner of the tile
    n = 2 ** zoom
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon

def fetch_tiles_parallel(tileset, zoom, base_x, base_y, grid):
    jobs = []
    for row in range(grid):
        for col in range(grid):
            jobs.append((row, col, tile_pool.submit(fetch_tile_bytes, tileset, zoom, base_x + col, base_y + row)))
    return jobs

def fetch_and_stitch_satellite(terrain_zoom, terrain_x, terrain_y):
    # higher zoom tiles give real extra detail for the same area
    zoom_offset = int(math.log2(SATELLITE_RES // 256))
    grid = 2 ** zoom_offset
    sat_zoom = terrain_zoom + zoom_offset
    x = terrain_x * grid
    y = terrain_y * grid

    jobs = fetch_tiles_parallel("mapbox.satellite", sat_zoom, x, y, grid)
    stitched = Image.new("RGB", (grid * 256, grid * 256))
    for (row, col, fut) in jobs:
        with Image.open(io.BytesIO(fut.result())) as tile:
            stitched.paste(tile, (col * 256, row * 256))
    return stitched

def decode_terrain_rgb(img):
    img = img.convert("RGB")
    pixels = np.asarray(img, dtype=np.float32)
    R, G, B = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
    return -10000 + ((R * 65536 + G * 256 + B) * 0.1)

def fetch_terrain_grid(zoom, x, y):
    # fetches a single mapbox tile's heightmap at the target zoom
    zoom_offset = int(math.log2(max(TERRAIN_RES // 64, 1)))
    sat_zoom = zoom + zoom_offset
    grid = 2 ** zoom_offset
    tx = x * grid
    ty = y * grid
    tile_size = 64

    jobs = fetch_tiles_parallel("mapbox.terrain-rgb", sat_zoom, tx, ty, grid)
    stitched = np.zeros((grid * tile_size, grid * tile_size), dtype=np.float32)
    for (row, col, fut) in jobs:
        with Image.open(io.BytesIO(fut.result())) as tile:
            tile = tile.resize((tile_size, tile_size))
            stitched[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size] = decode_terrain_rgb(tile)
    return stitched

def smooth_heights(heights, passes=3):
    # median filter kills building/noise spikes without flattening real hills
    for _ in range(passes):
        heights = median_filter(heights, size=5)
    return heights

def tilequery_at(qlat, qlon):
    url = f"https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/tilequery/{qlon},{qlat}.json"
    try:
        r = http_session.get(url, params={
            "access_token": MAPBOX_TOKEN,
            "radius": 1500,
            "limit": 50,  # was 30, more room per query point
            "layers": "poi_label,airport_label,natural_label",
        }, timeout=5)
        if r.status_code != 200:
            return []
        return r.json().get("features", [])
    except Exception:
        return []

@app.route('/terrain/<lat>/<lon>/<zoom>')
def get_terrain(lat, lon, zoom):
    try:
        lat, lon, zoom = float(lat), float(lon), int(zoom)
        x, y = lat_lon_to_tile(lat, lon, zoom)
        heights = fetch_terrain_grid(zoom, x, y)
        heights = smooth_heights(heights, passes=3)

        # float32 needs explicit mode F for PIL
        h_img = Image.fromarray(heights.astype(np.float32), mode='F')
        h_img = h_img.resize((TERRAIN_RES, TERRAIN_RES), Image.BILINEAR)
        heights = np.asarray(h_img)

        result = {"heights": heights.tolist(), "size": TERRAIN_RES}
        del h_img
        gc.collect()
        return jsonify(result)
    except Exception as e:
        print(f"[terrain] error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/satellite/<lat>/<lon>/<zoom>')
def get_satellite(lat, lon, zoom):
    try:
        lat, lon, zoom = float(lat), float(lon), int(zoom)
        x, y = lat_lon_to_tile(lat, lon, zoom)
        img = fetch_and_stitch_satellite(zoom, x, y)
        img = img.resize((SATELLITE_RES, SATELLITE_RES)).convert("RGBA")
        flat = list(img.tobytes())
        img.close()
        del img
        gc.collect()
        return jsonify({"pixels": flat, "size": SATELLITE_RES})
    except Exception as e:
        print(f"[satellite] error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/pois/<lat>/<lon>/<zoom>')
def get_pois(lat, lon, zoom):
    try:
        lat, lon, zoom = float(lat), float(lon), int(zoom)
        x, y = lat_lon_to_tile(lat, lon, zoom)

        # exact geographic bounds of this tile, used to reject overfetched pois
        lat_top, lon_left = tile_to_lat_lon(x, y, zoom)
        lat_bot, lon_right = tile_to_lat_lon(x + 1, y + 1, zoom)

        # tile's lat/lon spans (not equal — mercator squishes lat at high latitudes)
        lon_span = lon_right - lon_left
        lat_span = lat_top - lat_bot

        # 3x3 grid of tilequeries covers a zoom-12 tile
        # use each axis's real span so points stay inside the tile
        jobs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                qlat = lat + dy * lat_span / 3
                qlon = lon + dx * lon_span / 3
                jobs.append(tile_pool.submit(tilequery_at, qlat, qlon))

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
                plon, plat = coords[0], coords[1]

                # drop anything outside the requested tile's bounds
                if not (lat_bot <= plat <= lat_top and lon_left <= plon <= lon_right):
                    continue

                layer = props.get("tilequery", {}).get("layer", "")
                if layer == "airport_label":
                    category = "airport"
                elif layer == "natural_label":
                    category = "natural"
                else:
                    category = props.get("class", "general")

                pois.append({
                    "name": name,
                    "lat": plat,
                    "lon": plon,
                    "category": category,
                    "sizerank": sizerank,
                })

        # dedup by name — creeks/rivers get labeled at many points, same name
        # keep the best-ranked (lowest sizerank) instance per name
        best_by_name = {}
        for p in pois:
            existing = best_by_name.get(p["name"])
            if existing is None or p["sizerank"] < existing["sizerank"]:
                best_by_name[p["name"]] = p
        pois = list(best_by_name.values())

        pois.sort(key=lambda p: p["sizerank"])
        return jsonify({"pois": pois})
    except Exception as e:
        print(f"[pois] error: {e}")
        return jsonify({"pois": []}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)