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
POI_SIZERANK_MAX = 8

# small pool so we dont blow railway memory with too many concurrent images
tile_pool = ThreadPoolExecutor(max_workers=4)
http_session = requests.Session()

# reuse raw tile bytes across requests for adjacent tiles
# small cache size keeps memory predictable (256 tiles ~= 25mb)
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

def fetch_tiles_parallel(tileset, zoom, base_x, base_y, grid):
    # submit raw-bytes fetches, then open and paste one at a time
    jobs = []
    for row in range(grid):
        for col in range(grid):
            jobs.append((row, col, tile_pool.submit(fetch_tile_bytes, tileset, zoom, base_x + col, base_y + row)))
    return jobs

def fetch_and_stitch_satellite(terrain_zoom, terrain_x, terrain_y):
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
    # fetches the native 64x64 heightmap for a single mapbox tile at the target zoom
    # simpler than the old stitch - one tile per call, fast to cache
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
    for _ in range(passes):
        heights = median_filter(heights, size=5)
    return heights

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
    try:
        lat, lon, zoom = float(lat), float(lon), int(zoom)
        x, y = lat_lon_to_tile(lat, lon, zoom)

        # fetch the 4 tiles we need in parallel: center, east, south, se corner
        # the neighbor tiles are cached so follow-up calls reuse them
        futs = {
            "c":  tile_pool.submit(fetch_terrain_grid, zoom, x,     y),
            "e":  tile_pool.submit(fetch_terrain_grid, zoom, x + 1, y),
            "s":  tile_pool.submit(fetch_terrain_grid, zoom, x,     y + 1),
            "se": tile_pool.submit(fetch_terrain_grid, zoom, x + 1, y + 1),
        }
        hc = futs["c"].result()
        he = futs["e"].result()
        hs = futs["s"].result()
        hse = futs["se"].result()

        # stitch a (h+1) x (w+1) grid so shared edges come from the SAME source pixels
        # this is the fix for tile seams - both neighbors use the same edge data
        h, w = hc.shape
        overlap = np.empty((h + 1, w + 1), dtype=np.float32)
        overlap[:h, :w] = hc
        overlap[:h, w]  = he[:h, 0]
        overlap[h, :w]  = hs[0, :w]
        overlap[h, w]   = hse[0, 0]

        # smooth only the interior - edges must stay pristine for neighbor alignment
        interior = smooth_heights(overlap[1:-1, 1:-1], passes=3)
        overlap[1:-1, 1:-1] = interior

        # resize to TERRAIN_RES + 1 so the shared edge row/col survive
        out_size = TERRAIN_RES + 1
        h_img = Image.fromarray(overlap)
        h_img = h_img.resize((out_size, out_size), Image.BILINEAR)
        heights = np.asarray(h_img)

        result = {
            "heights": heights.tolist(),
            "size": out_size,
        }
        del h_img, overlap, hc, he, hs, hse, interior
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
        # tobytes + list is ~3x faster than a generator comp
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
        n = 2 ** zoom
        tile_deg = 360 / n

        # 3x3 grid of tilequeries covers a zoom-12 tile well
        jobs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                qlat = lat + dy * tile_deg / 3
                qlon = lon + dx * tile_deg / 3
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
    except Exception as e:
        print(f"[pois] error: {e}")
        return jsonify({"pois": []}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)