import os
import math
import io

from flask import Flask, jsonify
from PIL import Image
import numpy as np
import requests
from scipy.ndimage import median_filter, sobel, gaussian_filter
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN")
SATELLITE_RES = 1024
TERRAIN_RES = 128
NORMAL_STRENGTH = 1.5
NORMAL_SMOOTH_SIGMA = 2.0  # higher = softer pbr lighting, kills urban noise

# shared thread pool so mapbox tile fetches run in parallel
# 16 workers handles 4x4 satellite grid + 2x2 terrain grid comfortably
tile_pool = ThreadPoolExecutor(max_workers=16)

# shared session reuses tcp connections -> way faster than per-request requests.get
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
    # fetch all tiles at once instead of one-by-one, huge speedup
    jobs = []
    for row in range(grid):
        for col in range(grid):
            jobs.append((row, col, tile_pool.submit(fetch_tile, tileset, zoom, base_x + col, base_y + row)))
    return [(row, col, f.result()) for (row, col, f) in jobs]

def fetch_and_stitch_satellite(terrain_zoom, terrain_x, terrain_y):
    # higher zoom gives real extra detail for the same area
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
    # median filter kills building/noise spikes without flattening real hills
    for _ in range(passes):
        heights = median_filter(heights, size=5)
    return heights

def generate_normal_map(heights, size):
    # sobel gradients = slope in x and y
    gx = sobel(heights, axis=1) * NORMAL_STRENGTH
    gy = sobel(heights, axis=0) * NORMAL_STRENGTH

    # build unit normal vectors from slopes
    nz = np.ones_like(heights)
    length = np.sqrt(gx*gx + gy*gy + nz*nz)
    nx = -gx / length
    ny = -gy / length
    nz = nz / length

    # pack into rgba 0-255, tangent space convention
    rgba = np.stack([
        ((nx * 0.5) + 0.5) * 255,
        ((ny * 0.5) + 0.5) * 255,
        ((nz * 0.5) + 0.5) * 255,
        np.full_like(nz, 255),
    ], axis=-1).astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGBA").resize((size, size), Image.BILINEAR)
    # tobytes + list is ~3x faster than nested list comp
    return list(img.tobytes())

@app.route('/terrain/<lat>/<lon>/<zoom>')
def get_terrain(lat, lon, zoom):
    lat, lon, zoom = float(lat), float(lon), int(zoom)
    x, y = lat_lon_to_tile(lat, lon, zoom)
    heights = fetch_and_stitch_terrain(zoom, x, y)
    heights = smooth_heights(heights, passes=3)

    # extra gaussian blur just for normal generation, keeps mesh sharp
    # but makes urban areas lit gently instead of splotchy
    heights_for_normals = gaussian_filter(heights, sigma=NORMAL_SMOOTH_SIGMA)
    normal_pixels = generate_normal_map(heights_for_normals, SATELLITE_RES)

    # downsample heights for mesh vertices
    h_img = Image.fromarray(heights)
    h_img = h_img.resize((TERRAIN_RES, TERRAIN_RES), Image.BILINEAR)
    heights = np.array(h_img)

    return jsonify({
        "heights": heights.tolist(),
        "size": TERRAIN_RES,
        "min_height": float(heights.min()),
        "max_height": float(heights.max()),
        "normal_pixels": normal_pixels,
        "normal_size": SATELLITE_RES,
    })

@app.route('/satellite/<lat>/<lon>/<zoom>')
def get_satellite(lat, lon, zoom):
    lat, lon, zoom = float(lat), float(lon), int(zoom)
    x, y = lat_lon_to_tile(lat, lon, zoom)
    img = fetch_and_stitch_satellite(zoom, x, y)
    img = img.resize((SATELLITE_RES, SATELLITE_RES)).convert("RGBA")
    # tobytes + list is ~3x faster than flattening via generator
    flat = list(img.tobytes())
    return jsonify({"pixels": flat, "size": SATELLITE_RES})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)