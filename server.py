import os
import base64
import math
import io

from flask import Flask, jsonify
from PIL import Image
import numpy as np
import requests
from scipy.ndimage import median_filter, sobel

app = Flask(__name__)

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN")
SATELLITE_RES = 1024
TERRAIN_RES = 128
NORMAL_STRENGTH = 3.0  # higher = more dramatic normals

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

def fetch_and_stitch_satellite(terrain_zoom, terrain_x, terrain_y):
    zoom_offset = int(math.log2(SATELLITE_RES // 256))
    grid = 2 ** zoom_offset
    sat_zoom = terrain_zoom + zoom_offset
    x = terrain_x * grid
    y = terrain_y * grid
    stitched = Image.new("RGB", (grid * 256, grid * 256))
    for row in range(grid):
        for col in range(grid):
            tile = fetch_tile("mapbox.satellite", sat_zoom, x + col, y + row)
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
    for row in range(grid):
        for col in range(grid):
            tile = fetch_tile("mapbox.terrain-rgb", sat_zoom, tx + col, ty + row)
            tile = tile.resize((tile_size, tile_size))
            h = decode_terrain_rgb(tile)
            stitched[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size] = h
    return stitched

def smooth_heights(heights, passes=3):
    for _ in range(passes):
        heights = median_filter(heights, size=5)
    return heights

def generate_normal_map(heights, size):
    # sobel gradients give slope in x/y directions
    # then convert slope to a normal vector in tangent space
    gx = sobel(heights, axis=1) * NORMAL_STRENGTH
    gy = sobel(heights, axis=0) * NORMAL_STRENGTH

    # build normal vectors and normalize
    nz = np.ones_like(heights)
    length = np.sqrt(gx*gx + gy*gy + nz*nz)
    nx = -gx / length
    ny = -gy / length
    nz = nz / length

    # pack into rgb 0-255 (standard tangent-space normal map format)
    rgb = np.stack([
        ((nx * 0.5) + 0.5) * 255,
        ((ny * 0.5) + 0.5) * 255,
        ((nz * 0.5) + 0.5) * 255,
    ], axis=-1).astype(np.uint8)

    img = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def detect_water_mask(sat_img):
    # water is where blue dominates over red/green significantly
    # simple threshold but works great for rivers/lakes/ocean
    arr = np.array(sat_img.convert("RGB"), dtype=np.int16)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    # blue should be higher than red and green, and overall darker/cooler
    is_water = (B > R + 5) & (B > G - 5) & (R < 130) & (G < 150)
    mask = (is_water.astype(np.uint8) * 255)
    # downsample to a low-res coverage grid (how much of this region is water)
    img = Image.fromarray(mask).resize((32, 32), Image.BILINEAR)
    return np.array(img).flatten().tolist()

@app.route('/terrain/<lat>/<lon>/<zoom>')
def get_terrain(lat, lon, zoom):
    lat, lon, zoom = float(lat), float(lon), int(zoom)
    x, y = lat_lon_to_tile(lat, lon, zoom)
    heights = fetch_and_stitch_terrain(zoom, x, y)
    heights = smooth_heights(heights, passes=3)

    # generate normal map at satellite resolution for best visual match
    normal_b64 = generate_normal_map(heights, SATELLITE_RES)

    # resize heights to target terrain res for mesh use
    h_img = Image.fromarray(heights)
    h_img = h_img.resize((TERRAIN_RES, TERRAIN_RES), Image.BILINEAR)
    heights = np.array(h_img)

    return jsonify({
        "heights": heights.tolist(),
        "size": TERRAIN_RES,
        "min_height": float(heights.min()),
        "max_height": float(heights.max()),
        "normal_map_b64": normal_b64,
    })

@app.route('/satellite/<lat>/<lon>/<zoom>')
def get_satellite(lat, lon, zoom):
    lat, lon, zoom = float(lat), float(lon), int(zoom)
    x, y = lat_lon_to_tile(lat, lon, zoom)
    img = fetch_and_stitch_satellite(zoom, x, y)
    img = img.resize((SATELLITE_RES, SATELLITE_RES))

    # detect water coverage before converting to rgba
    water_mask = detect_water_mask(img)

    img = img.convert("RGBA")
    flat = [v for px in img.getdata() for v in px]
    return jsonify({
        "pixels": flat,
        "size": SATELLITE_RES,
        "water_mask": water_mask,
        "water_mask_size": 32,
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)