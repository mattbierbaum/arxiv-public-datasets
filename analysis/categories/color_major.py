import json
import colorsys
import binascii
import numpy as np
import random

catmap = json.load(open('./category-primary-map.json'))
catsecondaries = json.load(open('./category-secondaries.json'))
color_major = json.load(open('./color_major_good.json'))

def hex2rgb(h):
    return np.array([int(h[i+1:i+3], 16)/255 for i in (0, 2, 4)])

def rgb2hls(c):
    return np.array(colorsys.rgb_to_hls(*list(c)))

def hls2hex(c):
    rgb = np.array(colorsys.hls_to_rgb(*c))
    rgb = ((255*rgb).astype('uint8')).tobytes()
    return '#' + (binascii.b2a_hex(rgb)).decode('ascii')

def create_color_minor(variation=(0.07, 0.25, 0.07)):
    out = {}

    for primary, secondaries in catsecondaries.items():
        rgb = hex2rgb(color_major[primary])
        hls = rgb2hls(rgb)

        for secondary in secondaries:
            newl = np.clip(hls + variation*(np.random.rand(3) - 0.5), 0, 1)
            out[secondary] = hls2hex(newl)

    return out
