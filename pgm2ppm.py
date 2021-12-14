import sys, os, pathlib

import time
import argparse
import cv2 as cv
import numpy as np


# Monkey patch to allow raw concatenation
pathlib.Path.__add__ = lambda self, rhs: pathlib.Path(str(self) + rhs)

YUV2BGR = np.array([
    [1, 2.03211, 0],
    [1, -0.39465, -0.58060],
    [1, 0, 1.13983]
])

def write_rgb_ppm(img, outpath):
    with open(outpath, 'w') as f:

        f.write(f"P3\n{len(img[0])} {len(img)}\n255\n")

        for line in img:
            for pix in line:
                f.write(f"{pix[0]} {pix[1]} {pix[2]}\n")

#@profile
def bob_deinterlace(top, bottom):
    ty, tu, tv = top
    by, bu, bv = bottom

    ty = cv.resize(ty, None, None, 1, 2, cv.INTER_LINEAR)
    tu = cv.resize(tu, None, None, 1, 2, cv.INTER_LINEAR)
    tv = cv.resize(tv, None, None, 1, 2, cv.INTER_LINEAR)

    by = cv.resize(by, None, None, 1, 2, cv.INTER_LINEAR)
    bu = cv.resize(bu, None, None, 1, 2, cv.INTER_LINEAR)
    bv = cv.resize(bv, None, None, 1, 2, cv.INTER_LINEAR)

    return (ty, tu, tv), (by, bu, bv)

#@profile
def decompress420(u, v):
    u = cv.resize(u, None, None, 2, 2, cv.INTER_NEAREST)
    v = cv.resize(v, None, None, 2, 2, cv.INTER_NEAREST)

    return u, v

#@profile
def convert(path):
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    height, width = image.shape

    chroma_height = height // 3
    chroma_width = width // 2

    image = image.astype('int16')

    yuv_splited_top = image[0::2]
    yuv_splited_bottom = image[1::2]

    yuv_recomposed_top = np.zeros((height - chroma_height, width, 3))
    yuv_recomposed_bottom = np.zeros((height - chroma_height, width, 3))

    top_y = yuv_splited_top[:-chroma_height//2] - 16
    top_u = yuv_splited_top[-chroma_height//2:, :chroma_width] - 128
    top_v = yuv_splited_top[-chroma_height//2:, chroma_width:] - 128

    bottom_y = yuv_splited_bottom[:-chroma_height//2] - 16
    bottom_u = yuv_splited_bottom[-chroma_height//2:, :chroma_width] - 128
    bottom_v = yuv_splited_bottom[-chroma_height//2:, chroma_width:] - 128

    (top_y, top_u, top_v), (bottom_y, bottom_u, bottom_v) = \
        bob_deinterlace((top_y, top_u, top_v), (bottom_y, bottom_u, bottom_v))

    top_u, top_v = decompress420(top_u, top_v)
    bottom_u, bottom_v = decompress420(bottom_u, bottom_v)

    yuv_recomposed_top = np.dstack([top_y, top_u, top_v])
    yuv_recomposed_bottom = np.dstack([bottom_y, bottom_u, bottom_v])

    rgb_top = np.matmul(yuv_recomposed_top, YUV2BGR.T)
    rgb_bottom = np.matmul(yuv_recomposed_bottom, YUV2BGR.T)

    rgb_top = rgb_top.clip(0, 255).astype(np.uint8)
    rgb_bottom = rgb_bottom.clip(0, 255).astype(np.uint8)

    return rgb_top, rgb_bottom


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Convert YUV interlaced image to RGB ppm")
    parser.add_argument("input", help="The input file, in pgm format")
    parser.add_argument("-o", "--outpath", help="The output path to write the converted image")
    parser.add_argument("--display", help="Display image instead of converting to ppm", action="store_true")
    parser.add_argument("--frame_rate", help="Frame rate used when displaying output on screen", type=int, default=25)
    args = parser.parse_args()

    input_is_dir = os.path.isdir(args.input)

    if not input_is_dir and args.input[-4:] != ".pgm":
        parser.error("The input file should be a pgm image file or a directory containing pgm image files")

    if input_is_dir:
        image_list = os.listdir(args.input)
        image_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        image_list = [pathlib.Path(args.input, file) for file in image_list if file.endswith('.pgm')]
    else:
        image_list = [args.input]

    wait_time = 1 / args.frame_rate

    movie = []
    rgb_top, rgb_bottom = convert(image_list[0])
    movie.append(rgb_top)
    movie.append(rgb_bottom)
    frame_displayed = 0
    image_loaded = 1
    timer = 0
    while frame_displayed < len(movie):
        if image_loaded < len(image_list):
            image = image_list[image_loaded]
            rgb_top, rgb_bottom = convert(image)
            image_loaded += 1
            movie.append(rgb_top)
            movie.append(rgb_bottom)
            print(f"Loading image {image_loaded}/{len(image_list)}", end='\r')
        else:
            print("All image loaded\t\t ", end='\r')

        if args.display:
            if time.perf_counter() - timer >= wait_time:
                print(f"\t\t\t\t\t\tReal frame rate: {1/(time.perf_counter() - timer):.2f}", end='\r')
                timer = time.perf_counter()
                cv.imshow("ppm converted", movie[frame_displayed])
                cv.waitKey(1)
                frame_displayed += 1
        else:
            if args.outpath:
                outpath = args.outpath
                if outpath[-4:] == '.ppm':
                    outpath = outpath[:-4]
            else:
                outpath = args.input[:-4]

            if input_is_dir: # We suppose outpath is a directory too
                os.makedirs(outpath, exist_ok=True)
                outpath = pathlib.Path(outpath, image.stem)

            write_rgb_ppm(rgb_top, outpath + '-i.ppm')
