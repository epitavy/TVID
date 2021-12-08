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

    ty = np.repeat(ty, [2] * ty.shape[0], axis=0)
    tu = np.repeat(tu, [2] * tu.shape[0], axis=0)
    tv = np.repeat(tv, [2] * tv.shape[0], axis=0)

    by = np.repeat(by, [2] * by.shape[0], axis=0)
    bu = np.repeat(bu, [2] * bu.shape[0], axis=0)
    bv = np.repeat(bv, [2] * bv.shape[0], axis=0)

    return (ty, tu, tv), (by, bu, bv)

#@profile
def decompress420(u, v):
    u = np.repeat(u, [2] * u.shape[0], axis=0)
    u = np.repeat(u, [2] * u.shape[1], axis=1)

    v = np.repeat(v, [2] * v.shape[0], axis=0)
    v = np.repeat(v, [2] * v.shape[1], axis=1)

    return u, v

#@profile
def convert(path):
    with open(path, 'rb') as f:
        magic = f.readline()
        if magic != b'P5\n':
            raise RuntimeError("The input file is not a pgm image")

        width, height = list(map(lambda x: int(x), f.readline().split(b' ')))
        max_intensity = int(f.readline()[:-1])
        pixels = np.fromfile(f, dtype=np.uint8)

    chroma_height = height // 3
    chroma_width = width // 2

    yuv_splited_top = np.ndarray((height // 2, width))
    yuv_splited_bottom = np.ndarray((height // 2, width))


    for idx in range(0, len(pixels), width):
        read_line = idx // width
        if read_line % 2 == 0:
            yuv_splited_top[read_line//2] = pixels[idx:idx+width]
        else:
            yuv_splited_bottom[read_line//2] = pixels[idx:idx+width]

    yuv_splited_top *= 255 / max_intensity
    yuv_splited_bottom *= 255 / max_intensity

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

    rgb_top = np.dot(yuv_recomposed_top, YUV2BGR.T)
    rgb_bottom = np.dot(yuv_recomposed_bottom, YUV2BGR.T)

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
    frame_displayed = 0
    timer = 0
    for i, image in enumerate(image_list):
        rgb_top, rgb_bottom = convert(image)
        movie.append(rgb_top)
        movie.append(rgb_bottom)
        print(f"Loading image {i}/{len(image_list)}", end='\r')

        if args.display:
            if time.perf_counter() - timer >= wait_time:
                cv.imshow("ppm converted", movie[frame_displayed])
                frame_displayed += 1
                timer = time.perf_counter()
                cv.waitKey(1)
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
