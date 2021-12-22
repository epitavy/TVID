import sys, os, pathlib

import time
import argparse
import cv2 as cv
import numpy as np


# Monkey patch to allow raw concatenation
pathlib.Path.__add__ = lambda self, rhs: pathlib.Path(str(self) + rhs)

FRAME_RATE = None
DEFAULT_IPS = 25

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
    u = cv.resize(u, None, None, 2, 2, cv.INTER_LINEAR)
    v = cv.resize(v, None, None, 2, 2, cv.INTER_LINEAR)

    return u, v

#@profile
def convert(path, deinterlace):
    with open(path, 'rb') as f:
        buffer = f.read()

        width, height = int(buffer[3:6].decode('utf-8')), int(buffer[7:10].decode('utf-8'))

        comment_idx = buffer.find(b'#')
        if comment_idx != -1 and comment_idx < 30: # Worst check ever
            comment = buffer[comment_idx:comment_idx+27].decode('utf-8')
            ips_end = comment.find(' rff')
            ips = int(comment[5:ips_end])
            repeat_first_field = bool(int(comment[ips_end+5]))
            top_field_first = bool(int(comment[ips_end+11]))
            progressive_frame = bool(int(comment[ips_end+18]))
        else:
            ips = -1
            repeat_first_field = False
            top_field_first = True
            progressive_frame = True

        image = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv.IMREAD_GRAYSCALE)

    if deinterlace: # Overwrite image flags
        progressive_frame = False

    chroma_height = height // 3
    chroma_width = width // 2

    image = image.astype('int16')

    if progressive_frame:
        image_y = image[:-chroma_height] - 16
        image_u = image[-chroma_height:, :chroma_width] - 128
        image_v = image[-chroma_height:, chroma_width:] - 128
        u, v = decompress420(image_u, image_v)
        yuv_recomposed = np.dstack([image_y, u, v])
        rgb = np.matmul(yuv_recomposed, YUV2BGR.T)

        return rgb.clip(0, 255).astype(np.uint8), None, ips



    yuv_splited_top = image[0::2]
    yuv_splited_bottom = image[1::2]

    top_y = yuv_splited_top[:-chroma_height//2] - 16
    top_u = yuv_splited_top[-chroma_height//2:, :chroma_width] - 128
    top_v = yuv_splited_top[-chroma_height//2:, chroma_width:] - 128

    bottom_y = yuv_splited_bottom[:-chroma_height//2] - 16
    bottom_u = yuv_splited_bottom[-chroma_height//2:, :chroma_width] - 128
    bottom_v = yuv_splited_bottom[-chroma_height//2:, chroma_width:] - 128


    top_u, top_v = decompress420(top_u, top_v)
    bottom_u, bottom_v = decompress420(bottom_u, bottom_v)

    (top_y, top_u, top_v), (bottom_y, bottom_u, bottom_v) = \
        bob_deinterlace((top_y, top_u, top_v), (bottom_y, bottom_u, bottom_v))

    yuv_recomposed_top = np.dstack([top_y, top_u, top_v])
    yuv_recomposed_bottom = np.dstack([bottom_y, bottom_u, bottom_v])

    rgb_top = np.matmul(yuv_recomposed_top, YUV2BGR.T)
    rgb_bottom = np.matmul(yuv_recomposed_bottom, YUV2BGR.T)

    rgb_top = rgb_top.clip(0, 255).astype(np.uint8)
    rgb_bottom = rgb_bottom.clip(0, 255).astype(np.uint8)

    if top_field_first:
        return rgb_top, rgb_bottom, ips * 2
    else:
        return rgb_bottom, rgb_top, ips * 2


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Convert YUV interlaced image to RGB ppm")
    parser.add_argument("input", help="The input file, in pgm format")
    parser.add_argument("-o", "--outpath", help="The output path to write the converted image")
    parser.add_argument("--display", help="Display image instead of converting to ppm", action="store_true")
    parser.add_argument("--deinterlace", help="Force deinterlacing", action="store_true")
    parser.add_argument("--frame_rate", help="Frame rate used when displaying output on screen", type=int)
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


    movie = []
    rgb_top, rgb_bottom, _ = convert(image_list[0], args.deinterlace)
    movie.append(rgb_top)
    if rgb_bottom is not None:
        movie.append(rgb_bottom)
    frame_displayed = 0
    image_loaded = 1
    timer = 0
    frame_rate = 0
    while frame_displayed < len(movie):
        if image_loaded < len(image_list):
            image = image_list[image_loaded]
            rgb_top, rgb_bottom, ips = convert(image, args.deinterlace)
            if args.frame_rate:
                WAIT_TIME = 1 / args.frame_rate
            elif ips > 0:
                WAIT_TIME = 1 / ips
            elif WAIT_TIME is None:
                WAIT_TIME = 1 / DEFAULT_IPS

            if ips == -2: # Double frame rate, frame is interlaced
                WAIT_TIME /= 2

            image_loaded += 1
            movie.append(rgb_top)
            if rgb_bottom is not None:
                movie.append(rgb_bottom)
            print(f"Loading image {image_loaded}/{len(image_list)}", end='\r')
        else:
            print("All image loaded\t\t ", end='\r')

        if args.display:
            if time.perf_counter() - timer >= WAIT_TIME:
                frame_rate = 0.8 * frame_rate + 0.2 / (time.perf_counter() - timer) # Weighted average
                print(f"\t\t\t\t\t\tReal frame rate: {frame_rate:.2f}", end='\r')
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
