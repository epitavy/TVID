import sys, os, pathlib

import time
import argparse
import cv2 as cv
import numpy as np
from numpy.core.fromnumeric import mean


# Monkey patch to allow raw concatenation
pathlib.Path.__add__ = lambda self, rhs: pathlib.Path(str(self) + rhs)

WAIT_TIME = None
DEFAULT_IPS = 25

YUV2BGR = np.array([
    [1, 2.03211, 0],
    [1, -0.39465, -0.58060],
    [1, 0, 1.13983]
])

def bob_deinterlace(top, bottom):
    ty, tu, tv = top
    by, bu, bv = bottom

    ty = cv.resize(ty, (ty.shape[1], ty.shape[0]*2), None, 0, 0, cv.INTER_LINEAR)
    tu = cv.resize(tu, (tu.shape[1], tu.shape[0]*2), None, 0, 0, cv.INTER_LINEAR)
    tv = cv.resize(tv, (tv.shape[1], tv.shape[0]*2), None, 0, 0, cv.INTER_LINEAR)

    by = cv.resize(by, (by.shape[1], by.shape[0]*2), None, 0, 0, cv.INTER_LINEAR)
    bu = cv.resize(bu, (bu.shape[1], bu.shape[0]*2), None, 0, 0, cv.INTER_LINEAR)
    bv = cv.resize(bv, (bv.shape[1], bv.shape[0]*2), None, 0, 0, cv.INTER_LINEAR)

    return (ty, tu, tv), (by, bu, bv)

def decompress420(u, v):
    u = cv.resize(u, None, None, 2, 2, cv.INTER_LINEAR)
    v = cv.resize(v, None, None, 2, 2, cv.INTER_LINEAR)
    
    return u, v

def computeBlockDisplacment(x, y, block, target, win_size, block_size):
    # x and y are in block coordinates, so they should be multiplied
    # by the block size to get real coordinates
    whalf = win_size // 2

    xmins = [x * block_size + i for i in range(-whalf, whalf + 1)]
    xmaxs = [(x + 1) * block_size + i for i in range(-whalf, whalf + 1)]
    ymins = [y * block_size + i for i in range(-whalf, whalf + 1)]
    ymaxs = [(y + 1) * block_size + i for i in range(-whalf, whalf + 1)]

    # Get all blocks over the window
    blocksWin = np.ndarray((win_size, win_size, block_size, block_size))
    for i in range(win_size):
        for j in range(win_size):
            if xmins[i] < 0 or xmaxs[i] >= target.shape[0] or \
               ymins[j] < 0 or ymaxs[j] >= target.shape[1]:
                blocksWin[i,j] = np.nan # Will give NaN error and never be choosen
            else:
                blocksWin[i,j] = target[xmins[i]:xmaxs[i], ymins[j]:ymaxs[j]]

    diff = np.abs(blocksWin - block)
    diff = np.sum(diff.reshape(win_size, win_size, -1), axis=-1)
    if diff[whalf, whalf] == 0:
        return (0, 0)
    if np.all(np.isnan(diff)):
        return (0, 0)

    minXY = np.nanargmin(diff)
    minX, minY = (minXY // win_size, minXY % win_size)

    # Offset minX and minY according to the center of the window
    dx = minX - whalf
    dy = minY - whalf
    return dx, dy

def blockWiseMotionEstimation(frame1, frame2, window_size, block_size):
    height, width = frame1.shape[0], frame1.shape[1]
    mvectorsPixels = np.ndarray((height, width, 2), dtype=np.int16)


    # Compute block displacement for all blocks in the image
    for x, y in np.ndindex(((height + block_size - 1) // block_size,
                            (width + block_size - 1) // block_size)):
        # Block coordinates
        bxmin, bxmax = x * block_size, (x + 1) * block_size
        bymin, bymax = y * block_size, (y + 1) * block_size

        block = frame1[bxmin:bxmax, bymin:bymax]

        # Pad to correct the size of blocks on the border
        padH = bxmax - height
        padW = bymax - width
        padH = padH if padH > 0 else 0
        padW = padW if padW > 0 else 0

        block = np.pad(block, [(0, padH), (0, padW)], constant_values=0)
        mvectorsPixels[bxmin:bxmax,bymin:bymax] = computeBlockDisplacment(x, y, block, frame2, window_size, block_size)


    return mvectorsPixels

def get_vecs(frame, frame2):
    vecs = np.zeros((frame.shape[0] // 8, frame.shape[1] // 8, 2))
    for i in range(0, frame.shape[0], 8):
        #print(f"{i}/{frame.shape[0]}", end="\r")
        for j in range(0, frame.shape[1], 8):

            mini = np.Infinity
            minX = np.Infinity
            minY = np.Infinity

            for x in range(-8,9):

                if i + x < 0:
                    continue

                if i + x + 8 > frame.shape[0] - 1:
                    continue

                for y in range(-8,9):

                    if j + y < 0:
                        continue

                    if j + y + 8 > frame.shape[1] - 1:
                        continue
    
                    tmp = np.sum((frame2[i+x:i+x+8,j+y:j+y+8] - frame[i:i+8,j:j+8])**2)

                    if tmp < mini:
                        mini = tmp
                        minX = x
                        minY = y
            

            vecs[i // 8][j // 8] = [int(minX), int(minY)]
    return vecs

def spatial_deinterlace(top, top2, bottom, bottom2):
    ty, tu, tv = top
    ty2, tu2, tv2 = top2

    by, bu, bv = bottom
    by2, bu2, bv2 = bottom2

    #top_vecs = get_vecs(ty, ty2)
    #bottom_vecs = get_vecs(by, by2)

    top_vecs = blockWiseMotionEstimation(ty, ty2, 8, 8)
    bottom_vecs = blockWiseMotionEstimation(by, by2, 8, 8)

    res_ty = np.zeros((ty.shape[0] * 2, ty.shape[1]))
    res_tu = np.zeros((tu.shape[0] * 2, tu.shape[1]))
    res_tv = np.zeros((tv.shape[0] * 2, tv.shape[1]))

    res_by = np.zeros((by.shape[0] * 2, bu.shape[1]))
    res_bu = np.zeros((bu.shape[0] * 2, bu.shape[1]))
    res_bv = np.zeros((bv.shape[0] * 2, bv.shape[1]))

    mean = np.mean(np.dstack([top_vecs[:,:,0], bottom_vecs[:,:,0]]), axis=-1)
    threshold = mean > 5
    for i in range(ty.shape[0] // 8):
        for j in range(ty.shape[1] // 8):
            is_moving = threshold[i*8, j*8]

            top_block = (ty[i*8:(i+1)*8, j*8: (j+1)*8], tu[i*8:(i+1)*8, j*8: (j+1)*8], tv[i*8:(i+1)*8, j*8: (j+1)*8])
            bottom_block = (by[i*8:(i+1)*8, j*8: (j+1)*8], bu[i*8:(i+1)*8, j*8: (j+1)*8], bv[i*8:(i+1)*8, j*8: (j+1)*8])

            if (is_moving): 
                top_block, bottom_block = bob_deinterlace(top_block, bottom_block)

                res_ty[i*8*2:(i+1)*8*2, j*8: (j+1)*8] = top_block[0]
                res_tu[i*8*2:(i+1)*8*2, j*8: (j+1)*8] = top_block[1]
                res_tv[i*8*2:(i+1)*8*2, j*8: (j+1)*8] = top_block[2]

                res_by[i*8*2:(i+1)*8*2, j*8: (j+1)*8] = bottom_block[0]
                res_bu[i*8*2:(i+1)*8*2, j*8: (j+1)*8] = bottom_block[1]
                res_bv[i*8*2:(i+1)*8*2, j*8: (j+1)*8] = bottom_block[2]
            else:
                res_ty[i*8*2:(i+1)*8*2:2, j*8: (j+1)*8] = top_block[0]
                res_ty[i*8*2 + 1:(i+1)*8*2 + 1:2, j*8: (j+1)*8] = bottom_block[0]

                res_tu[i*8*2:(i+1)*8*2:2, j*8: (j+1)*8] = top_block[1]
                res_tu[i*8*2 + 1:(i+1)*8*2 + 1:2, j*8: (j+1)*8] = bottom_block[1]

                res_tv[i*8*2:(i+1)*8*2:2, j*8: (j+1)*8] = top_block[2]
                res_tv[i*8*2 + 1:(i+1)*8*2 + 1:2, j*8: (j+1)*8] = bottom_block[2]

                res_by[i*8*2:(i+1)*8*2:2, j*8: (j+1)*8] = top_block[0]
                res_by[i*8*2 + 1:(i+1)*8*2 + 1:2, j*8: (j+1)*8] = bottom_block[0]

                res_bu[i*8*2:(i+1)*8*2:2, j*8: (j+1)*8] = top_block[1]
                res_bu[i*8*2 + 1:(i+1)*8*2 + 1:2, j*8: (j+1)*8] = bottom_block[1]

                res_bv[i*8*2:(i+1)*8*2:2, j*8: (j+1)*8] = top_block[2]
                res_bv[i*8*2 + 1:(i+1)*8*2 + 1:2, j*8: (j+1)*8] = bottom_block[2]


    return (res_ty, res_tu, res_tv), (res_by, res_bu, res_bv)



#@profile
def convert(path, path2, deinterlace):
    with open(path, 'rb') as f:
        buffer = f.read()


        if buffer[3] == ord('#'):
            comment_end = buffer[3:].find(b'\n') + 3
            comment = buffer[3:comment_end].decode('utf-8')
            ips_end = comment.find(' rff')
            ips = int(comment[5:ips_end])
            repeat_first_field = bool(int(comment[ips_end+5]))
            top_field_first = bool(int(comment[ips_end+11]))
            progressive_frame = bool(int(comment[ips_end+18]))
            wh_line = buffer[comment_end+1:buffer[comment_end+1:].find(b'\n') + comment_end+1]
        else:
            ips = -1
            repeat_first_field = False
            top_field_first = True
            progressive_frame = True
            wh_line = buffer[3:buffer[3:].find(b'\n') + 3]


        wh_line = wh_line.split(b' ')
        width, height = int(wh_line[0].decode('utf-8')), int(wh_line[1].decode('utf-8'))
        image = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv.IMREAD_GRAYSCALE)

    if deinterlace == "force_bob" or deinterlace == "force_spatial": # Overwrite image flags
        progressive_frame = False
    elif deinterlace == "None":
        progressive_frame = True

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


    image2 = cv.imread(str(path2), cv.IMREAD_GRAYSCALE)

    yuv_splited_top2 = image2[0::2]
    yuv_splited_bottom2 = image2[1::2]

    top_y2 = yuv_splited_top2[:-chroma_height//2] - 16
    top_u2 = yuv_splited_top2[-chroma_height//2:, :chroma_width] - 128
    top_v2 = yuv_splited_top2[-chroma_height//2:, chroma_width:] - 128

    bottom_y2 = yuv_splited_bottom2[:-chroma_height//2] - 16
    bottom_u2 = yuv_splited_bottom2[-chroma_height//2:, :chroma_width] - 128
    bottom_v2 = yuv_splited_bottom2[-chroma_height//2:, chroma_width:] - 128


    top_u2, top_v2 = decompress420(top_u2, top_v2)
    bottom_u2, bottom_v2 = decompress420(bottom_u2, bottom_v2)

    if deinterlace == "spatial" or deinterlace == "force_spatial":
        (top_y, top_u, top_v), (bottom_y, bottom_u, bottom_v) = \
            spatial_deinterlace((top_y, top_u, top_v), (top_y2, top_u2, top_v2), (bottom_y, bottom_u, bottom_v), (bottom_y2, bottom_u2, bottom_v2))
    elif deinterlace == "bob" or deinterlace == "force_bob":
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

    parser = argparse.ArgumentParser(description="Convert YUV interlaced image to RGB ppm")
    parser.add_argument("input", help="The input file or input directory, in pgm format")
    parser.add_argument("-o", "--outpath", help="The output path to write the converted image")
    parser.add_argument("--display", help="Display image instead of converting to ppm", action="store_true")
    parser.add_argument("--deinterlace", help="Deinterlace with bob, spatial or nothing. You can also deinterlace all the frame with value force.", choices=["None", "bob", "spatial", "force_bob", "force_spatial"])
    parser.add_argument("--frame_rate", help="Frame rate used when displaying output on screen", type=int)
    args = parser.parse_args()

    input_is_dir = os.path.isdir(args.input)

    if not input_is_dir and args.input[-4:] != ".pgm":
        parser.error("The input file should be a pgm image file or a directory containing pgm image files")

    if input_is_dir:
        image_list = os.listdir(args.input)
        image_list = [file for file in image_list if file.endswith('.pgm')]
        image_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        image_list = [pathlib.Path(args.input, file) for file in image_list]
    else:
        image_list = [args.input]


    movie = []
    rgb_top, rgb_bottom, _ = convert(image_list[0], image_list[1], deinterlace=args.deinterlace)
    movie.append(rgb_top)
    if rgb_bottom is not None:
        movie.append(rgb_bottom)
    frame_displayed = 0
    image_loaded = 1
    timer = 0
    frame_rate = 0
    while frame_displayed < len(movie):
        if image_loaded < len(image_list) - 1:
            image = image_list[image_loaded]
            image2 = image_list[image_loaded + 1]
            rgb_top, rgb_bottom, ips = convert(image, image2, deinterlace=args.deinterlace)
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
            print("All image loaded        ", end='\r')

        if args.display:
            if time.perf_counter() - timer >= WAIT_TIME:
                frame_rate = 0.95 * frame_rate + 0.05 / (time.perf_counter() - timer) # Weighted average
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

            cv.imwrite(str(outpath + f'-{frame_displayed}.ppm'), movie[frame_displayed])
            frame_displayed += 1
