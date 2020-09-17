import multiprocessing as mp
from tqdm import tqdm
from itertools import repeat
import ffmpeg
import csv
import cv2
import numpy as np
import argparse
import itertools


DISPLAY_MODE = {
    'WHITE': [[0, 0, 0], [255, 255, 255]],
    'BLACK': [[255, 255, 255], [0, 0, 0]]
}

parser = argparse.ArgumentParser(
    description='This program converts text to videos.')

parser.add_argument('input', type=str, help='input text file')
parser.add_argument('output', type=str, help='output video')
parser.add_argument('-W', '--width', type=int, help='output video width')
parser.add_argument('-H', '--height', type=int, help='output video height')
parser.add_argument('-F', '--fps', type=int, help='FPS of output video')
parser.add_argument('-A', '--audio', type=str, help='input audio file')
parser.add_argument('-f', '--font', type=str, help='font path', required=True)


def getDisplayMode(mode):
    if mode not in DISPLAY_MODE:
        return DISPLAY_MODE['WHITE']
    else:
        return DISPLAY_MODE[mode]


def textToFrame(data):
    ind = data[0]
    text = data[1]
    opts = data[2]
    # print(os.getpid())

    monospace = cv2.freetype.createFreeType2()
    monospace.loadFontData(
        fontFileName=opts['font'], id=0)
    fontHeight = opts['output_height'] // opts['height']
    textSize, baseline = monospace.getTextSize(
        text='a'*opts['width'], fontHeight=fontHeight, thickness=-1)
    region = (textSize[0], fontHeight*opts['height'])

    mode, rendertext = text.split('\n', 1)
    colors = getDisplayMode(mode)

    frame = np.full((*reversed(region), 3), colors[1], dtype=np.uint8)

    for i, line in enumerate(text.split('\n')):
        monospace.putText(img=frame, text=line, org=(0, i*fontHeight),
                          fontHeight=fontHeight, color=colors[0], thickness=-1, line_type=cv2.LINE_4, bottomLeftOrigin=True)

    frame = cv2.resize(frame, dsize=(
        opts['output_width'], opts['output_height']))
    # cv2.imwrite('test/{:04}.png'.format(ind), frame)
    return (ind, frame)


def main():
    with open(args.input) as f:
        fp = next(itertools.islice(csv.DictReader(
            [f.readline(), f.readline()]), 1))
        print(fp)
        f.readline()
        HEIGHT = int(fp['origheight'])
        if args.height:
            HEIGHT = args.height
        WIDTH = int(fp['origwidth'])
        if args.width:
            WIDTH = args.width

        FPS = float(fp['FPS'])
        if args.fps:
            FPS = args.fps

        opts = {}

        texts = []
        cnt = 0
        tmpstr = ''
        frame_height = int(fp['height']) + 1
        opts['height'] = int(fp['height'])
        opts['width'] = int(fp['width'])
        opts['output_height'] = HEIGHT
        opts['output_width'] = WIDTH
        opts['font'] = args.font

        print(opts)
        for text in f:
            tmpstr += text
            cnt += 1
            if cnt == frame_height:
                cnt = 0
                texts.append(tmpstr)
                tmpstr = ''

        output = args.output
        if args.audio:
            output = args.output + '_tmp.mp4'
        print(output)

        frames_unord = []
        with mp.Pool(mp.cpu_count()) as pool:
            with tqdm(total=len(texts)) as t:
                for res in pool.imap_unordered(
                        textToFrame, zip(range(len(texts)), texts, repeat(opts))):
                    t.update()
                    frames_unord.append(res)

        frames = sorted(frames_unord)
        frames = list(map(lambda p: p[1], frames))

        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(
            *'mp4v'), FPS, (WIDTH, HEIGHT))

        with tqdm(total=len(frames)) as t:
            for frame in frames:
                out.write(frame)
                t.update()

        out.release()

        if args.audio:
            video = ffmpeg.input(output)
            audio = ffmpeg.input(args.audio)
            ffmpeg.output(audio.audio, video.video,
                          args.output, shortest=None).run()


if __name__ == "__main__":
    args = parser.parse_args()
    main()
