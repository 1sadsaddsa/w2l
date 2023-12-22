import threading
from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from flask import Flask, request, jsonify
import time
from tqdm import tqdm
from glob import glob
import torch, face_detection

from face_detection.detection.DBFace.dbface_detector import DBFace, DBFaceDetector
from models import Wav2Lip
import platform

global args
# 定义为flask
mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))
model = Wav2Lip()
checkpoint = torch.load("checkpoints/wav2lip.pth")
s = checkpoint["state_dict"]
new_s = {}
for k, v in s.items():
    new_s[k.replace('module.', '')] = v
model.load_state_dict(new_s)
model = model.to(device)
model = model.eval()

if not os.path.isfile("face_detection/detection/DBFace/dbface.pth")):
    raise FileNotFoundError(f"Model weight file not found at {"face_detection/detection/DBFace/dbface.pth"}")
else:
    model_weights = torch.load("face_detection/detection/DBFace/dbface.pth", map_location=torch.device('cpu'))  # 其他人脸检测模型,默认使用GPU

DBface = DBFace()
DBface.load_state_dict(model_weights)
DBface.to('cuda')
DBface.eval()
face_detector = DBFaceDetector(DBface)

def initialize_args(arg_list=None):
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--checkpoint_path', type=str,
                        help='Name of saved checkpoint to load weights from', required=True)

    parser.add_argument('--face', type=str,
                        help='Filepath of video/image that contains faces to use', required=True)
    parser.add_argument('--audio', type=str,
                        help='Filepath of video/audio file to use as raw audio source', required=True)
    parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                        default='results/result_voice.mp4')

    parser.add_argument('--static', type=bool,
                        help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)

    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=16)
    parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                             'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                             'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                             'Use if you get a flipped result, despite feeding a normal looking video')

    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    if arg_list is not None:
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args()
    print(args)
    return args

# 在视频中得到更平滑的矩形框范围
def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, args):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)
    batch_size = args.face_det_batch_size

    while 1:
        predictions = []  # 存储预测的人脸坐标
        try:
            for i in tqdm(range(0, len(images), batch_size)):   # 每批使用batch_size张图片
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            # 123
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        # image.shape[0] -> y, image.shape[1] -> x
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)  # T 快速移动的物体需要较小的T
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]  # 存储了裁剪后的人脸图像区域及其对应坐标

    del detector
    return results

# frames是所有视频帧或图像帧， mels是所有梅尔频谱图
def datagen(frames, mels, args):
    # 图像，梅尔频谱图，原始帧，人脸坐标数据
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    s = time.time()
    face_det_results = face_detect(frames, args)  # BGR2RGB for CNN face detection，每一帧
    e = time.time()
    print(f'图像预测时间：{e-s}')
    for i, m in enumerate(mels):
        idx = i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
        face = cv2.resize(face, (args.img_size, args.img_size))  # 保证图像尺寸的一致性

        img_batch.append(face)  # 裁剪下的脸部图像
        mel_batch.append(m)  # 梅尔频谱图
        frame_batch.append(frame_to_save)  # 视频帧
        coords_batch.append(coords)  # 人脸坐标数据

        if len(img_batch) >= args.wav2lip_batch_size:  # 控制批次，按批次处理图像
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()  # 创建遮罩图像
            img_masked[:, args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.  # 在axis=3上合并原图像和遮罩图像，并且对图片颜色归一化
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])  # mel_batch.shape[1]表示频率，mel_batch.shape[2]表示时间

        yield img_batch, mel_batch, frame_batch, coords_batch

def main(args):
    args.img_size = 96  # 图片大小

    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    video_stream = cv2.VideoCapture(args.face)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    full_frames = []
    mel_chunks = []
    audio_thread = threading.Thread(target=audio_processing, args=(args, fps, mel_chunks))
    image_thread = threading.Thread(target=video_processing, args=(args, video_stream, full_frames))
    audio_thread.start()
    image_thread.start()
    audio_thread.join()
    image_thread.join()

    full_frames = full_frames[:len(mel_chunks)]
    frame_h, frame_w = full_frames[0].shape[:-1]
    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks, args)
    out = cv2.VideoWriter('temp/result.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)  #将(batch_size, height, width, channels)转化为(batch_size, channels, height, width)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        # 15s,7s
        start_time = time.time()
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        end_time = time.time()
        print(f"wav2lip运行时间：{end_time - start_time}s")
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.  #再将维度变回来

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c

            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p  # 将图片重新贴回去
            cv2.rectangle(f, [x1, y1, x2, y2], color=(0, 0, 255), thickness=1)
            out.write(f)

    out.release()

    command = ['ffmpeg', '-y', '-i', 'temp/result.avi', '-i', args.audio, '-strict', '-2', '-q:v', '1', args.outfile]
    subprocess.call(command)

def audio_processing(args, fps, mel_chunks):
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = ['ffmpeg', '-y', '-i', args.audio, '-strict', '-2', 'temp/temp.wav']
        subprocess.call(command)
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)  # 生成梅尔频谱图
    print(mel.shape)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')  # TTS会产生nan错误

    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))
    return mel_chunks

def video_processing(args, video_stream, full_frames):
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if args.resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

        if args.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90°

        y1, y2, x1, x2 = args.crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]

        full_frames.append(frame)
    return full_frames
# def concreate(args):


if __name__ == '__main__':
    main(args)

