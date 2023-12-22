import os
import sys
import glob
import time

import torch

# from inference import main, initialize_args
from myinterface import main, initialize_args

def test_wav2lip():
    start_time = time.time()
    args_list = [
        '--checkpoint_path', 'checkpoints/wav2lip.pth',
        '--face', 'video/face/20231114030207.mp4',
        '--audio', 'video/audio/gina_scn004_039.ogg',
        '--outfile', 'results/output.mp4',
        # '--resize_factor', '5',
        # '--static', 'True',
    ]

    main(initialize_args (args_list))
    end_time = time.time()
    print(f"总运行时间：{end_time - start_time}s")

# def test_wav2lip():
#     audio_folder = 'video/audio'
#     face_folder = 'video/face'
#     checkpoint_path = 'checkpoints/wav2lip.pth'
#     output_folder = 'results'
#
#     # 确保音频文件夹和面部文件夹存在
#     if not os.path.exists(audio_folder) or not os.path.exists(face_folder):
#         raise FileNotFoundError("Audio or face folder not found.")
#
#     # 获取音频文件夹中的所有文件
#     audio_files = glob.glob(os.path.join(audio_folder, '*'))
#     face_file = os.path.join(face_folder, 'face.jpg')  # 假设face文件夹中只有一个文件
#
#     # 遍历音频文件并处理
#     for audio_file in audio_files:
#         outfile = os.path.join(output_folder, os.path.basename(audio_file).split('.')[0] + '_output.mp4')
#         args_list = [
#             '--checkpoint_path', checkpoint_path,
#             '--face', face_file,
#             '--audio', audio_file,
#             '--outfile', outfile,
#         ]
#         main(initialize_args(args_list))

if __name__ == '__main__':
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # x = torch.rand(5, 5).cuda()
    # print(x)

    test_wav2lip()


# 默认情况下图片和音频有下标并且一样
# def test_wav2lip(face_file, audio_file):
#     args_list = [
#         '--checkpoint_path', 'checkpoints/wav2lip.pth',
#         '--face', face_file,
#         '--audio', audio_file,
#         '--outfile', f'results/output_{os.path.basename(face_file)}_{os.path.basename(audio_file)}.mp4',
#     ]
#
#     main(initialize_args(args_list))
#     os.remove(face_file)
#     os.remove(audio_file)
#
# def match_files(face_files, audio_files):
#     matched_pairs = []
#     for face_file in face_files:
#         face_number = os.path.splitext(os.path.basename(face_file))[0]
#         for audio_file in audio_files:
#             audio_number = os.path.splitext(os.path.basename(audio_file))[0]
#             if face_number == audio_number:
#                 matched_pairs.append((face_file, audio_file))
#                 break
#     return matched_pairs
# def get_files_from_directory(directory):
#     return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
#
# if __name__ == '__main__':
#     while True:
#         face_files = get_files_from_directory('video/face')
#         audio_files = get_files_from_directory('video/audio')
#
#         matched_pairs = match_files(face_files, audio_files)
#         for face_file, audio_file in matched_pairs:
#             test_wav2lip(face_file, audio_file)
