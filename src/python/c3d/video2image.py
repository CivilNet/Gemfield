#!/usr/bin/env python

import sys
import os
import subprocess
import array
import cv2

def extract_frames(video, frame_dir, num_frames_to_extract=16):
    # check output directory
    if os.path.isdir(frame_dir):
        print("[Warning] frame_dir={} does exist. Will overwrite".format(frame_dir))
    else:
        os.makedirs(frame_dir)

    # get number of frames
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("[Error] video={} can not be opened.".format(video))
        sys.exit(-1)

    # get frame counts
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_num_frames = num_frames - 15
    fps = cap.get(cv2.CAP_PROP_FPS)

    # move to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("[Info] Extracting {} frames from {}".format(num_frames, video))

    # grab each frame and save
    start_frames = []
    for frame_count in range(num_frames):
        frame_count += 1

        ret, frame = cap.read()
        if not ret:
            print("[Error] Frame {} extraction was not successful".format(frame_count))
            if len(start_frames) == 0:
                break
            #remove this clip
            del start_frames[-1]
            break

        frame_file = os.path.join(frame_dir, '{0:06d}.jpg'.format(frame_count))
        cv2.imwrite(frame_file, frame)
        #for last clip
        if frame_count % num_frames_to_extract == 1 and frame_count <= max_num_frames:
            start_frames.append(frame_count)

    return start_frames


def main(video_dir, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # generate auxilliary files for C3D feature extraction
    input_file = os.path.join(output_dir, 'input.txt')
    output_prefix_file = os.path.join(output_dir, 'output_prefix.txt')

    # first, populate input.txt, and output_prefix.txt files
    # each line corresponds to a 16-frame video clip
    f_input = open(input_file, 'w')
    f_output_prefix = open(output_prefix_file, 'w')
    dummy_label = 0
    for d in os.listdir(video_dir):
        print("parsing label {}".format(dummy_label))
        sub_dir = os.path.join(video_dir, d)
        for f in os.listdir(sub_dir):
            video_file = os.path.join(sub_dir, f)
            video_id, video_ext = os.path.splitext(os.path.basename(video_file))

            # where to save extracted frames
            frame_dir = os.path.join(output_dir, video_id)
            start_frames = extract_frames(video_file, frame_dir)

            for start_frame in start_frames:

                # write "input.txt" with just one clip
                f_input.write("{} {} {}\n".format(frame_dir, start_frame, dummy_label))

                # write "output_prefix.txt" with one clip
                clip_id = os.path.join(output_dir, video_id + '_{0:06d}'.format(start_frame))
                f_output_prefix.write("{}\n".format(os.path.join(output_dir, clip_id)))
        dummy_label += 1
    f_input.close()
    f_output_prefix.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: {} <input_video_dir> <output_dir>'.format(sys.argv[0]))
        sys.exit(1)
    video_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(video_dir, output_dir)
