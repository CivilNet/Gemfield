import argparse
import logging
import os
import sys
import av
# 3 threads for numpy
import numpy as np
import time
# 1 thread for cv2
import cv2

def processStream(football_live_1):
    print('before av open')

    input_file = av.open(football_live_1)
    print('after av open')

    input_video_stream = next((s for s in input_file.streams if s.type == 'video'), None)
    input_audio_stream = next((s for s in input_file.streams if s.type == 'audio'), None)
    if input_video_stream is None:
        raise Exception('input video is None...')
    if input_audio_stream is None:
        raise Exception('input audio is None...')

    packet_c = 0
    video_c = 0
    audio_c = 0
    
    start_time = time.time()
    print('start_time: ',start_time)
    for packet in input_file.demux([s for s in (input_video_stream, input_audio_stream) if s]):
        packet_c += 1

        for frame in packet.decode():
            if packet.stream.type == b'video':
                video_c += 1
                # some other formats gray16be, bgr24, rgb24
                #img = frame.to_nd_array(format='bgr24')
                img = np.frombuffer(frame.planes[0], np.uint8).reshape(frame.height, frame.width, -1)
                #jpeg_bytes = cv2.imencode('.jpg', img)[1].tobytes()
                # print(len(jpeg_bytes))
            else:
                audio_c += 1

            # Signal to generate it's own timestamps.
            frame.pts = None

    end_time = time.time()
    duration = end_time - start_time
    print('end_time: ',end_time)
    print(packet_c, video_c,audio_c, duration)
    print('time per video frame: ', duration/video_c * 1.0)
    print('video frames per second: ', video_c * 1.0 / duration )

test_mp4 = '/home/gemfield/gemfield.mp4'
if __name__ == '__main__':
    print('try to open stream {}'.format(test_mp4))
    processStream(test_mp4)
