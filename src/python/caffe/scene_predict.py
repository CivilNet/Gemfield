#!/usr/bin/env python
import numpy as np
import os, stat
import sys
import argparse
import glob
import time
import timeit
import caffe
import cv2
#target dir
def getTargetDir(topdir, category, subdir):
    root_dir = '/home/gemfield/dataset/output'
    #background
    if category == 0:
        return '{}/{}/{}/{}'.format(root_dir, topdir, category, subdir)
    return '{}/{}/{}'.format(root_dir, topdir, category)
#/opt/caffe/build/tools/caffe train -solver solver.prototxt -weights inception-v3.caffemodel -gpu 1
#/opt/caffe/build/tools/caffe time --model train_val.prototxt -gpu 1
class Clf(object):
    def __init__(self, device=0, model_def='resnet_deploy.prototxt', pretrained_model='resnet_v79_iter_3000.caffemodel', dims = (160,352)):
        self.model_def = model_def
        self.pretrained_model = pretrained_model
        self.image_dims = dims
        self.favorite_size = (1280,720)

        caffe.set_mode_gpu()
        caffe.set_device(device)
        ######
        self.net = caffe.Net(model_def,pretrained_model,caffe.TEST)
        #self.net.blobs['data'].reshape(1,3,160,352)
        ######
        # self.classifier = caffe.Classifier(self.model_def, self.pretrained_model,
        #     image_dims=self.image_dims, mean=None,
        #     input_scale=None, raw_scale=1,
        #     channel_swap=None)

        for img_type in ['resnet', 'inception']:
            for category in range(17):
                frame_dir = getTargetDir(img_type, category, 'init')
                if not os.path.exists( frame_dir ):
                    os.makedirs( frame_dir )
                    os.chmod( frame_dir, stat.S_IRWXO )

    def scenePredict(self, local_filename):
        #generate the classfied images
        video_id = os.path.basename(local_filename)[:-4]
        for img_type in ['resnet', 'inception']:
            frame_dir = getTargetDir(img_type, 0, video_id)
            #workaround, since break in the middle phase
            if os.path.exists(frame_dir):
                print('directory {} already exists, generated before, omit...'.format(frame_dir))
                return False
            if not os.path.exists( frame_dir ):
                os.makedirs( frame_dir )
                os.chmod( frame_dir, stat.S_IRWXO )
      
        # to be desprated
        video_cap = cv2.VideoCapture(str(local_filename))
        fps = int(video_cap.get( cv2.CAP_PROP_FPS ))
        frame_cnt = int(video_cap.get( cv2.CAP_PROP_FRAME_COUNT ))
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        failed_cnt = frame_cnt/50
        frame_index = 0

        print("{} resolution:{}x{} ratio:{}".format(local_filename, video_width, video_height, video_width/video_height))

        while(failed_cnt > 0 and frame_index < frame_cnt):
            frame_index += 1
            status, frame = video_cap.read()
            if not status:
                #[mov,mp4,m4a,3gp,3g2,mj2 @ 0x7efba6a03600] stream 0, offset 0xdcd7614: partial file
                #[mov,mp4,m4a,3gp,3g2,mj2 @ 0x7efba6a03600] stream 1, offset 0xdcd7642: partial file
                #......
                print("failed at index {}".format(frame_index))
                failed_cnt -= 1
                continue
            #we have not too many gpu resource
            if frame_index % 1 != 0:
                continue
            ###################################################
            frame = cv2.resize(frame, self.favorite_size)
            #input_frame = frame[150-32:150+32, 640-96:640+96, :]
            input_frame_resnet = frame[150-80:150+80, 640-176:640+176, :]
            input_frame_inception = frame[150-100:150+260, 640-180:640+180, :]
            start_time = timeit.default_timer()
            
            # transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
            # transformer.set_transpose('data', (2,0,1))
            # transformer.set_mean('data', np.array([0,0,0]))#np.array([104,117,123]
            # transformer.set_raw_scale('data', 255)
            # transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead if BGR

            #input_img = input_frame_resnet[...]
            self.net.blobs['data'].data[0,...] = input_frame_resnet.transpose((2,0,1))
            #transformer.preprocess('data', input_img)
            #net_input = self.net.blobs['data'].data
            out = self.net.forward()
            predictions = self.net.blobs['prob'].data
            # print('prediction shape: ', predictions.shape)
            ####################################################
            pred_class = np.argmax(predictions, axis=1)[0]
            pred_score = predictions.max(axis=1)[0]
            end_time = timeit.default_timer()
            print('forward frame {} in {} cost {}'.format( frame_index, frame_cnt, str(end_time-start_time) ) )

            frame_dir_resnet = getTargetDir('resnet', pred_class, subdir=video_id)
            frame_dir_inception = getTargetDir('inception', pred_class, subdir=video_id)

            img_name = '{:.4f}_{}_{:06d}.jpg'.format( pred_score, video_id,  frame_index )
            frame_filename_resnet = os.path.join(frame_dir_resnet, img_name)
            frame_filename_inception = os.path.join(frame_dir_inception, img_name)

            #print("writing {}".format(frame_filename_resnet))
            #######begin the test#########
            # start_time = timeit.default_timer()
            # predictions = self.classifier.predict( [input_frame_resnet,], True)
            # pred_class = np.argmax(predictions, axis=1)[0]
            # pred_score = predictions.max(axis=1)[0]
            # end_time = timeit.default_timer()
            # print('forward frame {} in {} cost {}'.format( frame_index, frame_cnt, str(end_time-start_time) ) )

            # frame_dir_resnet = getTargetDir('resnet', pred_class, subdir=video_id)
            # frame_dir_inception = getTargetDir('inception', pred_class, subdir=video_id)

            # img_name = '{:.4f}_{}_{:06d}.jpg'.format( pred_score, video_id,  frame_index )
            # frame_filename_resnet = os.path.join(frame_dir_resnet, img_name)
            # frame_filename_inception = os.path.join(frame_dir_inception, img_name)
            # print("writing {}".format(frame_filename_resnet))
            # print('')
            #######end the test#########
            print("writing {}".format(frame_filename_resnet))
            cv2.imwrite(frame_filename_resnet, input_frame_resnet)

            print("writing {}".format(frame_filename_inception))
            cv2.imwrite(frame_filename_inception, input_frame_inception)
        video_cap.release()
        return True

if __name__ == '__main__':
    clf = Clf(device=1)
    #video_path = '/bigdata/wzry/gametagging/video/'
    video_path = '/bigdata/video_annotation_web/static/videos/'
    for f in os.listdir(video_path):
        if not f.endswith('.mp4'):
            continue
        abs_f = os.path.join(video_path, f)
        clf.scenePredict(abs_f)
