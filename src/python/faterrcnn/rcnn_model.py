#!/usr/bin/python 

import numpy as np
import os
import sys
import cv2
import scipy.io as sio
from syszux_ftp import *

sys.path.append("/workspace/py-faster-rcnn/caffe-fast-rcnn/python/")
sys.path.append("/workspace/py-faster-rcnn/lib")
import caffe
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from bitstring import BitArray
import collections

os.environ['CUDA_VISIBLE_DEVICES']='0'

CLASSES = ('__background__','AnQiLa_woddp','AnQiLa_yz','BaiLiShouYue_yz','BaiLiXuanCe_yz','BaiQi_bsss','BaiQi_wwzlz','BaiQi_yz','BuZhiHuoWu_yz','CaiWenJi_yz','CaoCao_ssll','CaoCao_yz','ChengJiSiHan_wjldz','ChengYaoJin_ayzy','ChengYaoJin_xjlzd','ChengYaoJin_yz','DaJi_mlwjs','DaJi_npkf','DaJi_snAl','DaJi_yz','DaMo_dfmj','DaQiao_yswn','DaQiao_yz','DiRenJie_cskzs','DiRenJie_yys','DianWei_hjws','DiaoChan_yywn','DiaoChan_yz','DongHuangTaiYi_dhlw','DongHuangTaiYi_yz','GanJiangMoXie_dqro','GanJiangMoXie_yz','GaoJianLi_swyg','GaoJianLi_yz','GongBenWuZang_wljy','GuanYu_bfzs','GuanYu_yz','HanXin_bly','HanXin_jtbw','HouYi_jlw','HouYi_yz','HuaMuLan_sjllz','HuaMuLan_yz','HuangZhong_zjgjf','JiangZiYa_yz','JuYouJing_yz','Kai_lylz','LanLingWang_aylsz','LiBai_fhx','LiBai_fqh','LiBai_qnzh','LiBai_yz','LiYuanFang_hmatg','LiYuanFang_tzbd','LianPo_yz','LiuBang_dglbj','LiuBang_sdzg','LiuBang_yz','LuBanQiHao_dwxz','LuBanQiHao_flxd','LuBanQiHao_moqyj','LuBanQiHao_yz','LuBu_mrjj','LuNa_yz','MaKeBoLuo_jqly','MaKeBoLuo_yz','MiYue_yz','MoZi_lqs','NaKeLuLu_yz','NeZha_yz','NiuMo_yz','PianQue_hsbs','PianQue_jszt','PianQue_ljw','PianQue_yz','SuLie_yz','SunBin_yjw','SunShangXiang_hpqj','SunShangXiang_mrjj','SunShangXiang_qwlr','SunShangXiang_sgtx','SunShangXiang_yz','SunWuKong_yz','WangZhaoJun_oxgs','WuZeTian_hyzx','XiaHouDun_cfpl','XiaHouDun_yz','XiangYu_yz','XiangYu_zbwp','XiaoQiao_bfdjs','XiaoQiao_cbhj','XiaoQiao_wsqy','XiaoQiao_yz','YaDianNa_yz','YaSe_swqs','YaSe_sxw','YaSe_yz','YangJian_ajfl','YangJian_yz','YingZheng_yylr','YingZheng_yz','YuJi_bwbj','YuJi_jlbxj','YuJi_ketnw','YuJi_yz','ZhangFei_lshc','ZhangFei_wftx','ZhangFei_yz','ZhangLiang_ttfy','ZhangLiang_yqlyy','ZhangLiang_yz','ZhaoYun_yz','ZhenJi_bxywq','ZhenJi_hhrj','ZhenJi_yyjm','ZhenJi_yz','ZhongKui_dfpg','ZhongKui_yz','ZhongWuYan_htly','ZhongWuYan_shjj','ZhongWuYan_wzzc','ZhouYu_hjdj','ZhouYu_yz','ZhouYu_zazs','ZhuGeLiang_hjffgl','ZhuGeLiang_xhzhg','ZhuGeLiang_yz','ZhuangZhou_slw','ZhuangZhou_yz','other')
MODEL_DEF = '../faster_rcnn_deploy.prototxt'
PRETRAINED_MODEL = '../models/vgg16_faster_rcnn_130cls_iter_214000.caffemodel'
NUM_CLASSES = 128
PATTERN_DICT = collections.OrderedDict()
PATTERN_DICT['0b101'] = '0b111'
PATTERN_DICT['0b110011'] = '0b111111'
PATTERN_DICT['0b111000111'] = '0b111111111'
PATTERN_DICT['0b111100001111'] = '0b111111111111'
PATTERN_DICT['0b010'] = '0b000'
PATTERN_DICT['0b001100'] = '0b000000'

label_map_file = 'label_map.txt'
db_map_file = 'db_map.txt'
label_map = {}
db_map = {}
def create_map():
    # label_map = dict( [x.strip().split(',') for x in open(map_file).readlines()] )
    with open(label_map_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line is None or len(line) == 0:
                continue
            k, v = line.split(',')
            label_map[int(k)] = v.strip()

    with open(db_map_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line is None or len(line) == 0:
                continue
            k, v = line.split(',')
            db_map[k] = v.strip() 

def post_process(mat, fps=25.0, th=0.9):
    print('start post_process')
    result_dic = []
    frame_cnt, heros = mat.shape
    assert heros==NUM_CLASSES, 'mat shape is invalid, (128 vs {})'.format(heros)
    for j in range(heros):
        bstr = BitArray()
        for i in mat[:,j]:
            if i > th:
                bstr.append('0b1')
            else:
                bstr.append('0b0')
        for k in PATTERN_DICT:
            bstr.replace(k, PATTERN_DICT[k])
            bstr.replace(k, PATTERN_DICT[k])
        i = 0
        while i < frame_cnt:
            while (i < frame_cnt and bstr[i] == False):
                i+=1
            start_frame = i*10
            while(i < frame_cnt and bstr[i] == True):
                i+=1
            end_frame = i*10
            if end_frame > start_frame:
                result_dic.append( {'hero_id':db_map[label_map[j+1].split('_')[0]], 'start_frame':start_frame, 'end_frame':end_frame, 'start_time':start_frame/fps, 'end_time':(end_frame)/fps} ) # map and fps
    return result_dic


def modelInit():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(MODEL_DEF, PRETRAINED_MODEL, caffe.TEST)
    return net

def objectPredict(remote_filename, task_id, net):
    #remote_filename = remote_filename.encode('utf-8')
    obj_video_recv = 'object/video_recv/{}'.format(task_id)
    base_remote_filename = os.path.basename(remote_filename)
    if not os.path.exists( obj_video_recv ):
        os.makedirs( obj_video_recv )
    if not os.getenv('WITH_DEBUG'):
        local_filename = os.path.join(obj_video_recv, base_remote_filename)
        print('prepare getFtpFile from %s to %s' %(remote_filename, local_filename))
        getFtpFile(remote_filename, local_filename)
        print('getFtpFile Done for objectPredict')
    else:
        local_filename = remote_filename

    video_cap = cv2.VideoCapture(str(local_filename))
    frame_cnt = int(video_cap.get( cv2.CAP_PROP_FRAME_COUNT ))
    fps = video_cap.get( cv2.CAP_PROP_FPS )
    sample_frames = range(0,frame_cnt,10)
    result_mat = np.zeros([len(sample_frames), NUM_CLASSES])

    # create image directory if not exist
    frame_dir = os.path.join(obj_video_recv, os.path.basename(local_filename) + '_frames')
    if not os.path.exists( frame_dir ):
        os.makedirs( frame_dir )

    ERROR_FRAMES = 10;
    for i, frame_index in enumerate(sample_frames):
        is_print = (i%100 == 0)
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        status, frame = video_cap.read()
        if not status:
            ERROR_FRAMES -= 1
            print('[TFWARING] fetch frame {} failed, total frames is {}, task_id is {}'.format(frame_index, frame_cnt, task_id))
            if ERROR_FRAMES == 0:
                sess.close()
                raise Exception( 'fetch frame {} failed, total frames is {}, task_id is {}'.format(frame_index, frame_cnt, task_id) )
            continue
                
        scores, boxes = im_detect(net, frame)
        scores = scores[:, 1:-1]
        max_scores = np.max(scores, axis=0) # the highest score for each cls
        result_mat[i, :] = max_scores
        #img_name = '{}.jpg'.format( str(frame_index) )
        #frame_filename = os.path.join(frame_dir, img_name)
        # print('save image.. ', frame_filename)
        #cv2.imwrite(frame_filename, frame)
        if is_print:
            percentage = '{:.2f}%'.format(frame_index * 100.0/ frame_cnt)
            print("| {} | {} | {} | {:06d}/{:06d} | {} |".format('OBJECT', task_id, base_remote_filename, frame_index, frame_cnt, percentage))

    result_dic = post_process(result_mat, fps=fps)
    for i in result_dic:
        print('==================')
        for k in i:
            print(k, i[k])
        print('==================') 
    return result_dic

def upgradeObject(remote_filename, task_id, *args):
    obj_video_recv = 'object/upgrade_object/{}'.format(task_id)
    if not os.path.exists( obj_video_recv ):
        os.makedirs( obj_video_recv )
    local_filename = os.path.join(obj_video_recv, os.path.basename(remote_filename))
    if os.path.exists( local_filename ):
        os.remove(local_filename)
    print('prepare getFtpFile from %s to %s' %(remote_filename, local_filename))
    getFtpFile(remote_filename, local_filename)
    msg = 'getFtpFile Done for upgradeObject'
    print(msg)
    return msg


if __name__ == '__main__':
    os.environ['WITH_DEBUG']='1'
    net = modelInit()
    create_map()

    video_f = 'wzry222.mp4'
    abs_video_f = '/home/dxx/object_detection/test/wzry222.mp4'
    objectPredict(abs_video_f, video_f[:-4] , net)
    print(label_map)
    print(db_map)
