# Author: Gemfield
import argparse
import numpy as np
import time
import os
import sys
import cv2
from matplotlib import pyplot

from caffe2.python import core,workspace,utils,net_drawer,cnn,optimizer,model_helper,brew,visualize
from caffe2.proto import caffe2_pb2
from caffe2.python.modeling.initializers import Initializer
import caffe2.python.models.resnet as resnet

INIT_NET = 'resnet50_pb/init_net.pb'
PREDICT_NET = 'resnet50_pb/predict_net.pb'

# to decode input videos
class DecodeBuffer(object):
    def __init__(self):
        self.decode_buffer = []
        self.buffer_size = 16
        self.yield_size = 8
        self.stride = 2
        self.size = (224, 224)
        self.width, self.height = self.size
        self.use_rgb = False
        self.std = 128
        self.mean = 128
        self.frame = None

    def processFrame(self):
        if self.frame is None:
            raise Exception('internal error')
        # in current case, gemfield just crop part of input image to do inference
        w = self.frame.shape[1]
        h = self.frame.shape[0]
        w_s = w//6
        w_e = w - w//6
        h_s = 0
        h_e = h//2
        self.frame = self.frame[h_s:h_e, w_s:w_e,:]

        self.frame = cv2.resize(self.frame, self.size)
        if self.use_rgb:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

    def decode(self, f):
        cap = cv2.VideoCapture(f)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while True:
            status, self.frame = cap.read()
            current_frame_index = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not status:
                if current_frame_index > total_frame_num - 5:
                    return
                continue
            if current_frame_index % self.stride != 0:
                continue

            current_time = current_frame_index/fps
            print('[DECODER] {} {}/{}'.format(current_time, current_frame_index, total_frame_num))
            self.processFrame()
            yield self.frame

# for train and test
def AddImageInput(model, reader, batch_size, is_test):
    print('device_type: ', model._device_type)
    data, label = brew.image_input(
        model,
        reader, ["data", "label"],
        batch_size=batch_size,
        output_type='float',
        use_gpu_transform=True,
        use_caffe_datum=False,
        #mean=128.,
        #std=128.,
        std=57.375,
        mean=114.75,
        scale=227,
        crop=227,
        mirror=1,
        is_test=is_test,
    )
    data = model.StopGradient(data, data)

def vis(blob):
    channel = 0
    prefix = 'gpu_0'
    pyplot.title("Input Image Channel {} (by Gemfield)".format(channel))
    data = workspace.FetchBlob('{}/{}'.format(prefix, blob))
    #data = data[:,[channel],:,:]
    visualize.NCHW.ShowMultiple(data)
    pyplot.show()

def testVideo(model_name, video_file):
    dbuf = DecodeBuffer()
    gemfield = []
    print('DECODE: ',video_file)
    for frame in dbuf.decode(video_file):
        frame = (frame -114.75) / 57.375
        frame = frame.transpose([2,0,1])

        gemfield.append(frame)
        if len(gemfield) != 32:
            continue

        clip = np.array(gemfield).reshape([32, 3, 224, 224]).astype('float32')
        workspace.FeedBlob("gpu_0/data", clip,  device_option=core.DeviceOption(caffe2_pb2.CUDA, 0))
        workspace.RunNet(model_name)
        pred = workspace.FetchBlob("gpu_0/softmax")
        print(pred)
        vis('data')

def saveNet(model) :
    with open(PREDICT_NET, 'wb') as f:
        f.write(model.net._net.SerializeToString())

    init_net = caffe2_pb2.NetDef()
    extra_params = []
    for blob in workspace.Blobs():
        name = str(blob)
        if name.endswith("_rm") or name.endswith("_riv"):
            extra_params.append(name)

    for name in extra_params:
        model.params.append(name)

    for param in model.params:
        print(param)
        blob = workspace.FetchBlob(param)
        shape = blob.shape
        op = core.CreateOperator("GivenTensorFill", [], [param],arg=[ utils.MakeArgument("shape", shape),utils.MakeArgument("values", blob)])
        init_net.op.extend([op])
    init_net.op.extend([core.CreateOperator("ConstantFill", [], ["gpu_0/data"], shape=(1,224,224))])
    with open(INIT_NET, 'wb') as f:
        f.write(init_net.SerializeToString())

def deployAndTest(args):
    device_opt = core.DeviceOption(caffe2_pb2.CUDA, 0)

    init_def = caffe2_pb2.NetDef()
    with open(INIT_NET, 'rb') as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(device_opt)
        workspace.RunNetOnce(init_def)
    
    net_def = caffe2_pb2.NetDef()
    with open(PREDICT_NET, 'rb') as f:
        net_def.ParseFromString(f.read())
        net_def.device_option.CopyFrom(device_opt)
        workspace.CreateNet(net_def, overwrite=True)

    if args.input_video is None:
        raise Exception('Should provide input video when you do test')
    testVideo(net_def.name, args.input_video)

#train_test_deploy=0 for train
#train_test_deploy=1 for test
#train_test_deploy=2 for deploy  
def CivilNet(name, train_test_deplopy=0):
    arg_scope = {'order': 'NCHW','use_cudnn': True,'cudnn_exhaustive_search': True,'ws_nbytes_limit': (64 * 1024 * 1024)}
    model = model_helper.ModelHelper(name=name, arg_scope=arg_scope)

    model._device_type = caffe2_pb2.CUDA
    model._device_prefix = "gpu"
    model._shared_model = False
    model._devices = [0]
    device_opt = core.DeviceOption(caffe2_pb2.CUDA, 0)
    
    #for deploy
    if train_test_deplopy == 2:
        with core.DeviceScope(device_opt):
            with core.NameScope("{}_{}".format(model._device_prefix,0)):
                with brew.arg_scope([brew.conv, brew.fc], WeightInitializer=Initializer,
                        BiasInitializer=Initializer,enable_tensor_core=False,float16_compute=False):
                    resnet.create_resnet50(model,"data",num_input_channels=3,num_labels=args.num_labels,no_bias=True,no_loss=False)
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net)
        return model

    reader_name = "reader" if train_test_deplopy == 0 else "test_reader"
    reader_data = args.train_data if train_test_deplopy == 0 else args.test_data
    reader = model.CreateDB(reader_name,db=reader_data,db_type='lmdb',num_shards=1,shard_id=0)

    is_test = True if train_test_deplopy == 1 else False
    loss = None
    with core.DeviceScope(device_opt):
        with core.NameScope("{}_{}".format(model._device_prefix,0)):
            AddImageInput(model,reader, batch_size=32,is_test=is_test)
            with brew.arg_scope([brew.conv, brew.fc], WeightInitializer=Initializer,
                    BiasInitializer=Initializer,enable_tensor_core=False,float16_compute=False):
                pred = resnet.create_resnet50(model,"data",num_input_channels=3,num_labels=args.num_labels,no_bias=True,no_loss=True)
            softmax, loss = model.SoftmaxWithLoss([pred, 'label'],['softmax', 'loss'])
            brew.accuracy(model, [softmax, "label"], "accuracy")
    #for test
    if train_test_deplopy == 1:
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net)
        return model
    
    #for train
    loss_grad = {}
    losses_by_gpu = {}
    losses_by_gpu[0] = [loss]
    #add grad
    def create_grad(lossp):
        return model.ConstantFill(lossp, str(lossp) + "_grad", value=1.0)

    # Explicitly need to create gradients on GPU 0
    device = core.DeviceOption(model._device_type, 0)
    with core.DeviceScope(device):
        for l in losses_by_gpu[0]:
            lg = create_grad(l)
            loss_grad[str(l)] = str(lg)

        model.AddGradientOperators(loss_grad)
    #end add grad
        optimizer.add_weight_decay(model, args.weight_decay)
        stepsz = int(30 * args.epoch_size / 32)
        opt = optimizer.build_multi_precision_sgd(model,args.base_learning_rate,
            momentum=0.9,nesterov=1,policy="step",stepsize=stepsz,gamma=0.1)
        model._optimizer = opt

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    return model

    
def trainAndSave(args):
    # train_arg_scope = {'order': 'NCHW','use_cudnn': True,'cudnn_exhaustive_search': True,'ws_nbytes_limit': (64 * 1024 * 1024)}
    # train_model = model_helper.ModelHelper(name="resnet50", arg_scope=train_arg_scope)
    # reader = train_model.CreateDB("reader",db=args.train_data,db_type='lmdb',num_shards=1,shard_id=0)

    # train_model._device_type = caffe2_pb2.CUDA
    # train_model._device_prefix = "gpu"
    # train_model._shared_model = False
    # train_model._devices = [0]

    # device_opt = core.DeviceOption(train_model._device_type, 0)
    # with core.DeviceScope(device_opt):
    #     with core.NameScope("{}_{}".format(train_model._device_prefix,0)):
    #         AddImageInput(train_model,reader, batch_size=32,is_test=False)
    #         with brew.arg_scope([brew.conv, brew.fc], WeightInitializer=Initializer,
    #                 BiasInitializer=Initializer,enable_tensor_core=False,float16_compute=False):
    #             pred = resnet.create_resnet50(train_model,"data",num_input_channels=3,num_labels=args.num_labels,no_bias=True,no_loss=True)
    #         softmax, loss = train_model.SoftmaxWithLoss([pred, 'label'],['softmax', 'loss'])
    #         brew.accuracy(train_model, [softmax, "label"], "accuracy")
    # losses_by_gpu = {}
    # losses_by_gpu[0] = [loss]
    # #add grad
    # def create_grad(lossp):
    #     return train_model.ConstantFill(lossp, str(lossp) + "_grad", value=1.0)

    # loss_grad = {}
    # # Explicitly need to create gradients on each GPU
    # device = core.DeviceOption(train_model._device_type, 0)
    # with core.DeviceScope(device):
    #     for l in losses_by_gpu[0]:
    #         lg = create_grad(l)
    #         loss_grad[str(l)] = str(lg)

    #     train_model.AddGradientOperators(loss_grad)
    # #end add grad
    #     optimizer.add_weight_decay(train_model, args.weight_decay)
    #     stepsz = int(30 * args.epoch_size / 32)
    #     opt = optimizer.build_multi_precision_sgd(train_model,args.base_learning_rate,
    #         momentum=0.9,nesterov=1,policy="step",stepsize=stepsz,gamma=0.1)
    #     train_model._optimizer = opt

    # workspace.RunNetOnce(train_model.param_init_net)
    # workspace.CreateNet(train_model.net)

    # # start test
    # test_arg_scope = {'order': 'NCHW','use_cudnn': True,'cudnn_exhaustive_search': True}
    # test_model = model_helper.ModelHelper(name="resnet50_test", arg_scope=test_arg_scope)
    # test_reader = test_model.CreateDB("test_reader",db=args.test_data,db_type='lmdb',num_shards=1,shard_id=0)

    # test_model._device_type = caffe2_pb2.CUDA
    # test_model._device_prefix = "gpu"
    # test_model._shared_model = False
    # test_model._devices = [0]

    # device_opt_test = core.DeviceOption(test_model._device_type, 0)
    # with core.DeviceScope(device_opt_test):
    #     with core.NameScope("{}_{}".format(test_model._device_prefix,0)):
    #         AddImageInput(test_model,test_reader, batch_size=32,is_test=True)
    #         with brew.arg_scope([brew.conv, brew.fc], WeightInitializer=Initializer,
    #                 BiasInitializer=Initializer,enable_tensor_core=False,float16_compute=False):
    #             resnet.create_resnet50(test_model,"data",num_input_channels=3,num_labels=args.num_labels,no_bias=True,no_loss=False)
    # workspace.RunNetOnce(test_model.param_init_net)
    # workspace.CreateNet(test_model.net)
    ####

    train_model = CivilNet('resnet50', 0)
    test_model = CivilNet('resnet50_test', 1)
    deploy_model = CivilNet('resnet50_deploy', 2)
    #start train loop
    for epoch_iter in range(args.num_epochs):
        iter_num = args.epoch_size // args.batch_size
        for i in range(iter_num):
            t1 = time.time()
            workspace.RunNet(train_model.net.Proto().name)
            t2 = time.time()
            dt = t2 - t1
            fmt = "Finished iteration {}/{} of epoch {} ({:.2f} images/sec)"
            print(fmt.format(i + 1, iter_num, epoch_iter + 1, 32/dt))
            prefix = "{}_{}".format(train_model._device_prefix,train_model._devices[0])
            loss = workspace.FetchBlob(prefix + '/loss')
            accuracy = workspace.FetchBlob(prefix + '/accuracy')
            #print(workspace.FetchBlob(prefix + '/label'))
            train_fmt = "Training loss in gemfield.org: {}, accuracy: {}"
            print(train_fmt.format(loss, accuracy))
            #vis('data')
            assert loss < 40, "Exploded gradients :("
        # test video after every epoch
        if args.test_data is None:
            continue
        
        for i in range(50):
            workspace.RunNet(test_model.net.Proto().name)
            #vis('data')
            prefix = "{}_{}".format(test_model._device_prefix,test_model._devices[0])
            loss = workspace.FetchBlob(prefix + '/loss')
            accuracy = workspace.FetchBlob(prefix + '/accuracy')
            test_fmt = "Testing prediction in gemfield.org: {}, accuracy: {}"
            print(test_fmt.format(loss, accuracy))

        if args.input_video is None:
            continue
        
        testVideo(deploy_model.net.Proto().name, args.input_video)
        saveNet(deploy_model)

def makeSureDirExistForFile(file):
    #create dir to save and load models
    dir = os.path.dirname(file)
    if not os.path.exists(dir):
        os.mkdir(dir)

# main function
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    parser = argparse.ArgumentParser(description="Caffe2: Resnet-50 training by Gemfield")
    parser.add_argument("--train_data", type=str, default=None,help="Path to training data (or 'null' to simulate)")
    parser.add_argument("--test_data", type=str, default=None,help="Path to test data")
    parser.add_argument("--image_size", type=int, default=227,help="Input image size (to crop to)")
    parser.add_argument("--num_labels", type=int, default=1000,help="Number of labels")
    parser.add_argument("--batch_size", type=int, default=32,help="Batch size, total over all GPUs")
    parser.add_argument("--epoch_size", type=int, default=1500000,help="Number of images/epoch, total over all machines")
    parser.add_argument("--num_epochs", type=int, default=1000,help="Num epochs.")
    parser.add_argument("--base_learning_rate", type=float, default=0.01,help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,help="Weight decay (L2 regularization)")
    parser.add_argument("--input_video", type=str, default=None,help="Input video for test")
    parser.add_argument("--deploy", type=bool, default=False,help="Train/Test or Deploy")
    args = parser.parse_args()

    makeSureDirExistForFile(INIT_NET)
    makeSureDirExistForFile(PREDICT_NET)
    #Train
    if not args.deploy:
        trainAndSave(args)
    else:
        deployAndTest(args)


