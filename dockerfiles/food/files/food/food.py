#!/usr/bin/env python
import os
import logging
import optparse
from PIL import Image
import numpy as np
import caffe
import time
import tornado.wsgi
import tornado.httpserver
import flask
import datetime
import cStringIO as StringIO
import werkzeug
import urllib
import sys
reload(sys)  
sys.setdefaultencoding('utf8')

ORIENTATIONS = {   # used in apply_orientation
    2: (Image.FLIP_LEFT_RIGHT,),
    3: (Image.ROTATE_180,),
    4: (Image.FLIP_TOP_BOTTOM,),
    5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
    6: (Image.ROTATE_270,),
    7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
    8: (Image.ROTATE_90,)
}


REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = '/tmp/food_uploads'

# Obtain the flask app object
app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image %s open error: %s', imageurl, err)
        return flask.render_template('index.html', has_result=True, result=(False, 'Cannot open image from URL.'))

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template('index.html', has_result=True, result=result, imagesrc=imageurl)

@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving uploaded file to %s.', filename)
        image = app.clf.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template('index.html', has_result=True, result=(False, 'Cannot open uploaded image.'))

    result = app.clf.classify_image(image)
    #for i in result:
    #    print i[0] + ' : ' + i[1]
    return flask.render_template('index.html', has_result=True, result=result, imagesrc=embed_image_html(image))


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

class ImagenetClassifier(object):
    default_args = {
        'model_def_file': ('{}/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': ('{}/model.caffemodel'.format(REPO_DIRNAME)),
        'labels_cn_file': ('{}/food_cn.txt'.format(REPO_DIRNAME)),
        'labels_en_file': ('{}/food_en.txt'.format(REPO_DIRNAME))
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception("File for {} is missing. Should be at: {}".format(key, val))

    def __init__(self, model_def_file, pretrained_model_file, labels_cn_file, labels_en_file, gpu_mode):
        logging.info('Loading net and associated files...')

        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Classifier(model_def_file, pretrained_model_file, image_dims=(256, 256), 
            raw_scale=255,
            mean=np.array([98.0, 129.0, 153.0], dtype='f4'),
            #mean=None,
            channel_swap=(2, 1, 0))

        with open(labels_cn_file) as f:
            self.label_list = [l.strip() for l in f.readlines()]

    def open_oriented_im(self, im_path):
        im = Image.open(im_path)
        if hasattr(im, '_getexif'):
            exif = im._getexif()
            if exif is not None and 274 in exif:
                orientation = exif[274]
                im = self.apply_orientation(im, orientation)

        img = np.asarray(im).astype(np.float32) / 255.
        
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            img = np.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def apply_orientation(self, im, orientation):
        if orientation in ORIENTATIONS:
            for method in ORIENTATIONS[orientation]:
                im = im.transpose(method)
        return im


    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=False).flatten()
            #scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()

            indices = (-scores).argsort()[:5]
            logging.info('gemfield debug indices: %s and spent time: %s ', str(indices), '%.3f' % (endtime - starttime))
            predictions = [self.label_list[indice - 1] for indice in indices] 

            for i in range(0,5):
                print('gemfield debug each value: ' + predictions[i])

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [ (p, '%.5f' % scores[i]) for i, p in zip(indices, predictions)]

            logging.info('result: %s', str(meta))

            return (True, meta, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the image. Maybe try another one?')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    #create /tmp/food_upload
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    parser = optparse.OptionParser()
    parser.add_option('-g', '--gpu',help="use gpu mode",action='store_true', default=False)
    #parser.add_option('-f', '--file', help="image path")
    parser.add_option('-p', '--port', help="which port to serve content on", type='int', default=7030)


    opts, args = parser.parse_args()
    #if not opts.file:
    #    parser.error('filename not given')

    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    starttime = time.time()
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    endtime = time.time()
    print('gemfield construct time is: %.3f\n\n' % (endtime - starttime))

    app.clf.net.forward()
    endtime2 = time.time()
    print('gemfield forward time is: %.3f\n\n' % (endtime2 - endtime))

    #start the web server
    http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    http_server.listen(opts.port)
    print("Tornado server starting on port {}".format(opts.port))
    tornado.ioloop.IOLoop.instance().start()