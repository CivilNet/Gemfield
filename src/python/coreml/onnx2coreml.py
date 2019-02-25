import sys
from onnx import onnx_pb
from onnx_coreml import convert

model_in = sys.argv[1]
model_out = sys.argv[2]
model_file = open(model_in, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
print("prepare to convert...")
coreml_model = convert(model_proto, preprocessing_args= {'image_scale' : (1.0/255.0/58.50182),
                                                         'blue_bias':(-109.496254/255.0/58.50182),
                                                         'green_bias':(-118.698456/255.0/58.50182),
                                                         'red_bias':(-124.68751/255.0/58.50182),
                                                         'is_bgr':True},
    image_input_names=['gemfield'])#, image_output_names=['745'])
#coreml_model = convert(model_proto, preprocessing_args= {'image_scale' : (1.0/255), 'is_bgr':True}, image_input_names=['gemfield'],image_output_names=['745'])
coreml_model.save(model_out)
####
import coremltools
from coremltools.models.neural_network import flexible_shape_utils
spec = coremltools.utils.load_spec(model_out)
img_size_ranges = flexible_shape_utils.NeuralNetworkImageSizeRange()
img_size_ranges.add_height_range((384, 640))
img_size_ranges.add_width_range((384, 640))
flexible_shape_utils.update_image_size_range(spec, feature_name='gemfield', size_range=img_size_ranges)
#flexible_shape_utils.update_image_size_range(spec, feature_name='745', size_range=img_size_ranges)
coremltools.utils.save_spec(spec, 'flex_{}'.format(model_out))
######
