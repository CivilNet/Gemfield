import sys
from onnx import onnx_pb
from onnx_coreml import convert

model_in = sys.argv[1]
model_out = sys.argv[2]
model_file = open(model_in, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
coreml_model = convert(model_proto, preprocessing_args= {'image_scale' : (1.0/255.0)}, image_input_names=['0'], image_output_names=['gemfield'])
coreml_model.save(model_out)
