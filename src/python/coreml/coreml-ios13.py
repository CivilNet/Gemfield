from onnx_coreml import convert
class_labels = ["travel","sport","playground","station","office","pet"]

scale = 1.0 / (0.226 * 255.0)
red_scale = -0.485 / (0.229 * 255.0)
green_scale = -0.456 / (0.224 * 255.0)
blue_scale = -0.406 / (0.225 * 255.0)

args = dict(is_bgr=False, image_scale = scale, red_bias = red_scale, green_bias = green_scale, blue_bias = blue_scale,image_format='NCHW')

model_coreml = convert(model="syszux_scene.onnx",preprocessing_args= args,mode='classifier', image_input_names=['image'],class_labels=class_labels, predicted_feature_name='classLabel',minimum_ios_deployment_target='13')
model_coreml.save('mobilenet_v3.mlmodel')
