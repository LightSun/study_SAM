

WORK_DIR=/home/heaven7/heaven7/work
PRE_MODULE=${WORK_DIR}/IAS/infer/SegmentAnything-TensorRT-main/src/models/sam_vit_h_preprocess.onnx
SAM_MODEL=${WORK_DIR}/IAS/infer/SegmentAnything-TensorRT-main/h7/sam_vit_h_4b8939.onnx
#IMAGE_PATH=${WORK_DIR}/IAS/infer/sam/images/truck.jpg
IMAGE_PATH=/home/heaven7/Pictures/farmhouse.jpg
#cuda:0
PRE_DEVICE=cpu
SAM_DEVICE=cpu
./sam_cpp_test -pre_model=${PRE_MODULE} -sam_model=${SAM_MODEL} -image=${IMAGE_PATH} -pre_device=${PRE_DEVICE} -sam_device=${SAM_DEVICE}
