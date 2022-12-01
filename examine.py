import os
import sys
import numpy as np
import tensorrt as trt
from common import *
import pycuda.driver as cuda
import pycuda.autoinit
import pickle as pkl

if __name__ == '__main__':

    #expoert onnx
    onnx_f = "height_trans.onnx"
    onnx_values_f = "flownet.onnx.values"


    front_left_path = "/home/zhao/workspace/FastFlowNet/tensorrt_workspace/4cam_comperasion/training/front_left_feature/000008.npy"
    front_right_path = "/home/zhao/workspace/FastFlowNet/tensorrt_workspace/4cam_comperasion/training/front_right_feature/000008.npy"
    side_left_path = "/home/zhao/workspace/FastFlowNet/tensorrt_workspace/4cam_comperasion/training/side_right_feature/000008.npy"
    side_right_path = "/home/zhao/workspace/FastFlowNet/tensorrt_workspace/4cam_comperasion/training/side_right_feature/000008.npy"
    calib_path = "/home/zhao/workspace/FastFlowNet/tensorrt_workspace/4cam_comperasion/training/calib/000008.pkl"
    bev_feature = "/home/zhao/workspace/FastFlowNet/tensorrt_workspace/4cam_comperasion/training/bev_features/000008.npy"
    
    front_left_feature = np.load(front_left_path).transpose((0, 2, 3, 1))
    front_right_feature = np.load(front_right_path).transpose((0, 2, 3, 1))
    side_left_feature = np.load(side_left_path).transpose((0, 2, 3, 1))
    side_right_feature = np.load(side_right_path).transpose((0, 2, 3, 1))
    
    
    with open(calib_path, 'rb') as f:
        calib = pkl.load(f, encoding='latin1')
        # calib = ret['calib'][batch_idx]
        # print(calib)
        P1 = calib['P1']
        P2 = calib['P2']
        Tr_cam_to_imu = calib['Tr_cam_to_imu']
        Tr_imu_to_cam = np.linalg.pinv(Tr_cam_to_imu)

        P_side_left = calib['P_side_left']
        Tr_cam_to_imu_side_left = calib['Tr_cam_to_imu_side_left']
        Tr_imu_to_cam_side_left = np.linalg.pinv(Tr_cam_to_imu_side_left)
        P_side_right = calib['P_side_right']
        Tr_cam_to_imu_side_right = calib['Tr_cam_to_imu_side_right']
        Tr_imu_to_cam_side_right = np.linalg.pinv(Tr_cam_to_imu_side_right)

        xyz2left = np.matmul(P1, Tr_imu_to_cam)
        xyz2right = np.matmul(P2, Tr_imu_to_cam)
        xyz2side_left = np.matmul(P_side_left, Tr_imu_to_cam_side_left)
        xyz2side_right = np.matmul(P_side_right, Tr_imu_to_cam_side_right)
        
        xyz2left = np.expand_dims(np.expand_dims(xyz2left, 0), 0)
        xyz2right = np.expand_dims(np.expand_dims(xyz2right, 0), 0)
        xyz2side_left = np.expand_dims(np.expand_dims(xyz2side_left, 0), 0)
        xyz2side_right = np.expand_dims(np.expand_dims(xyz2side_right, 0), 0)
        
        # print(front_left_feature.shape)
        # print(front_right_feature.shape)
        # print(side_left_feature.shape)
        # print(side_left_feature.shape)
        
        # print(xyz2left.shape)
        # print(xyz2right.shape)
        # print(xyz2side_left.shape)
        # print(xyz2side_right.shape)
        # exit()

    with build_engine_onnx(onnx_f,fp16=True) as engine, open(onnx_f.split('.')[0],'wb') as f:
        f.write(engine.serialize())
        inputs,outputs,bindings,stream = allocate_buffers(engine, True, 1)
        for binding in engine:
            print('-------------------')
            print(engine.get_binding_shape(binding))
            print(engine.get_binding_name(engine.get_binding_index(binding)))
        with engine.create_execution_context() as context:
        #     input_t = input_t.float().numpy()
            inputs[0].host = np.ascontiguousarray(front_left_feature).astype(np.float16)
            inputs[1].host = np.ascontiguousarray(front_right_feature).astype(np.float16)
            inputs[2].host = np.ascontiguousarray(side_left_feature).astype(np.float16)
            inputs[3].host = np.ascontiguousarray(side_right_feature).astype(np.float16)
            
            inputs[4].host = np.ascontiguousarray(xyz2left).astype(np.float16)
            inputs[5].host = np.ascontiguousarray(xyz2right).astype(np.float16)
            inputs[6].host = np.ascontiguousarray(xyz2side_left).astype(np.float16)
            inputs[7].host = np.ascontiguousarray(xyz2side_right).astype(np.float16)
            
            trt_outputs = do_inference_v2(context,bindings=bindings,inputs=inputs,outputs=outputs,stream=stream)
            output = trt_outputs[0].reshape((1, 64, 80, 200))
            np.save("bev_feature.npy", output)

        #     flow = div_flow * output
        #     flow = np.transpose(flow,[1,2,0])
        #     flow_color = flow_to_color(flow,convert_to_bgr=True)

        #     cv2.namedWindow('tensorrt flow',cv2.WINDOW_NORMAL)
        #     cv2.imshow('tensorrt flow',flow_color)
        #     cv2.waitKey(0)

