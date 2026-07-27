import sys
sys.path.append('/media/strawberry/KINGSTON/code/work-2/paper-3/ExposureMoE/ultralytics-8.3.0')
import os
from ultralytics import YOLO
import time 
import torch 
 
if __name__ == '__main__':
    imgs_path = 'ultralytics/datasets/MixExposure/images/val_640' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model = YOLO("runs/ripe/exposuremoe_640_01/weights/best.pt", verbose=False).to(device)
    images = os.listdir(imgs_path)
    
    #FPS
    for i in range(10):
        image = imgs_path + os.sep + images[i]
        results = model(image, verbose=False) #, imgsz=640

    start_time = time.time()  
    for item in images:
        image = imgs_path + os.sep + item

        results = model(image, imgsz=640, verbose=False)

    end_time = time.time()  
    elapsed_time = end_time - start_time 
    print('elapsed_time:', elapsed_time) 
    
    # 计算 FPS  
    fps = len(images) / elapsed_time  
    print(f"处理 {len(images)} 帧所耗时间: {elapsed_time:.2f} 秒")  
    print(f"FPS: {fps:.2f}")

    #计算模型的参数数量  
    v8_ripe = sum(p.numel() for p in model.model.parameters() )  

    print(f"v8_ripe total parameter /M: {(v8_ripe)/1000000}")

    #flops
    from thop import profile  

    input = torch.randn(1, 3, 480, 640)  # 示例输入
    input = input.to(device) 
     

    # 计算 FLOPs  
    flops, params = profile(model.model, inputs=(input,))  
    print(f'FLOPs: {flops/ 1e9}, Parameters: {params/1000000}')  

