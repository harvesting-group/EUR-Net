import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import sys
sys.path.append('/media/strawberry/KINGSTON/code/work-2/paper-3/ExposureMoE/ultralytics-8.3.0')  

from ultralytics import YOLO

def main():
    project_dir = os.path.join(os.path.dirname(__file__),'runs', 'ripe')
    model = YOLO("ultralytics/cfg/models/11/yolo11n-ripe-lightmoe.yaml")
    model.train(
        data="ultralytics/datasets/mixedexposure.yaml",
        epochs=1000,
        device="0",
        project=project_dir,
        name="train_exposuremoe",
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        batch=3,
        # batch=8,
        light_single_aux=True,
        light_corr=0.2,
        light_route=1,
        light_identity=0.5,
        light_smooth=0.3,
        light_balance=0.3,
        light_diverse=0.3,
        lr0=0.0025,
        imgsz=1280,
    )


if __name__ == "__main__":
    main()
