import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import sys
sys.path.append('/media/strawberry/KINGSTON/code/work-2/paper-3/ExposureMoE/ultralytics-8.3.0')  

from ultralytics import YOLO

def main():
    project_dir = os.path.join(os.path.dirname(__file__),'runs', 'ripe')
    model = YOLO("ultralytics/cfg/models/11/yolo11-ripe.yaml")
    model.train(
        data="ultralytics/datasets/mixedexposure.yaml",
        epochs=1000,
        device="0",
        project=project_dir,
        name="uncertainty_gstd",
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        batch=16,
        light_single_aux=False,
        lr0=0.005
    )


if __name__ == "__main__":
    main()
