import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import sys
sys.path.append('/media/strawberry/KINGSTON/code/work-2/paper-3/ExposureMoE/ultralytics-8.3.0')  

from ultralytics import YOLO
from pathlib import Path
def main():
    project_dir = os.path.join(os.path.dirname(__file__), 'runs', 'ripe')
    model = YOLO("runs/ripe/moe_640/weights/best.pt")
    model.predict(
        source ='ultralytics/datasets/MixExposure/images/val',
        name="predict_moe_640_txt",
        project=project_dir,
        imgsz=640,
        save_txt=True,
        save=True,
        batch=1
    )
    

if __name__ == "__main__":
    main()
