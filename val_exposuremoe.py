import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
import sys
sys.path.append('/media/strawberry/KINGSTON/code/work-2/paper-3/ExposureMoE/ultralytics-8.3.0')  

from ultralytics import YOLO
from pathlib import Path
def main():
    project_dir = os.path.join(os.path.dirname(__file__), 'runs', 'ripe')
    save_dir = Path("/media/strawberry/KINGSTON/code/work-2/paper-3/ExposureMoE/runs/ripe/val_exposuremoe")
    model = YOLO("runs/ripe/moe_640/weights/best.pt")
    results = model.val(
        name="val_exposuremoe",
        project=project_dir,
        plots=True,
        exist_ok=True,
        verbose=True,
        imgsz=640,
        batch=1,
    )
    print(results.confusion_matrix.matrix)
    results.confusion_matrix.plot(
        save_dir=save_dir,
        normalize=False,
    )
    


if __name__ == "__main__":
    main()
