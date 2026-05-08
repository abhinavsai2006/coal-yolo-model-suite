# Final Model (Hybrid EfficientNet-B3)

- Checkpoint: `yolo11_hybridmodel/runs/b3_seed1337/weights/best.pt`
- Test accuracy (no TTA): 91.21%
- Test accuracy (hflip TTA): 91.76%
- Classes (6): destructive_coal, fully_pulverized_coal, non_destructive_coal, not_coal, pulverized_coal, strongly_destructive_coal

## Quick inference
PowerShell:

```
& "E:\Yolo\yolov9_gpu\python.exe" "E:\Yolo\YOLOv11_Coal_Classification_Project\yolo11_hybridmodel\inference\evaluate_ensemble_on_test.py" \
  --weights "E:\Yolo\YOLOv11_Coal_Classification_Project\yolo11_hybridmodel\runs\b3_seed1337\weights\best.pt" \
  --tta hflip --output evaluation_b3_hflip_tta
```

## Notes
- We recommend enabling `--tta hflip` for the small extra gain (~+0.55%).
- The evaluator auto-detects backbone type; no config changes needed.
- All artifacts are saved under `yolo11_hybridmodel/evaluation_b3_*`.

## Optional: ONNX export (for deployment)
A simple export utility can be added on request (export to ONNX opset 12 with dynamic batch).