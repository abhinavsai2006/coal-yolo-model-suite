import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_coal_classifier import train_coal_classification_model

if __name__ == '__main__':
    # Train yolov8m on balanced data YAML
    dataset_yaml = str(Path(__file__).resolve().parents[1] / 'data' / 'balanced_data.yaml')
    print(f"Starting full training using dataset: {dataset_yaml}")

    # You can tweak epochs and batch size here
    best_model, results = train_coal_classification_model(
        dataset_path=dataset_yaml,
        model_size='yolov8m-cls.pt',
        epochs=20,
        batch_size=16,
        project_name='coal_classification_runs',
        experiment_name='coal_yolov8_m_balanced'
    )
    print(f"Training finished. Best model at: {best_model}")
