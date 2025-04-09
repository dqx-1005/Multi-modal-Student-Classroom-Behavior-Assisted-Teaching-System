from ultralytics import YOLO
import os
import cv2
def train_yolov8(data_yaml,model_path):
    epochs = 50
    imgsz = 640
    batch_size = 16

    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size
    )
    print("训练完成！")

def evaluate_yolov8():
    # 评估阶段
    data_yaml = '/path/to/data.yaml'
    best_model_path = '/path/to/best.pt'
    model = YOLO(best_model_path)  # 加载训练好的模型
    results = model.val(data=data_yaml)  # 在验证集上评估模型
    print("评估结果：", results)

def predict_yolov8(source,model_path):
    save_dir = 'runs/detect/predict'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = YOLO(model_path)
    results = model.predict(source)
    print("推理结果：", results)

    for i, result in enumerate(results):
        output_image = result.plot()
        output_path = os.path.join(save_dir, f"predicted_image_{i}.jpg")
        cv2.imwrite(output_path, output_image)
        print(f"预测图片 {i} 已保存到 {output_path}")

if __name__ == "__main__":
    train_yolov8()
    evaluate_yolov8()
    predict_yolov8()
