import torch
import torch.nn as nn
import cv2
import os
import time
import psutil
from torchvision import transforms
from PIL import Image
from thop import profile as thop_profile
from memory_profiler import profile as memory_profile


class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 16x112x112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x56x56
            
            nn.Conv2d(16, 32, 3, padding=1),  # 32x56x56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x28x28
            
            nn.Conv2d(32, 64, 3, padding=1),  # 64x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x14x14
            
            nn.Conv2d(64, 32, 3, padding=1),  # 32x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x7x7
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze()
    
    @memory_profile
    def memory_analysis(self, input_size=(1,3,224,224)):
        """分层内存消耗分析"""
        mem_usage = {}
        with torch.no_grad():
            x = torch.randn(*input_size)
            for name, layer in self.named_children():
                mem_before = self.memory_monitor.get_memory_usage()
                x = layer(x)
                mem_after = self.memory_monitor.get_memory_usage()
                mem_usage[name] = mem_after - mem_before
        return mem_usage



def load_model():
    """加载预训练模型"""
    model = LightweightCNN()
    checkpoint = torch.load('models/person_classifier_best.pth', 
                           map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image):
    """图像预处理"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def run_inference(model, image):
    """执行推理"""
    with torch.no_grad():
        start_time = time.perf_counter()
        output = model(image)
        return output, (time.perf_counter() - start_time) * 1000  # 返回毫秒

def run_camera_inference():
    """摄像头实时推理"""
    process = psutil.Process(os.getpid())
    model = load_model()

    cap = cv2.VideoCapture(0)
    
    print("=== 模型分析 ===")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")



    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 预处理
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess_image(pil_img)
        
        # 推理
        output, infer_time = run_inference(model, input_tensor)
        prob = output.item()
        is_person = prob > 0.15
        
        
        # 绘制显示信息
        info_line = f"Class: {is_person},Prob: {prob:.2f} | Time: {infer_time:.1f}ms"
        # res_line = f"Mem: {mem_usage:.1f}MB | CPU: {cpu_usage}%"
        cv2.putText(frame, info_line, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        # cv2.putText(frame, res_line, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        
        cv2.imshow('Real-time Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_inference()
