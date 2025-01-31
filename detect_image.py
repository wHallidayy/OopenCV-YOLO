import cv2
import numpy as np
import os
from datetime import datetime

class YOLODetector:
    def __init__(self, model_path='models', confidence_threshold=0.5, nms_threshold=0.4):
        """
        กำหนดค่าเริ่มต้นสำหรับ YOLO Detector
        
        Args:
            model_path (str): path ไปยังโฟลเดอร์ที่เก็บโมเดล
            confidence_threshold (float): ค่าความเชื่อมั่นขั้นต่ำ (0-1)
            nms_threshold (float): ค่า threshold สำหรับ non-maximum suppression
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # โหลดโมเดล
        weights_path = os.path.join(model_path, 'yolov3.weights')
        cfg_path = os.path.join(model_path, 'yolov3.cfg')
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        
        # โหลดชื่อคลาส
        names_path = os.path.join(model_path, 'coco.names')
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # สร้างสีแบบสุ่ม
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # ตั้งค่าให้ใช้ CPU
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def preprocess_image(self, image):
        """
        แปลงภาพเป็นรูปแบบที่เหมาะสมสำหรับ input ของโมเดล
        
        Args:
            image (numpy.ndarray): ภาพ input
            
        Returns:
            tuple: (blob, height, width)
        """
        height, width = image.shape[:2]
        
        # สร้าง blob จากภาพ
        blob = cv2.dnn.blobFromImage(
            image, 
            1/255.0,      # scale factor
            (416, 416),   # size
            swapRB=True,  # swap RED and BLUE channels
            crop=False    # don't crop
        )
        
        return blob, height, width

    def detect_objects(self, image):
        """
        ตรวจจับวัตถุในภาพ
        
        Args:
            image (numpy.ndarray): ภาพที่ต้องการตรวจจับ
            
        Returns:
            list: รายการของวัตถุที่ตรวจพบ [(class_id, confidence, box), ...]
        """
        # เตรียมภาพ
        blob, height, width = self.preprocess_image(image)
        
        # ส่งภาพเข้าโมเดล
        self.net.setInput(blob)
        
        # รันการทำนาย
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        
        # แปลผลลัพธ์
        boxes = []
        confidences = []
        class_ids = []
        
        # วนลูปผ่านแต่ละ output layer
        for output in outputs:
            # วนลูปผ่านแต่ละการตรวจจับ
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # แปลงพิกัดเป็นพิกเซล
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # หาพิกัดมุมซ้ายบน
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # ใช้ non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append((
                    class_ids[i],
                    confidences[i],
                    boxes[i]
                ))
        
        return results

    def draw_predictions(self, image, results):
        """
        วาดผลการตรวจจับลงบนภาพ
        
        Args:
            image (numpy.ndarray): ภาพต้นฉบับ
            results (list): ผลลัพธ์จาก detect_objects()
            
        Returns:
            numpy.ndarray: ภาพที่วาดผลการตรวจจับแล้ว
        """
        output_image = image.copy()
        
        for class_id, confidence, box in results:
            x, y, w, h = box
            color = self.colors[class_id]
            
            # วาดกล่อง
            cv2.rectangle(
                output_image, 
                (x, y), 
                (x + w, y + h), 
                color, 
                2
            )
            
            # เตรียมข้อความ
            label = f'{self.classes[class_id]}: {confidence:.2f}'
            
            # คำนวณขนาดข้อความ
            (label_width, label_height), baseline = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                2
            )
            
            # วาดพื้นหลังสำหรับข้อความ
            cv2.rectangle(
                output_image, 
                (x, y - label_height - 10), 
                (x + label_width, y), 
                color, 
                -1
            )
            
            # เขียนข้อความ
            cv2.putText(
                output_image, 
                label, 
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
        
        return output_image

def process_image(input_path, output_path):
    """
    ประมวลผลภาพและบันทึกผลลัพธ์
    
    Args:
        input_path (str): path ไปยังภาพ input
        output_path (str): path สำหรับบันทึกภาพผลลัพธ์
    """
    # สร้าง detector
    detector = YOLODetector()
    
    # อ่านภาพ
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"ไม่สามารถอ่านภาพจาก {input_path}")
    
    # ตรวจจับวัตถุ
    results = detector.detect_objects(image)
    
    # วาดผลการตรวจจับ
    output_image = detector.draw_predictions(image, results)
    
    # บันทึกผลลัพธ์
    cv2.imwrite(output_path, output_image)
    
    # สร้าง summary
    summary = {
        'total_objects': len(results),
        'objects': {}
    }
    
    for class_id, confidence, _ in results:
        class_name = detector.classes[class_id]
        if class_name not in summary['objects']:
            summary['objects'][class_name] = 0
        summary['objects'][class_name] += 1
    
    return summary

def main():
    try:
        # กำหนด paths
        input_dir = 'images/input'
        output_dir = 'images/output'
        
        # เช็คว่าโฟลเดอร์มีอยู่จริง
        if not os.path.exists(input_dir):
            print(f"Error: Input directory '{input_dir}' not found!")
            return
            
        # เช็คว่ามีไฟล์ในโฟลเดอร์
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print(f"Error: No image files found in '{input_dir}'")
            return
            
        print(f"Found {len(files)} image(s): {files}")
        
        # ประมวลผลแต่ละไฟล์
        for filename in files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f'result_{filename}')
            
            print(f"\nProcessing {filename}...")
            try:
                # โหลดภาพ
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Error: Could not load image '{input_path}'")
                    continue
                    
                print(f"Image loaded: {image.shape}")
                
                # ตรวจจับวัตถุ
                detector = YOLODetector()
                results = detector.detect_objects(image)
                
                print(f"Detection complete: found {len(results)} objects")
                
                # วาดผลและบันทึก
                output_image = detector.draw_predictions(image, results)
                cv2.imwrite(output_path, output_image)
                
                print(f"Result saved to {output_path}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                
    except Exception as e:
        print(f"Program error: {str(e)}")

if __name__ == '__main__':
    main()