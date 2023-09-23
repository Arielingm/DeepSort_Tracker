from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time
from ultralytics import YOLO

if __name__ == '__main__':
    # Captura
    cap = cv2.VideoCapture('people.mp4')  # 'people.mp4'

    model = YOLO("yolov8n.pt")  # Cargar el Modelo

    tracker = DeepSort(max_age=15)  # Cargar el objeto_tracker de la librería DeepSort / Ajustar max_age
    
    while cap.isOpened():
        status, frame = cap.read()  # Leer el Frame
        start = time.perf_counter()
        
        if not status:  # Comprobación del Frame
            break
        
        results = model(frame, verbose=False)  # Resultados de YOLO por frame
        
        # Convierte los resultados de YOLO a un formato adecuado para DeepSort
        bbs = []
        for det in results[0].boxes:
            bbox = det.xyxy.numpy()[0].tolist()
            confidence = float(det.conf.item())
            detection_class = int(det.cls.item())
            
            if detection_class == 0 and confidence > 0.65:
                bbs.append((bbox, confidence, detection_class))
        
        # Actualiza el rastreador DeepSort con los cuadros delimitadores (boxes)
        tracks = tracker.update_tracks(bbs, frame=frame)
        
        # Basicamente aquí extraemos todos los cuadros delimitadores para crear los(boxes) en el frame
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltwh()  # Use to_ltwh() to get [x1, y1, x2, y2]
            bbox = ltrb
            
            cv2.putText(img=frame, text=f"Id: {track_id}", org=(int(bbox[0]), int(bbox[1]) - 10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
            
            cv2.rectangle(img=frame, pt1=(int(bbox[0]), int(bbox[1])), pt2=(int(bbox[2]), int(bbox[3])),
                          color=(0, 255, 0), thickness=2)
        
        end = time.perf_counter()  # tiempo de end
        fps = int(1 / (end - start))
        cv2.putText(img=frame, text=f"FPS: {fps}", org=(20, 70), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5,
                    color=(0, 0, 255), thickness=2)
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
