import cv2
from detection import ObjectDetector
from tracking import ObjectTracker
from hazards import HazardDetector
from alerts import AlertSystem

def safety_pipeline(video_source=0):
    """Main vision-based safety pipeline."""
    detector = ObjectDetector()
    tracker = ObjectTracker()
    hazard_detector = HazardDetector()
    alerter = AlertSystem('sender@example.com', 'receiver@example.com')

    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.track(detections, frame)
        hazards = hazard_detector.detect_hazards(tracks)

        for hazard in hazards:
            alerter.send_alert(hazard)

        # Optional: Draw boxes for visualization
        for track in tracks:
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track.track_id}", (int(bbox[0]), int(bbox[1])-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Safety Monitoring', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()