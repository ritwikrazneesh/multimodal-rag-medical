from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    """DeepSORT-based tracking for detected objects."""
    def __init__(self, max_age=30, nn_budget=100):
        self.tracker = DeepSort(max_age=max_age, nn_budget=nn_budget)

    def track(self, detections, frame):
        """Update tracks with new detections."""
        # Format detections: [(left, top, width, height), conf, class]
        dets = [((d['bbox'][0], d['bbox'][1], d['bbox'][2]-d['bbox'][0], d['bbox'][3]-d['bbox'][1]), 
                 d['conf'], d['class']) for d in detections]
        tracks = self.tracker.update_tracks(dets, frame=frame)
        return tracks