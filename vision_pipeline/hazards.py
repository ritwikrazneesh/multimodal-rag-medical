class HazardDetector:
    """Rule-based hazard detection."""
    def __init__(self, safe_distance=50):  # Pixels or calibrated distance
        self.safe_distance = safe_distance

    def detect_hazards(self, tracks):
        """Check for hazards like proximity between workers and equipment."""
        hazards = []
        workers = [t for t in tracks if t.det_class == 0]  # Workers
        equipment = [t for t in tracks if t.det_class == 2]  # Equipment

        for w in workers:
            w_center = ((w.to_tlbr()[0] + w.to_tlbr()[2]) / 2, (w.to_tlbr()[1] + w.to_tlbr()[3]) / 2)
            for e in equipment:
                e_center = ((e.to_tlbr()[0] + e.to_tlbr()[2]) / 2, (e.to_tlbr()[1] + e.to_tlbr()[3]) / 2)
                dist = ((w_center[0] - e_center[0])**2 + (w_center[1] - e_center[1])**2)**0.5
                if dist < self.safe_distance:
                    hazards.append({
                        'worker_id': w.track_id,
                        'equip_id': e.track_id,
                        'dist': dist
                    })
        return hazards