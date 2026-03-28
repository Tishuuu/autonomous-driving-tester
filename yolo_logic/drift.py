import cv2
import numpy as np

# ==========================================
# 1. מנוע ההתרעות (הממוצע הנע שביקשת!)
# ==========================================
class DriftDetector:
    def __init__(self, history_size=30, drift_threshold_percent=0.10):
        self.center_history = []
        self.history_size = history_size
        self.drift_threshold_percent = drift_threshold_percent 

    def update_and_check(self, current_center_x, frame_width):
        self.center_history.append(current_center_x)
        if len(self.center_history) > self.history_size:
            self.center_history.pop(0)

        # מחשבים את הממוצע הנע (האמצע היציב)
        average_center_x = int(np.mean(self.center_history))

        # חישוב הסטייה
        deviation = abs(current_center_x - average_center_x)
        deviation_percent = deviation / frame_width
        is_drifting = deviation_percent > self.drift_threshold_percent

        return average_center_x, deviation_percent, is_drifting

# ==========================================
# 2. מנוע הזרימה האופטית (Optical Flow)
# ==========================================
class OpticalFlowVanishingPoint:
    def __init__(self):
        self.prev_gray = None
        # הגדרות לאלגוריתם המעקב Lucas-Kanade
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=50, qualityLevel=0.3, minDistance=10, blockSize=7)

    def get_vanishing_point(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        if self.prev_gray is None:
            self.prev_gray = gray
            return width // 2, [] # מחזיר את האמצע כברירת מחדל בפריים הראשון

        # אנחנו מחפשים פיקסלים לזיהוי רק בשליש התחתון של המסך (הכביש)
        mask = np.zeros_like(gray)
        mask[int(height * 0.6):, :] = 255

        # מוצאים פיקסלים "מעניינים" בכביש
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)

        vp_x = width // 2
        flow_lines = []

        if p0 is not None:
            # בודקים לאן הפיקסלים האלו זזו בפריים החדש
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
            
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            A = []
            b = []
            
            # עוברים על כל הנקודות שזזו ומייצרים מהן משוואות של קווים ישרים
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, c = new.ravel()
                d, f = old.ravel()
                dx = a - d
                dy = c - f

                # מסננים תזוזות מזעריות או נקודות שעפות למעלה בטעות
                if np.sqrt(dx**2 + dy**2) < 1.0 or dy > 0:
                    continue
                
                # שומרים את הקווים לציור כדי שתראה את זה עובד
                flow_lines.append((int(a), int(c), int(d), int(f)))

                # בניית המטריצה למציאת נקודת החיתוך (Least Squares)
                A.append([dy, -dx])
                b.append([dy * d - dx * f])

            A = np.array(A)
            b = np.array(b)

            # אם מצאנו מספיק קווים טובים, נחשב את נקודת המפגש שלהם!
            if len(A) >= 2:
                res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                calculated_x = int(res[0][0])
                
                # בדיקת שפיות: הנקודה חייבת להיות בתוך גבולות המסך
                if 0 < calculated_x < width:
                    vp_x = calculated_x

        self.prev_gray = gray
        return vp_x, flow_lines

# ==========================================
# 3. מנהל ההרצה (טסט לסרטון)
# ==========================================
def run_drift_test(video_path, output_path):
    print("Starting Optical Flow Drift Detection... 🚗")
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    
    of_tracker = OpticalFlowVanishingPoint()
    drift_detector = DriftDetector(history_size=15, drift_threshold_percent=0.12)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. מציאת נקודת המגוז בעזרת הזרימה האופטית
        current_vp_x, flow_lines = of_tracker.get_vanishing_point(frame)

        # 2. בדיקת הסטייה מול הממוצע
        avg_x, deviation_pct, is_drifting = drift_detector.update_and_check(current_vp_x, width)

        # 3. ציור הגרפיקה המטורפת על הפריים!
        
        # ציור וקטורי התנועה (החצים של הכביש) - שתראה את האלגוריתם "חושב"
        for (x_new, y_new, x_old, y_old) in flow_lines:
            cv2.line(frame, (x_new, y_new), (x_old, y_old), (0, 255, 0), 2)
            cv2.circle(frame, (x_new, y_new), 3, (0, 200, 0), -1)

        horizon_y = int(height * 0.45) # גובה האופק המשוער

        # כוונת אדומה: ה"אמצע" הנוכחי שהאלגוריתם מזהה כרגע
        cv2.drawMarker(frame, (current_vp_x, horizon_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
        
        # נקודה כחולה/ירוקה: הממוצע הנע שלנו! (היציבות)
        cv2.circle(frame, (avg_x, horizon_y), 8, (255, 255, 0), -1)

        # התרעות וטקסט
        color = (0, 0, 255) if is_drifting else (0, 255, 0)
        cv2.putText(frame, f"Dev: {deviation_pct*100:.1f}%", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        if is_drifting:
            cv2.rectangle(frame, (0,0), (width, height), (0, 0, 255), 10) # מסגרת אדומה קריטית
            cv2.putText(frame, "!!! LANE DEPARTURE !!!", (width//2 - 250, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 4)

        out.write(frame)

    cap.release()
    out.release()
    print("Test Complete! Check the output video.")

if __name__ == '__main__':
    run_drift_test('input_video(5).mp4', 'drift_output.avi')