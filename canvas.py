"""
Iron Canvas Pro — Python 3.13 + mediapipe 0.10.x compatible
Uses the NEW mediapipe Tasks API (HandLandmarker) — zero mp.solutions usage.
"""

import cv2
import numpy as np
import math
import time
import threading
import platform
import colorsys
import urllib.request
import os

# ── mediapipe Tasks API (works on 0.10.x / Python 3.13) ─────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Optional Windows sound ───────────────────────────────────────────────────
AUDIO_AVAILABLE = False
PLATFORM = platform.system()
if PLATFORM == "Windows":
    try:
        import winsound
        AUDIO_AVAILABLE = True
    except ImportError:
        pass


# ── Hand model auto-download ─────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading hand model (~3 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Done.")
    else:
        print(f"[INFO] Model found: {MODEL_PATH}")


# ── Config ───────────────────────────────────────────────────────────────────
class Config:
    WIDTH, HEIGHT    = 1280, 720
    PINCH_THRESHOLD  = 40
    SMOOTHING        = 0.55
    BRUSH_SIZE       = 8
    ARC_CENTER       = (640, 0)
    ARC_RADIUS       = 150
    ARC_THICKNESS    = 60
    BRUSH_MODES      = ["pen", "spray", "stamp", "eraser"]
    DEFAULT_MODE     = 0
    SHOW_FPS         = True
    SHOW_INSTRUCTIONS = True


# ── Sound engine ─────────────────────────────────────────────────────────────
class SoundEngine:
    def __init__(self):
        self.active     = False
        self.velocity   = 0
        self.stop_event = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def set_drawing(self, is_drawing, velocity):
        self.active   = is_drawing
        self.velocity = velocity

    def _loop(self):
        while not self.stop_event.is_set():
            if AUDIO_AVAILABLE and self.active and PLATFORM == "Windows":
                try:
                    freq = max(100, min(800, int(200 + self.velocity * 5)))
                    winsound.Beep(freq, 40)
                except Exception:
                    pass
            else:
                time.sleep(0.05)


# ── Particle FX ──────────────────────────────────────────────────────────────
class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, x, y, color, count=5):
        for _ in range(count):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(1, 4)
            life  = np.random.randint(10, 25)
            size  = np.random.randint(2, 5)
            self.particles.append([
                float(x), float(y),
                math.cos(angle) * speed, math.sin(angle) * speed,
                life, life, color, size
            ])

    def update_and_draw(self, img):
        alive = []
        for p in self.particles:
            x, y, vx, vy, life, max_life, color, size = p
            if life <= 0:
                continue
            alpha = life / max_life
            draw_color = (int(color[0]*alpha), int(color[1]*alpha), int(color[2]*alpha))
            px, py = int(x), int(y)
            if 0 <= px < Config.WIDTH and 0 <= py < Config.HEIGHT:
                cv2.circle(img, (px, py), size, draw_color, -1)
            p[0] += vx;  p[1] += vy
            p[2] *= 0.92; p[3] *= 0.92
            p[4] -= 1
            alive.append(p)
        self.particles = alive


# ── Hand tracking — Tasks API only ───────────────────────────────────────────
class HandSystem:
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
    ]

    def __init__(self, model_path: str):
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(opts)

    def process(self, bgr_frame):
        """Returns list of 21 (x,y) pixel coords, or None."""
        h, w = bgr_frame.shape[:2]
        rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)
        if not result.hand_landmarks:
            return None
        lms = result.hand_landmarks[0]
        return [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    def draw_sci_fi_hud(self, img, points, pinch_dist):
        if not points:
            return img
        overlay = img.copy()

        hue = (time.time() * 60) % 360
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue / 360, 1, 1)]
        skel_color = (b, g, r)

        for p1, p2 in self.CONNECTIONS:
            cv2.line(overlay, points[p1], points[p2], skel_color, 1, cv2.LINE_AA)
        for pt in points:
            cv2.circle(overlay, pt, 3, (0, 165, 255), -1)
            cv2.circle(overlay, pt, 6, skel_color, 1)

        ix, iy = points[8]
        cs = 12
        cv2.line(overlay, (ix-cs, iy), (ix+cs, iy), (255,255,255), 1, cv2.LINE_AA)
        cv2.line(overlay, (ix, iy-cs), (ix, iy+cs), (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (ix, iy), 14, skel_color, 1, cv2.LINE_AA)

        bar_len  = 40
        fill     = max(0.0, min(1.0, (100 - pinch_dist) / 60))
        bar_color = (0,255,0) if pinch_dist < Config.PINCH_THRESHOLD else (0,0,255)
        if pinch_dist < Config.PINCH_THRESHOLD:
            cv2.putText(overlay, "DRAW", (ix+20, iy-10),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, bar_color, 2)
        cv2.rectangle(overlay, (ix+15, iy), (ix+55, iy+6), (50,50,50), -1)
        cv2.rectangle(overlay, (ix+15, iy),
                      (ix+15+int(bar_len*fill), iy+6), bar_color, -1)

        return cv2.addWeighted(overlay, 0.75, img, 0.25, 0)


# ── Arc colour palette ────────────────────────────────────────────────────────
class ArcPalette:
    def __init__(self):
        self.colors = [
            ((0,   0, 255), "RED"),
            ((0, 100, 255), "ORANGE"),
            ((0, 255, 255), "YELLOW"),
            ((0, 255,   0), "GREEN"),
            ((255,255,  0), "CYAN"),
            ((255,  0,180), "PINK"),
            ((255,  0,255), "PURPLE"),
            ((255,255,255), "WHITE"),
            ((0,   0,   0), "CLEAR"),
        ]
        self.selected_index = 4

    def draw(self, img, hover_pt):
        n      = len(self.colors)
        sector = 180 / n
        cx, cy = Config.ARC_CENTER
        r      = Config.ARC_RADIUS
        hover_index = -1

        if hover_pt:
            hx, hy = hover_pt
            dist = math.hypot(hx - cx, hy - cy)
            if r < dist < r + Config.ARC_THICKNESS:
                angle = math.degrees(math.atan2(hy - cy, hx - cx))
                if angle < 0:
                    angle += 360
                if 0 <= angle <= 180:
                    hover_index = min(int(angle / sector), n - 1)

        for i, (color, name) in enumerate(self.colors):
            sa    = i * sector
            ea    = (i + 1) * sector
            thick = Config.ARC_THICKNESS
            shift = 0
            if i == self.selected_index:
                shift = 15
                cv2.ellipse(img, (cx, cy), (r+shift, r+shift),
                            0, sa, ea, (255,255,255), -1)
            if i == hover_index:
                thick += 10
                cv2.putText(img, name, (cx-50, cy+r+80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.ellipse(img, (cx, cy),
                        (r+shift+thick//2, r+shift+thick//2),
                        0, sa, ea, color, thick)
        return hover_index


# ── Brush mode side-panel ─────────────────────────────────────────────────────
class BrushModePanel:
    ICONS  = {"pen": "PEN", "spray": "SPRAY", "stamp": "STAR", "eraser": "ERASE"}
    PX     = 10
    BTN_H  = 50
    BTN_W  = 110

    def draw(self, img, cur):
        for i, mode in enumerate(Config.BRUSH_MODES):
            y = 200 + i * (self.BTN_H + 10)
            active = (i == cur)
            cv2.rectangle(img, (self.PX, y),
                          (self.PX+self.BTN_W, y+self.BTN_H), (30,30,30), -1)
            cv2.rectangle(img, (self.PX, y),
                          (self.PX+self.BTN_W, y+self.BTN_H),
                          (0,255,255) if active else (80,80,80), 2)
            cv2.putText(img, self.ICONS[mode], (self.PX+6, y+32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,255,255) if active else (180,180,180), 1, cv2.LINE_AA)

    def check_hover(self, pt, dist):
        if not pt or dist >= Config.PINCH_THRESHOLD:
            return -1
        hx, hy = pt
        for i in range(len(Config.BRUSH_MODES)):
            y = 200 + i * (self.BTN_H + 10)
            if self.PX <= hx <= self.PX+self.BTN_W and y <= hy <= y+self.BTN_H:
                return i
        return -1


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_spray(canvas, x, y, color, size=22, density=35):
    for _ in range(density):
        a  = np.random.uniform(0, 2*math.pi)
        rv = np.random.uniform(0, size)
        px, py = int(x + rv*math.cos(a)), int(y + rv*math.sin(a))
        if 0 <= px < Config.WIDTH and 0 <= py < Config.HEIGHT:
            cv2.circle(canvas, (px, py), 1, color, -1)

def draw_star(canvas, cx, cy, color, size=18):
    pts = []
    for i in range(10):
        a  = math.radians(i*36 - 90)
        rv = size if i % 2 == 0 else size // 2
        pts.append((int(cx + rv*math.cos(a)), int(cy + rv*math.sin(a))))
    cv2.fillPoly(canvas, [np.array(pts, np.int32)], color)


# ── FPS counter ───────────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self):
        self.prev = time.time()
        self.fps  = 0.0

    def tick(self):
        now      = time.time()
        self.fps = 1.0 / max(now - self.prev, 1e-6)
        self.prev = now

    def draw(self, img):
        cv2.putText(img, f"FPS: {int(self.fps)}",
                    (Config.WIDTH - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)


# ── Instructions bar ──────────────────────────────────────────────────────────
def draw_instructions(img, mode, color_name):
    for i, line in enumerate([
        f"Mode: {mode.upper()}  |  Color: {color_name}",
        "Pinch = Draw/Select    Q = Quit",
        "Left panel = Brush mode",
    ]):
        cv2.putText(img, line, (10, Config.HEIGHT - 70 + i*22),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (200,200,200), 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    ensure_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.HEIGHT)

    hand_sys    = HandSystem(MODEL_PATH)
    palette     = ArcPalette()
    sound       = SoundEngine()
    particles   = ParticleSystem()
    brush_panel = BrushModePanel()
    fps_counter = FPSCounter()

    canvas = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)

    smooth_x = smooth_y    = 0
    current_color          = (255, 255, 0)
    current_color_name     = "CYAN"
    current_mode_idx       = Config.DEFAULT_MODE
    prev_draw_pos          = None

    print("=" * 42)
    print("  IRON CANVAS PRO  —  Python 3.13 Ready")
    print(f"  Platform : {PLATFORM}")
    print(f"  Audio    : {'ON' if AUDIO_AVAILABLE else 'OFF (Windows only)'}")
    print("  Press Q to quit")
    print("=" * 42)

    while True:
        ok, img = cap.read()
        if not ok:
            break

        img    = cv2.flip(img, 1)
        points = hand_sys.process(img)

        is_drawing = False
        velocity   = 0

        if points:
            idx_tip = points[8]
            thm_tip = points[4]
            cx, cy  = idx_tip

            if smooth_x == 0:
                smooth_x, smooth_y = cx, cy
            smooth_x = int(smooth_x*(1-Config.SMOOTHING) + cx*Config.SMOOTHING)
            smooth_y = int(smooth_y*(1-Config.SMOOTHING) + cy*Config.SMOOTHING)

            dist = math.hypot(idx_tip[0]-thm_tip[0], idx_tip[1]-thm_tip[1])

            img       = hand_sys.draw_sci_fi_hud(img, points, dist)
            new_mode  = brush_panel.check_hover((smooth_x, smooth_y), dist)
            if new_mode != -1:
                current_mode_idx = new_mode

            hover_idx = palette.draw(img, (smooth_x, smooth_y))

            if hover_idx != -1 and dist < Config.PINCH_THRESHOLD:
                color, name = palette.colors[hover_idx]
                if name == "CLEAR":
                    canvas[:] = 0
                    particles.particles.clear()
                else:
                    palette.selected_index = hover_idx
                    current_color          = color
                    current_color_name     = name

            elif (dist < Config.PINCH_THRESHOLD
                  and smooth_y > 180
                  and smooth_x > BrushModePanel.PX + BrushModePanel.BTN_W + 10):

                is_drawing = True
                mode       = Config.BRUSH_MODES[current_mode_idx]
                velocity   = math.hypot(smooth_x - cx, smooth_y - cy)

                if mode == "pen":
                    if prev_draw_pos:
                        cv2.line(canvas, prev_draw_pos, (smooth_x, smooth_y),
                                 current_color, Config.BRUSH_SIZE)
                    cv2.circle(canvas, (smooth_x, smooth_y),
                               Config.BRUSH_SIZE//2, current_color, -1)
                elif mode == "spray":
                    draw_spray(canvas, smooth_x, smooth_y, current_color)
                elif mode == "stamp":
                    draw_star(canvas, smooth_x, smooth_y, current_color,
                              size=Config.BRUSH_SIZE+10)
                elif mode == "eraser":
                    cv2.circle(canvas, (smooth_x, smooth_y), 25, (0,0,0), -1)

                if mode != "eraser" and np.random.rand() < 0.4:
                    particles.emit(smooth_x, smooth_y, current_color, count=4)

                prev_draw_pos = (smooth_x, smooth_y)
            else:
                prev_draw_pos = None

            smooth_x, smooth_y = cx, cy

        else:
            palette.draw(img, None)
            prev_draw_pos = None

        sound.set_drawing(is_drawing, velocity)
        particles.update_and_draw(img)

        # Neon glow
        small   = cv2.resize(canvas, (0,0), fx=0.2, fy=0.2)
        blur    = cv2.GaussianBlur(small, (15,15), 0)
        blur_up = cv2.resize(blur, (Config.WIDTH, Config.HEIGHT))
        glow    = cv2.addWeighted(canvas, 1.0, blur_up, 1.5, 0)

        # Merge onto camera
        gray    = cv2.cvtColor(glow, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        img_bg  = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
        img     = cv2.add(img_bg, glow)

        brush_panel.draw(img, current_mode_idx)
        if Config.SHOW_FPS:
            fps_counter.tick()
            fps_counter.draw(img)
        if Config.SHOW_INSTRUCTIONS:
            draw_instructions(img, Config.BRUSH_MODES[current_mode_idx], current_color_name)

        cv2.imshow("Iron Canvas Pro", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sound.stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    print("Iron Canvas closed.")


if __name__ == "__main__":
    main()