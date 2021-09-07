import cv2
from typing import (
    Dict,
    Any,
    Tuple
)
import numpy as np
import time
import math


def draw_text(img: np.ndarray, text: str, position: Tuple[int, int], color: Tuple[int, int, int]):
    fps_kwargs: Dict[str, Any] = dict(
        img=img,
        text=text,
        org=position,
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=0.7,
        color=color,
        thickness=2
    )
    cv2.putText(**fps_kwargs)


def main() -> None:
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    tracker: cv2.Tracker = cv2.TrackerCSRT_create()
    start = False

    xs: np.ndarray = np.array([])
    ys: np.ndarray = np.array([])
    t: np.ndarray = np.array([])

    start_time = time.time()

    while True:
        timer = cv2.getTickCount()
        success, img = cap.read()

        timestep = time.time() - start_time
        if start:
            success, bbox = tracker.update(img)

            if success:
                x, y, w, h = list(map(int, bbox))
                cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=3, lineType=1)
                y_normalized = y
                draw_text(img, "Tracking", position=(75, 80), color=(0, 255, 0))
                middle_point_x: int = int((x + (w / 2)))
                middle_point_y: int = int(y_normalized + (h/2))

                img = cv2.circle(img, (middle_point_x, middle_point_y), 3, (255, 0, 0), 5)

                t = np.append(t, [timestep])
                xs = np.append(xs, [middle_point_x])
                ys = np.append(ys, [middle_point_y])

                if len(t) > 100:
                    t = t[-50:]
                    xs = t[-50:]
                    ys = ys[-50:]

                if len(ys) > 1:
                    delta_t = (t[-1] - t[-2])
                    velocity_y = int((ys[-1] - ys[-2]) / delta_t)
                    velocity_x = int((xs[-1] - xs[-2]) / delta_t)
                    speed = math.sqrt(velocity_y ** 2 + velocity_x ** 2)

                    starting_point = (middle_point_x, middle_point_y)
                    end_point = (middle_point_x + velocity_x, middle_point_y + velocity_y)

                    img = cv2.arrowedLine(
                        img,
                        pt1=starting_point,
                        pt2=end_point,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                    draw_text(img, text=f"Speed: {speed} cm/s", position=(70, img.shape[0] - 50), color=(0, 0, 255))
            else:
                draw_text(img, "Lost", position=(75, 80), color=(255, 0, 0))

        draw_text(img, f"Time: {int(timestep)} s", position=(70, img.shape[0] - 70), color=(0, 0, 255))
        actual_time = cv2.getTickCount()
        prev_frame_timestep = timer

        fps = cv2.getTickFrequency() / (actual_time - prev_frame_timestep)
        draw_text(img, text=f"FPS: {int(fps)}", position=(75, 50), color=(255, 0, 0))
        cv2.imshow("Tracking", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        if cv2.waitKey(1) & 0xff == ord("r"):
            print(f"Pressed r")
            bbox = cv2.selectROI(
                windowName="Tracking",
                img=img,
                showCrosshair=True,
                fromCenter=False
            )

            tracker.init(
                image=img,
                boundingBox=bbox
            )
            start = True


if __name__ == '__main__':
    main()
