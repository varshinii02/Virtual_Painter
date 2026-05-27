import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Load and resize color icons
def load_and_resize_image(path, size=(100, 100)):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return cv2.resize(img, size)

# Define paths for the new color icons
red_color = load_and_resize_image("red.png")
green_color = load_and_resize_image("lightgreen.png")
yellow_color = load_and_resize_image("yellow.png")
purple_color = load_and_resize_image("pink.jpg")
cyan_color = load_and_resize_image("blue.png")
eraser = load_and_resize_image("white.png")

detector = HandDetector(detectionCon=0.8, maxHands=1)
img_canvas = np.zeros((720, 1280, 3), np.uint8)
xp, yp = 0, 0
draw_color = (0, 0, 0)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    # Draw color selection boxes
    img[80:180, 410:510] = red_color
    img[80:180, 550:650] = green_color
    img[80:180, 690:790] = yellow_color
    img[80:180, 830:930] = purple_color
    img[80:180, 970:1070] = cyan_color
    img[80:180, 1110:1210] = eraser

    if hands:
        hand = hands[0]
        lm_list = hand["lmList"]
        fingers = detector.fingersUp(hand)
        cx, cy = lm_list[8][0], lm_list[8][1]

        # Color Selection Mode
        if fingers[1] and fingers[2]:  # If both index and middle fingers are up
            xp, yp = 0, 0
            if 410 < cx < 510 and 80 < cy < 180:
                draw_color = (0, 0, 255)  # Red
            elif 550 < cx < 650 and 80 < cy < 180:
                draw_color = (0, 255, 0)  # Green
            elif 690 < cx < 790 and 80 < cy < 180:
                draw_color = (0, 255, 255)  # Yellow
            elif 830 < cx < 930 and 80 < cy < 180:
                draw_color = (255, 0, 255)  # Purple
            elif 970 < cx < 1070 and 80 < cy < 180:
                draw_color = (255, 255, 0)  # Cyan
            elif 1110 < cx < 1210 and 80 < cy < 180:
                draw_color = (0, 0, 0)  # Eraser

        # Drawing Mode
        if fingers[1] and not fingers[2]:  # If only index finger is up
            cv2.circle(img, (cx, cy), 15, draw_color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = cx, cy

            if draw_color == (0, 0, 0):  # Eraser
                cv2.line(img, (xp, yp), (cx, cy), draw_color, 50)
                cv2.line(img_canvas, (xp, yp), (cx, cy), draw_color, 50)
            else:
                cv2.line(img, (xp, yp), (cx, cy), draw_color, 10)
                cv2.line(img_canvas, (xp, yp), (cx, cy), draw_color, 10)
            xp, yp = cx, cy

        else:
            xp, yp = 0, 0

    # Combine the canvas and the image
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # Display the result
    cv2.imshow("Virtual Painter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
