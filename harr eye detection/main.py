import numpy as np
import cv2

eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
cap = cv2.VideoCapture("eye_recording.flv")

while 1:
    ret, img = cap.read()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(img, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        roi = img[ey: ey + eh, ex: ex + ew]
        rows, cols, _ = roi.shape
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

        _, threshold = cv2.threshold(gray_roi, 3, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)

            cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.line(roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (0, 255, 0), 2)
            cv2.line(roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (0, 255, 0), 2)
            #cv2.line(img, (ex + int(ew / 4), 0), (x + int(3*ew / 4), rows), (0, 255, 0), 2)

            break

        cv2.imshow("Threshold", threshold)

        # cv2.imshow("gray roi", gray_roi)
        cv2.imshow("Roi", roi)
        key = cv2.waitKey(60) & 0xff
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()

