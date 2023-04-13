"""
    Programmer: Emmanuel Samuel
    Date Created: 12th April 2023
    Date Revised: 13th April 2023
    Purpose: Traffic monitoring system using computer vision
"""

# import modules
import numpy as np
import cv2
import sys
import time
import validator
from random import randint

line_in_color = (64, 255, 0)
line_out_color = (0, 0, 255)
bounding_box_color = (255, 128, 0)
tracker_color = (randint(0, 255), randint(0, 255), randint(0, 255))
centroid_color = (randint(0, 255), randint(0, 255), randint(0, 255))
text_color = (randint(0, 255), randint(0, 255), randint(0, 255))
text_position_bgs = (10, 50)
text_position_count_cars = (10, 100)
text_position_count_trucks = (10, 150)
text_size = 1.2
font_type = cv2.FONT_HERSHEY_SIMPLEX
save_image = True
image_dir = "./vehicles"
video_source = "cars_people.mp4"
output_video = "result_traffic.avi"

# define list to hold the algorithms
BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
# select an algorithm
bgs_type = BGS_TYPES[2]


# function to get kernel type
def get_kernel(kernel_type):
    if kernel_type == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if kernel_type == "opening":
        kernel = np.ones((3, 3), np.uint8)
    if kernel_type == "closing":
        kernel = np.ones((11, 11), np.uint8)

    return kernel


# function to apply filter to image
def get_filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel("closing"), iterations=2)

    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel("opening"), iterations=2)

    if filter == 'dilation':
        return cv2.dilate(img, get_kernel("dilation"), iterations=2)

    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, get_kernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, get_kernel("dilation"), iterations=2)

        return dilation


# function to get background subtractor algorithm
def get_bgsubtractor(bgs_type):
    if bgs_type == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=.8)
    if bgs_type == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=.7, noiseSigma=0)
    if bgs_type == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=False, varThreshold=200)
    if bgs_type == "KNN":
        return cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=400, detectShadows=True)
    if bgs_type == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True,
                                                        maxPixelStability=15 * 60, isParallel=True)
    print("Invalid detector")
    sys.exit(1)


# function to get the central position of the object
def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# function to save images of the car
def save_frame(frame, file_name, flip=True):
    if flip:
        # BGR -> RGB
        cv2.imwrite(file_name, np.flip(frame, 2))
    else:
        cv2.imwrite(file_name, frame)


cap = cv2.VideoCapture(video_source)
success, frame = cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer_video = cv2.VideoWriter(output_video, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

# Region of Interest(ROI) selector
bbox = cv2.selectROI(frame, False)
# print(bbox)
(w1, h1, w2, h2) = bbox
# print(w1, h1, w2, h2)

frame_area = h2 * w2
# print(frameArea)
min_area = int(frame_area / 250)
# print(minArea)
max_area = 15000

line_IN = int(h1)
line_OUT = int(h2 - 20)
# print(line_IN, line_OUT)

down_limit = int(h1 / 4)
print('Down IN limit Y', str(down_limit))
print('Down OUT limit Y', str(line_OUT))

# call the algorithm function
bg_subtractor = get_bgsubtractor(bgs_type)


# main function definition
def main():
    frame_number = -1
    cnt_cars, cnt_trucks = 0, 0
    objects = []
    max_p_age = 2
    pid = 1
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Finished processing the video")
            break

        # define
        roi = frame[h1:h1 + h2, w1:w1 + w2]

        for i in objects:
            i.age_one()

        frame_number += 1
        bg_mask = bg_subtractor.apply(roi)
        bg_mask = get_filter(bg_mask, 'combine')
        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Counting the cars
            if area > min_area and area <= max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                centroid = get_centroid(x, y, w, h)
                cx = centroid[0]
                cy = centroid[1]
                new = True
                cv2.rectangle(roi, (x, y), (x + 50, y - 13), tracker_color, -1)
                cv2.putText(roi, 'Cars', (x, y - 2), font_type, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                for i in objects:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_DOWN(down_limit) == True:
                            cnt_cars += 1
                            if save_image:
                                save_frame(roi, image_dir + '/car_DOWN_%04d.png' % frame_number)
                                print('ID:', i.getId(), ' passed by the road in', time.strftime('%c'))
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > line_OUT:
                            i.setDone()
                    if i.timedOut():
                        index = objects.index(i)
                        objects.pop(index)
                        del i
                if new == True:
                    p = validator.MyValidator(pid, cx, cy, max_p_age)
                    objects.append(p)
                    pid += 1
                cv2.circle(roi, (cx, cy), 5, centroid_color, -1)

            # Counting the trucks
            elif area >= max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                centroid = get_centroid(x, y, w, h)
                cx = centroid[0]
                cy = centroid[1]

                new = True

                cv2.rectangle(roi, (x, y), (x + 50, y - 13), tracker_color, -1)
                cv2.putText(roi, 'TRUCK', (x, y - 2), font_type, .5, (255, 255, 255), 1, cv2.LINE_AA)

                for i in objects:
                    if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                        new = False
                        i.updateCoords(cx, cy)

                        if i.going_DOWN(down_limit) == True:
                            cnt_trucks += 1
                            if save_image:
                                save_frame(roi, image_dir + "/truck_DOWN_%04d.png" % frame_number, flip=False)
                                print("ID:", i.getId(), 'passed by the road', time.strftime("%c"))
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > line_OUT:
                            i.setDone()
                    if i.timedOut():
                        index = objects.index(i)
                        objects.pop(index)
                        del i
                if new == True:
                    p = validator.MyValidator(pid, cx, cy, max_p_age)
                    objects.append(p)
                    pid += 1
                cv2.circle(roi, (cx, cy), 5, centroid_color, -1)

        for i in objects:
            cv2.putText(roi, str(i.getId()), (i.getX(), i.getY()), font_type, 0.3, text_color, 1, cv2.LINE_AA)

        str_cars = 'Cars: ' + str(cnt_cars)
        str_trucks = 'Trucks: ' + str(cnt_trucks)

        frame = cv2.line(frame, (w1, line_IN), (w1 + w2, line_IN), line_in_color, 2)
        frame = cv2.line(frame, (w1, h1 + line_OUT), (w1 + w2, h1 + line_OUT), line_out_color, 2)

        cv2.putText(frame, str_cars, text_position_count_cars, font_type, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, str_cars, text_position_count_cars, font_type, 1, (232, 162, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, str_trucks, text_position_count_trucks, font_type, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, str_trucks, text_position_count_trucks, font_type, 1, (232, 162, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, 'Background Subtractor: ' + bgs_type, text_position_bgs, font_type, text_size,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Background Subtractor: ' + bgs_type, text_position_bgs, font_type, text_size, (128, 0, 255),
                    2,
                    cv2.LINE_AA)

        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', bg_mask)

        writer_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
