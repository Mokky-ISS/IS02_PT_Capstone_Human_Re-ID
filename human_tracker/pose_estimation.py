import cv2
import os
import numpy as np
from database import get_patch_np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = ["./database/img/2_3SBK_20210605T230326_150.jpg"]


def check_pose_manual():
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            base = os.path.basename(file)
            img_name = os.path.splitext(base)[0]
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                print("no landmark!")
                continue
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[0].x * image_width}, '
                f'{results.pose_landmarks.landmark[0].y * image_height})'
            )
            print("landmarks:", results.pose_landmarks.landmark)
            print("len landmark:", len(results.pose_landmarks.landmark))
            # Draw pose landmarks on the image.
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite('./database/img/tmp/' + str(img_name) + '.jpg', annotated_image)


def cal_angle(pt1, pt2, pt3):
    # pt1: (x1,y1), pt2: (x2,y2), pt3: (x3,y3)
    # pt_2 is the base point of the angle
    a = np.array(pt1)
    b = np.array(pt2)
    c = np.array(pt3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


draw_skeleton = True
test_image_from_db = False
img_count = 0


def check_pose(img_patch, id):
    b_pose = False
    global img_count
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.9) as pose:
        #img_patch = cv2.imread("./database/img/3_U5ZE_20210607T131744_210.jpg")
        if test_image_from_db:
            img_patch, img_id = get_patch_np(3)
            print("img_id:", img_id)
        image_height, image_width, _ = img_patch.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print("no landmark!")
            return b_pose
        # In order to get the full body view, check for head (0-10), ankles (27-28), and 4 points of the body (11,12,23,24).
        # Also check for the left and right hip angles to make sure the person is in standing mode.
        # Note: Does not need to check for shoulders as we only need the full height images.
        # https://google.github.io/mediapipe/solutions/pose.html#static_image_mode
        head = list(range(0, 7))
        ankles = list(range(27, 29))
        body = [11, 12, 23, 24]
        b_head = True
        b_ankle = True
        b_body = True
        b_hip = False
        for h in head:
            if results.pose_landmarks.landmark[h].visibility < 0.7:
                b_head = False
                break
        for a in ankles:
            if results.pose_landmarks.landmark[a].visibility < 0.7:
                b_ankle = False
                break
        for b in body:
            # if the landmark is out of range in x-axis
            if results.pose_landmarks.landmark[b].x < 0 or results.pose_landmarks.landmark[b].x > 1:
                b_body = False
                break
            # if the landmark is not visible enough
            if results.pose_landmarks.landmark[b].visibility < 0.7:
                b_body = False
                break
        pt_11 = [results.pose_landmarks.landmark[11].x * image_width, results.pose_landmarks.landmark[11].y * image_height]
        pt_23 = [results.pose_landmarks.landmark[23].x * image_width, results.pose_landmarks.landmark[23].y * image_height]
        pt_25 = [results.pose_landmarks.landmark[25].x * image_width, results.pose_landmarks.landmark[25].y * image_height]
        pt_12 = [results.pose_landmarks.landmark[12].x * image_width, results.pose_landmarks.landmark[12].y * image_height]
        pt_24 = [results.pose_landmarks.landmark[24].x * image_width, results.pose_landmarks.landmark[24].y * image_height]
        pt_26 = [results.pose_landmarks.landmark[26].x * image_width, results.pose_landmarks.landmark[26].y * image_height]
        hip_angle_left = cal_angle(pt_11, pt_23, pt_25)
        hip_angle_right = cal_angle(pt_12, pt_24, pt_26)
        shoulder_angle_left = cal_angle(pt_12, pt_11, pt_23)
        shoulder_angle_right = cal_angle(pt_11, pt_12, pt_24)
        print("hip_angle_left:", hip_angle_left)
        print("hip_angle_right:", hip_angle_right)
        print("shoulder_angle_left:", shoulder_angle_left)
        print("shoulder_angle_right:", shoulder_angle_right)

        if hip_angle_left > 135 and hip_angle_right > 135:
            b_hip = True
        # pt_8 = np.asarray((hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z))
        # pt_4 = np.asarray((hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z))
        # dist = np.linalg.norm(pt_8-pt_4)
        print("dist_left:", np.linalg.norm(np.asarray(tuple(pt_11))-np.asarray(tuple(pt_23))))
        print("dist_right:", np.linalg.norm(np.asarray(tuple(pt_12))-np.asarray(tuple(pt_24))))

        b_pose = b_head and b_ankle and b_body and b_hip

        print("landmark len:", len(results.pose_landmarks.landmark))
        #print("landmark:", results.pose_landmarks.landmark)
        print("b_head", b_head)
        print("b_ankle", b_ankle)
        print("b_body", b_body)
        print("b_hip", b_hip)
        if b_pose and draw_skeleton:
            # Draw pose landmarks on the image.
            annotated_image = img_patch.copy()
            mp_drawing.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite('./database/img/from_dp/' + str(id) + "_" + str(img_count) + '.jpg', annotated_image)
    # if img_count > 21:
    #     exit(-1)
    img_count += 1
    return b_pose


if __name__ == "__main__":
    # check_pose_manual()
    print("bool:", check_pose("test", "Test"))
