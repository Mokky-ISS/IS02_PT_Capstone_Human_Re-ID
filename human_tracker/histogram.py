# test histogram equilization on blur images
import cv2
import numpy as np

# run historgram on color image
def run_histogram_equalization(image_path):
    rgb_img = cv2.imread(image_path)

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    res = np.hstack((rgb_img, equalized_img))
    
    cv2.imshow('equalized_img', res)
    cv2.waitKey(0)
    
# run histogram on grey scaled image
def run_histogram_equalization_grey(image_path):
    # read a image using imread
    img = cv2.imread(image_path, 0)
      
    # creating a Histograms Equalization
    # of a image using cv2.equalizeHist()
    equ = cv2.equalizeHist(img)
      
    # stacking images side-by-side
    res = np.hstack((img, equ))

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    srp = cv2.filter2D(img, -1, kernel)

    res = np.hstack((res, srp))

    # show image input vs output
    cv2.imshow("image", res)
      
    cv2.waitKey(0)

test_img = "data/helpers/test.jpg"
run_histogram_equalization(test_img)
run_histogram_equalization_grey(test_img)

cv2.destroyAllWindows()

