import numpy as np
import cv2
import matplotlib.pyplot as plt
import statistics
from scipy.signal import find_peaks

# find the largest element in array
def largest(arr, n):
    # Initialize maximum element
    max = abs(arr[1]-arr[0])

    # Traverse array elements from second
    # and compare every element with
    # current max
    for i in range(1, n-1):
        if abs(arr[i]-arr[i-1]) > max:
            max = abs(arr[i]-arr[i-1])
    return max

# find if the array contains an overshoot
def containOverShoot(arr, n, k):
    for i in range(1, n-1):
        if abs(arr[i]-arr[i-1]) > k:
            return True
    return False

# find an average difference
def avgDiff(peaks):
    diffs = []
    for i in range(1, len(peaks)-1):
        diff = peaks[i]-peaks[i-1]
        diffs.append(diff)
    return sum(diffs)/len(diffs)



cap = cv2.VideoCapture('01-01.mp4')
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 1,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.03))


# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)



face = face_cascade.detectMultiScale(old_gray, 1.1, 4)
x = face[0][0]
y = face[0][1]
w = face[0][2]
h = face[0][3]

xAdj = 20
yAdj_topRec = 150
yAdj_botRec = 80

# r = cv2.rectangle(old_frame, upper_left, bottom_right, (100, 100, 100), 1)
mask = np.zeros_like(old_frame)

results = np.zeros((300,2), np.float32)
results2 = np.zeros((300,1), np.float32)

count = 0


result3 = []


while(count != 299):

    # Continuous readings of frame after the initial frame
    ret, frame = cap.read();
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ROI of initial frame converted to 3 channel
    top_rect_img = old_frame[y: y+h-yAdj_topRec, x+xAdj: x+w-xAdj]
    top_rect_img_gray = cv2.cvtColor(top_rect_img, cv2.COLOR_BGR2GRAY)

    bot_rect_img = old_frame[y+yAdj_botRec: y+h, x+xAdj: x+w-xAdj]
    bot_rect_img_gray = cv2.cvtColor(bot_rect_img, cv2.COLOR_BGR2GRAY)

    # Corners found in initial top
    p0_top = cv2.goodFeaturesToTrack(top_rect_img_gray, 300, 0.01, 5)
    p0_bot = cv2.goodFeaturesToTrack(bot_rect_img_gray, 300, 0.01, 5)
    # optical flow
    p1_top, st_top, err_top = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_top, None, **lk_params)
    p1_bot, st_bot, err_bot = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_bot, None, **lk_params)

    k = 0

    # add time with the y value
    for i, (new, old) in enumerate(zip(p1_top, p0_top)):
        if count == 0:
            result3.append([])
        a, b = new.ravel()
        c, d = old.ravel()
        diff = b-d
        #arr = [count, diff]
        result3[k].append(diff)
        k = k + 1



    # draw the tracks
    for i, (new, old) in enumerate(zip(p1_top, p0_top)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask1 = cv2.line(mask, (a, b), (c, d), 255, 2)
        frame1 = cv2.circle(frame, (int(a + x+xAdj), int(b + y)), 1, 255, -3)

    for i, (new, old) in enumerate(zip(p1_bot, p0_bot)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask2 = cv2.line(mask, (a,b), (c,d), 255, 2)
        frame2 = cv2.circle(frame, (int(a + x+xAdj), int(b + y+yAdj_botRec)), 1, 255, -3)
    count = count + 1


    img1 = cv2.add(frame1, mask1)
    img2 = cv2.add(frame2, mask2)

    img = cv2.add(img1,img2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0_top = p1_top.reshape(-1, 1, 2)
    p0_bot = p1_bot.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()



#Get rid of points with too big movements
max = []
for arrs in result3:
    max.append(round(largest(arrs, len(arrs))))

a = statistics.mean(max)
print('Average is {0}'.format(a))
print('before trimming is {0}'.format(len(result3)))

for arr in result3:
    if (containOverShoot(arr, len(arr), a)):
        result3.remove(arr)


result3 = np.asarray(result3, dtype = np.float32)



converted = cv2.cvtColor(result3[10], cv2.COLOR_GRAY2BGR)

# resize the frequency
outF = 250/30
height = int(converted.shape[0]*outF)
width = int(converted.shape[1])

resultResized = cv2.resize(converted, (width, height), interpolation = cv2.INTER_CUBIC)


graph = []
for i in range(len(resultResized)):
    graph.append(resultResized[i][0][0])

graph = np.array(graph)

peaks, _ = find_peaks(graph, prominence=0.2)
print(peaks)
pulse = avgDiff(peaks)
print('calculated pulse is {0}'.format(pulse, 0))


# show peak detections
plt.plot(graph)
plt.plot(peaks, graph[peaks], "x")
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()













