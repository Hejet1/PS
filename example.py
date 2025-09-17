from photostereo import photometry
import cv2 as cv
import time

IMAGES = 4
subject_name = "battery"
root_fold = "./samples/"+str(subject_name)+'/'
obj_name = ""
format = ".bmp"
light_manual = True
level = 3
image_array = []
for id in range(0, IMAGES):
    try:
        filename = root_fold + str(obj_name) + str(id) + format
        path = root_fold + str(obj_name) + str(id) + format
        im = cv.imread(path, cv.IMREAD_GRAYSCALE)
        image_array.append(im)
    except cv.error as err:
        print(err)
myps = photometry(IMAGES,False)
slants = [45, 45, 45, 45]
tilts = [0, 90, 180, 270]
myps.setlmfromts(tilts, slants)
tic = time.process_time()
normal_map, p_grad, q_grad = myps.runphotometry(image_array, None)
med = myps.computemedian(level,ksize=15)
toc = time.process_time()
normal_map = cv.normalize(normal_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
albedo = myps.getalbedo()
albedo = cv.normalize(albedo, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
print("Process duration: " + str(toc - tic))
cv.imwrite('./result/'+str(subject_name)+'normal_map.png',normal_map)
cv.imwrite('./result/'+str(subject_name)+'albedo.png',albedo)
cv.imwrite('./result/'+str(subject_name)+'med.png',med)
