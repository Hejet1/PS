import cv2 as cv
import numpy as np
import time
class photometry:

    def __init__(self, numimg, display):
        self.IMAGES = numimg
        self.display = display
        self.normalmap = []
        self.albedo = []
        self.pgrads = []
        self.qgrads = []
        self.gaussgrad = []
        self.meangrad = []
        self.mask = []
        self.Z = []
    def runphotometry(self, input_array, mask=None):
        print("Running main process. Be patient...")
        print("Computing normal map. Be patient...")
        normaltic = time.process_time()
        if (mask is not None):
            self.mask = mask
            for id in range(0, self.IMAGES):
                mult = (mask / 255)
                input_array[id] = np.multiply(input_array[id], mult.astype(np.uint8))
        # Convert input array to float img array
        input_arr_conv = []
        for id in range (0, self.IMAGES):
            im_fl = np.float32(input_array[id])
            im_fl = im_fl / 255
            input_arr_conv.append(im_fl)
        h = input_arr_conv[0].shape[0]
        w = input_arr_conv[0].shape[1]
        self.normalmap = np.zeros((h, w, 3), dtype=np.float32)
        self.pgrads = np.zeros((h, w), dtype=np.float32)
        self.qgrads = np.zeros((h, w), dtype=np.float32)
        lpinv = np.linalg.pinv(self.light_mat)
        intensities = []
        norm = []
        for imid in range(0, self.IMAGES):
            a = np.array(input_arr_conv[imid]).reshape(-1)
            intensities.append(a)
        intensities = np.array(intensities)
        rho_z = np.einsum('ij,jk->ik', lpinv, intensities)
        rho = rho_z.transpose()
        norm.append(np.sum(np.abs(rho)**2, axis=-1)**(1./2))
        norm_t = np.array(norm).transpose()
        norm_t = np.clip(norm_t, 0 , 1)
        norm_t = np.where(norm_t==0, 1, norm_t)
        self.albedo = np.reshape(norm_t, (h, w))
        rho = np.divide(rho , norm_t)
        rho[:, 2] = np.where(rho[:, 2] == 0, 1, rho[:, 2])
        rho = np.asarray(rho).transpose()
        self.normalmap[:, :, 0] = np.reshape(rho[0], (h, w))
        self.normalmap[:, :, 1] = np.reshape(rho[1], (h, w))
        self.normalmap[:, :, 2] = np.reshape(rho[2], (h, w))
        self.pgrads[0:h, 0:w] = self.normalmap[:, :, 0] / self.normalmap[:, :, 2]
        self.qgrads[0:h, 0:w] = self.normalmap[:, :, 1] / self.normalmap[:, :, 2]
        self.normalmap1 = self.normalmap.astype(np.float32)
        self.normalmap = cv.cvtColor(self.normalmap1, cv.COLOR_BGR2RGB)
        output_int = cv.normalize(self.normalmap, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
        output_int = cv.bitwise_and(output_int, output_int, mask = mask)
        self.normalmap = cv.bitwise_and(self.normalmap, self.normalmap, mask = mask)
        if self.display:
            cv.imshow('normal_normalized', output_int)
            cv.imshow('albedo', self.albedo)
            cv.imshow('self.pgrads', self.pgrads)
            cv.imshow('self.qgrads', self.qgrads)
            cv.waitKey(0)
            cv.destroyAllWindows()
        print("Normal map computation end ")
        normaltoc = time.process_time()
        print("Normal map duration: " + str(normaltoc - normaltic))
        return self.normalmap, self.pgrads, self.qgrads
    def scale_image_range(self, src, new_min_val, new_max_val):
        mult = (255 - 0)/(new_max_val - new_min_val)
        add = -mult*new_min_val+0
        dst = cv.convertScaleAbs(src, alpha=mult, beta=add)
        return dst
    def computemedian(self, enhance_level, ksize=15):

        print("Computing median curvature. Be patient...")
        medtic = time.process_time()

        h, w = self.pgrads.shape
        self.meangrad = np.zeros((h, w), dtype=np.float32)

        Ixx = cv.Sobel(self.pgrads, cv.CV_32F, 1, 0, ksize=ksize)
        Ixy = cv.Sobel(self.pgrads, cv.CV_32F, 0, 1, ksize=ksize)
        Iyy = cv.Sobel(self.qgrads, cv.CV_32F, 0, 1, ksize=ksize)
        Iyx = cv.Sobel(self.qgrads, cv.CV_32F, 1, 0, ksize=ksize)

        a = (1 + self.pgrads ** 2) * Iyy
        b = self.pgrads * self.qgrads * (Ixy + Iyx)
        c = (1 + self.qgrads ** 2) * Ixx
        d = (1 + self.pgrads ** 2 + self.qgrads ** 2) ** (3 / 2)

        self.meangrad = (a - b + c) / d

        print("Median curvature computation end.")
        medtoc = time.process_time()
        print(f"Median duration: {medtoc - medtic:.3f}s")

        meangrad_norm = cv.normalize(self.meangrad, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        mean_val = np.mean(meangrad_norm[meangrad_norm > 0])

        if 0 < enhance_level <= 10:
            meangrad_norm = self.scale_image_range(
                meangrad_norm,
                mean_val - (110 - enhance_level * 10),
                mean_val + (110 - enhance_level * 10)
            )

        if self.display:
            cv.imshow('meangrad', meangrad_norm)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return meangrad_norm
    def setlmfromts(self, tilt, slant):
        # todo: add check on tilt and slant size
        self.light_mat = np.zeros((self.IMAGES, 3), dtype=np.float32)
        rads = 180 / np.pi
        for id in range (0 , self.IMAGES):
            self.light_mat[id , 0] = np.cos(tilt[id] / rads)
            self.light_mat[id , 1] = np.sin(tilt[id] / rads)
            self.light_mat[id , 2] = np.cos(slant[id] / rads)
            norm = np.linalg.norm(self.light_mat[id])
            self.light_mat[id] = self.light_mat[id]/norm
    def getalbedo(self):
        return self.albedo
