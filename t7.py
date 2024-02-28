from PIL import Image

gamma = 0.001
image_path = "Dataset/ZZMImgs/forRegressionTrain/raw/ImagingDistance_225-ExposureTime_700-PowerdividedbyArea_757.5-Fnumber_1.4-Focal_8-CMOSPixelSize_2.4-CMOSQE_0.5.bmp"
# 打开图片
img = Image.open(image_path)

# 进行伽马变换
img_gamma = img.point(lambda x: x**gamma)

image_save_path = "Dataset/ZZMImgs/forRegressionTrain/raw/Gamma_ImagingDistance_225-ExposureTime_700-PowerdividedbyArea_757.5-Fnumber_1.4-Focal_8-CMOSPixelSize_2.4-CMOSQE_0.5.bmp"
img_gamma.save(image_save_path)
#
# plt.subplot(121)
# plt.imshow(img, cmap='gray')
#
# plt.subplot(122)
# plt.imshow(img_gamma, cmap='gray')
