import cv2
import numpy as np
import matplotlib.pyplot as plt 

# تحميل الصورة وتحويلها لرمادية
img = cv2.imread(r'C:\xampp\htdocs\php-practice\New folder\project\doc.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# عرض الصورة الأصلية
plt.figure()
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# FFT ثنائي الأبعاد
ft2 = np.fft.fft2(img)
Fmag = np.log(1 + np.abs(ft2))

# عرض Magnitude Fourier
plt.figure()
plt.imshow(Fmag, cmap='gray')
plt.title('Magnitude Fourier')
plt.axis('off')

# نقل مركز الترددات
ft = np.fft.fftshift(ft2)
Fmag_shifted = np.fft.fftshift(Fmag)

plt.figure()


# إعداد الفلاتر
rows, cols = ft.shape
radius = 100
rm, clm = rows // 2, cols // 2
x = np.arange(-rm, rm)
y = np.arange(-clm, clm)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)

cL = Z < radius
cH = ~cL

# تطبيق الفلاتر
l_ft = ft * cL 
h_ft = ft * cH

# التحويل العكسي
low_filtered_image = np.fft.ifft2(np.fft.ifftshift(l_ft))
high_filtered_image = np.fft.ifft2(np.fft.ifftshift(h_ft))

# تحويل الصور لقيم قابلة للعرض
low_f = cv2.normalize(np.abs(low_filtered_image), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
high_f = cv2.normalize(np.abs(high_filtered_image), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# عرض النتائج
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(cL, cmap='gray')
plt.title('Low-frequency filter')
plt.axis('on')

plt.subplot(2, 2, 2)
plt.imshow(cH, cmap='gray')
plt.title('High-frequency filter')
plt.axis('on')

plt.subplot(2, 2, 3)
plt.imshow(low_f, cmap='gray')
plt.title('Low-frequency image')
plt.axis('on')

plt.subplot(2, 2, 4)
plt.imshow(high_f, cmap='gray')
plt.title('High-frequency image')
plt.axis('on')

plt.tight_layout()
plt.show()
