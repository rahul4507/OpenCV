# # Erosion
# # Dilation
# # Opening
# # Closing
# # Morphological gradient
# # Black hat
# # Top hat (also called “White hat”)
import os
import cv2 as cv
import matplotlib.pyplot as plt

script_name = os.path.basename(__file__)

# Read and resize the image
img = cv.imread('../Resources/Photos/park.jpg', cv.IMREAD_COLOR)
img = cv.resize(img, (500, 500), cv.INTER_AREA)

# Kernel sizes to apply for erosion and dilation
kernel_sizes = [(3, 3), (5, 5), (7, 7)]

# Create subplots for erosion and dilation
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle(f'Erosion and Dilation with Different Kernel Sizes - {script_name}')

# Show the original image in the first column of each row
for row in range(2):
    axes[row, 0].imshow(img)
    axes[row, 0].set_title('Original Image')
    axes[row, 0].axis('off')

# 1. Erosion for each kernel size
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    eroded = cv.erode(img, kernel, iterations=1)

    # Show erosion result in the first row
    axes[0, i+1].imshow(eroded)
    axes[0, i+1].set_title(f'Erosion: {kernel_size}')
    axes[0, i+1].axis('off')

# 2. Dilation for each kernel size
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    dilated = cv.dilate(img, kernel, iterations=1)

    # Show dilation result in the second row
    axes[1, i+1].imshow(dilated)
    axes[1, i+1].set_title(f'Dilation: {kernel_size}')
    axes[1, i+1].axis('off')

# Display the plots
plt.tight_layout()
plt.show()

# Create subplots for opening and closing
fig, axes = plt.subplots(3, 4, figsize=(25, 8))
fig.suptitle(f'Opening and Closing with Different Kernel Sizes - {script_name}')

# Show the original image in the first column of each row
for row in range(3):
    axes[row, 0].imshow(img)
    axes[row, 0].set_title('Original Image')
    axes[row, 0].axis('off')

# 1. Opening for each kernel size
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel )

    # Show erosion result in the first row
    axes[0, i+1].imshow(opened)
    axes[0, i+1].set_title(f'Opening: {kernel_size}')
    axes[0, i+1].axis('off')

# 2. Closing for each kernel size
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    # Show dilation result in the second row
    axes[1, i+1].imshow(closed)
    axes[1, i+1].set_title(f'Closing: {kernel_size}')
    axes[1, i+1].axis('off')

# 3. Morphological Gradient for each kernel size
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    morph_grad = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

    # Show dilation result in the second row
    axes[2, i+1].imshow(morph_grad)
    axes[2, i+1].set_title(f'Morphological Gradient: {kernel_size}')
    axes[2, i+1].axis('off')

# Display the plots
plt.tight_layout()
plt.show()


# Create subplots for TopHat and BlackHat
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle(f'TopHat and BlackHat with Different Kernel Sizes - {script_name}')

# Show the original image in the first column of each row
for row in range(2):
    axes[row, 0].imshow(img)
    axes[row, 0].set_title('Original Image')
    axes[row, 0].axis('off')

# 1. BlackHat for each kernel size
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    black_hat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

    # Show erosion result in the first row
    axes[0, i+1].imshow(black_hat)
    axes[0, i+1].set_title(f'BlackHat: {kernel_size}')
    axes[0, i+1].axis('off')

# 2. TopHat for each kernel size
for i, kernel_size in enumerate(kernel_sizes):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_size)
    top_hat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

    # Show dilation result in the second row
    axes[1, i+1].imshow(top_hat)
    axes[1, i+1].set_title(f'TopHat: {kernel_size}')
    axes[1, i+1].axis('off')

# Display the plots
plt.tight_layout()
plt.show()
