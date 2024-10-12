import cv2
import numpy as np
import matplotlib.pyplot as plt


def connected_component_label(path):
    # Load the input image in grayscale
    img = cv2.imread(path, 0)

    # Apply a binary threshold or OTSU's threshold for better separation
    _, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply connected components with stats to get more information
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_thresh, connectivity=8)

    # Map component labels to different colors for visualization
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Convert to BGR so we can visualize in color
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # Set the background label to black
    labeled_img[label_hue == 0] = 0

    # Visualizing original, thresholded, and labeled images
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2RGB))
    axs[1].set_title("Thresholded Image")
    axs[1].axis('off')

    axs[2].imshow(labeled_img)
    axs[2].set_title(f"Connected Components: {num_labels}")
    axs[2].axis('off')

    plt.show()

connected_component_label('../Resources/Photos/cats.jpg')