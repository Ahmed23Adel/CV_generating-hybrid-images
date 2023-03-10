import matplotlib.pyplot as plt
from time import time
from helpers import vis_hybrid_image, load_image, save_image, my_imfilter, gen_hybrid_image

# Read images and convert to floating point format
image1 = load_image('../data/dog.bmp')
image2 = load_image('../data/cat.bmp')

# Parameters
type = 1  # 1 for FFT based convolution, 0 for normal convolution
cutoff_frequency = 5  # We used the formula: sigma= (k-1)/6
k = 31  # Filter size

# Do the logic and calculate taken time
t1 = time()
low_freq, high_freq, hybrid_image = gen_hybrid_image(image1, image2, cutoff_frequency, k, type)
vis = vis_hybrid_image(hybrid_image)
t2 = time()
print("Time taken", t2 - t1)

# display images before and after editing
plt.imshow((image1));
plt.title("Image 1 (To extract Low freq)");
plt.show()
plt.imshow((image2));
plt.title("Image 2 (To extract High freq)");
plt.show()
plt.imshow((low_freq));
plt.title("Low Freq Image");
plt.show()
plt.imshow((high_freq + 0.5).clip(0, 1));
plt.title("High Freq Image");
plt.show()
plt.imshow((hybrid_image));
plt.title("hybrid");
plt.show()
plt.imshow((vis));
plt.title("Different Views");
plt.show()

save_image('../results/part2_low_frequencies.jpg', low_freq)
save_image('../results/part2_high_frequencies.jpg', high_freq)
save_image('../results/part2_hybrid_image.jpg', hybrid_image)
save_image('../results/part2_image_scales.jpg', vis)
