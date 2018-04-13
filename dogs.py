import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds) # average greyhound height +- 4 inches
lab_height = 24 + 4 * np.random.randn(labs) # average labrador height +- 4 inches

plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

# graph shows data which means that features are useless because classifier cannot predict accurately
