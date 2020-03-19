import matplotlib.pyplot as plt
import numpy as np

f = open("losses_record.txt", "r")
w = open("new_losses.txt", "w")
fl = f.readlines()
frame = []
rewards = []

index = 0
for line in fl:
    if index % 10000 == 0:
        one_line = line.split(",")
        # frame.append(float(one_line[0]) / 100000)
        # rewards.append(one_line[1])
        w.writelines(str(one_line[0]) + " , " + str(one_line[1]))

    index += 1
f.close()
w.close()
# plt.yticks(np.arange(-25, 26, 5))
# plt.tick_params(axis='both', which='both', length=10)
# plt.plot(frame, rewards)
# plt.yticks(np.arange(-20, 20, 5))
# plt.tight_layout

# plt.show()