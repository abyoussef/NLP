import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 7, 1)


# Without dropout
acc = [0.8032,0.8898,0.9248,0.9539,0.9721,0.9820 ]
acc_val = [0.8533,0.8427,0.8356,0.8150,0.8260,0.8187 ]

fig1 = plt.figure(1)
plt.plot(epochs,acc,'r',lw=2)
plt.plot(epochs,acc_val,'g',lw=2)
plt.legend({'Acc Train','Acc Valide'})
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#plt.show()
plt.title('Accuracy for training without dropout')
plt.savefig('acc.png')

# With dropout
acc_do = [0.7341,0.8317,0.8589,0.8758,0.8892,0.9002]
acc_do_val = [0.8316,0.8324,0.8416,0.8383,0.8397,0.8383]

fig2 = plt.figure(2)
plt.plot(epochs,acc_do,'r',lw=2)
plt.plot(epochs,acc_do_val,'g',lw=2)
plt.legend({'Acc Train','Acc Valide'})
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy for training with dropout')
#plt.show()
plt.savefig('acc_do.png')