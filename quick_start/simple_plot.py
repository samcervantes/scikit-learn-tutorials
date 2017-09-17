import matplotlib.pyplot as plt 

"""
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
"""

# Red circles
#plt.plot([1,2,3,4], [1,4,9,16], 'ro')
# Blue line
plt.plot([1,2,3,4], [1,4,9,16], 'b-')
# Set the X axis from 0-6 and the Y axis from 0-20
plt.axis([0,6,0,20])
plt.show()
