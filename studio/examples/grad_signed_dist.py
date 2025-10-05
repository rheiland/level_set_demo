# Compute gradient of a distance field (2D grid)
# python grad_signed_dist.py Z_dist.dat
_author__ = "Randy Heiland"

import sys
import math
import matplotlib.pyplot as plt
# plt.style.use('seaborn-white')
import numpy as np
from numpy import genfromtxt
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as mcolors


csv_file = sys.argv[1]
dval = genfromtxt(csv_file, delimiter=',')
print("dval.shape = ",dval.shape)
n = len(dval[:,0])
print("type(dval)= ",type(dval))
print("# dval (rows)= ",n)
print("# dval (col)= ",len(dval[0,:]))

#sys.exit(1)

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


nx_voxels = 88
ny_voxels = 75

# with double resolution of domain, i.e., dx=dy=dz=10
# nx_voxels = 175
# ny_voxels = 150

# # increase again, dx=dy=dz=2
# nx_voxels = 875
# ny_voxels = 750

# x = np.linspace(-800, 600, nvoxels)
# y = np.linspace(-700, 700, nvoxels)
x = np.linspace(-750, 1000, nx_voxels)
y = np.linspace(-750, 750, ny_voxels)

X, Y = np.meshgrid(x, y)
# Z = f(X, Y)   # Z.shape = (50, 50)
Z = dval

#grad = np.gradient(Z, 20.0)  # 2nd arg is the voxel width?
grad = np.gradient(Z)  # 2nd arg is the voxel width?
print("type(grad) = ",type(grad))
#print("grad.shape = ",grad.shape)
print("len(grad) = ",len(grad))
print("grad[0].shape = ",grad[0].shape)
print("grad[1].shape = ",grad[1].shape)

print("Z.min() = ",Z.min())
print("Z.max() = ",Z.max())
out_file = "Z_dist.dat"
#np.savetxt(out_file,Z,delimiter=',')
#print("--> ",out_file, " (rename to read as a substrate)")

# plt.contourf(X, Y, Z, 40, cmap='RdGy')

#cmap = plt.get_cmap('PuOr')
cmap = plt.get_cmap('PiYG')
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize and forcing 0 to be part of the colorbar!
# print("np.min(Z) = ",np.min(Z))
# print("np.max(Z) = ",np.max(Z))
# bounds = np.arange(np.min(Z),np.max(Z),50)  # 0.5
# idx=np.searchsorted(bounds,0)
# bounds=np.insert(bounds,idx,0)
# norm = BoundaryNorm(bounds, cmap.N)
norm = MidpointNormalize( midpoint = 0 )


# interpolation='none',norm=norm,cmap=cmap
#plt.contourf(X, Y, Z, 40, cmap='PiYG')
#plt.contourf(X, Y, Z, 40, norm=norm, cmap=cmap)
CS1 = plt.contourf(X, Y, Z, 40, norm=norm, cmap='seismic')
CS2 = plt.contour(X, Y, Z, [0.0])
# CS2.collections[0].set_linewidth(2)
# CS2.collections[0].set_color('black')

#V = np.array([[-1,0], [0,1], [2,2], [1,0]])
#origin = np.array([[-400, -100, 200, 600],[0, 0, 0,0 ]]) # origin point
#plt.quiver(*origin, V[:,0], V[:,1], color=['r','g','b','k'], scale=20)
U = grad[0]
V = grad[1]
plt.quiver(X, Y, U, V, scale_units='x', scale=1) #, scale=0.005)

plt.colorbar(CS1) 
plt.title("gradient and signed dist; grid: " + str(nx_voxels) + "x" + str(ny_voxels))
plt.show()
