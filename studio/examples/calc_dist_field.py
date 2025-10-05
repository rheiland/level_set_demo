# Compute distance from all (voxel) pts to nearest boundaries
#
# python calc_dist_field.py pbm_gbm_capped.csv 0  # unsigned distance
# or,
# python calc_dist_field.py pbm_gbm_capped.csv 1  # signed distance
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
pts = genfromtxt(csv_file, delimiter=',')
print(pts.shape)
n = len(pts[:,0])
print("# pts = ",n)

signed_dist_flag = False
signed_flag = int(sys.argv[2])
if signed_flag > 0:
    signed_dist_flag = True

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def perp( a ):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom)*db + b1

fig = plt.figure(figsize=(9,7))
#ax = fig.gca()

l1_p0 = np.array( [0.0, 0.0] )
l1_p1 = np.array( [0.0, 0.0] )
l2_p0 = np.array( [0.0, 0.0] )
l2_p1 = np.array( [0.0, 0.0] )
for idx in range(n-1):
    # print(idx,") ",xpts_new[idx],ypts_new[idx], " -> ", xpts_new[idx+1],ypts_new[idx+1])
    l2_p0[0] = pts[idx,0]
    l2_p0[1] = pts[idx,1]
    l2_p1[0] = pts[idx+1,0]
    l2_p1[1] = pts[idx+1,1]

    # ptint = seg_intersect(l1_p0,l1_p1, l2_p0,l2_p1)
    # xmin = min(lseg_p0[0], lseg_p1[0])
    # xmax = max(lseg_p0[0], lseg_p1[0])
    # if ptint[0] >= xmin and ptint[0] <= xmax:
    #     print("--> ",ptint[0],ptint[1])
    # # else:
    #     # print("-- no intersection.")

def f(x, y):
    return np.sin(x) + np.cos(y)

nvoxels = 10
nvoxels = 20
nvoxels = 80

nx_voxels = 88
ny_voxels = 75

# with double resolution of domain, i.e., dx=dy=dz=10
nx_voxels = 175
ny_voxels = 150

# increase again, dx=dy=dz=2
nx_voxels = 875
ny_voxels = 750

print("Using grid nx x ny = ",nx_voxels,ny_voxels)

# x = np.linspace(-800, 600, nvoxels)
# y = np.linspace(-700, 700, nvoxels)
x = np.linspace(-750, 1000, nx_voxels)
y = np.linspace(-750, 750, ny_voxels)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)   # Z.shape = (50, 50)

l1_p0 = np.array( [0.0, 0.0] )
l1_p1 = np.array( [0.0, 0.0] )
l2_p0 = np.array( [0.0, 0.0] )
l2_p1 = np.array( [0.0, 0.0] )


intpts = []
# for every point (voxel center) in our grid...
for iy in range(len(y)):
    yval = y[iy]
    l1_p0[1] = yval
    l1_p1[1] = 800  # outside domain
    for ix in range(len(x)):
        xval = x[ix]
        l1_p0[0] = xval
        l1_p1[0] = xval

        # 1st pass: compute min distance to each bdy pt
        dist_min = 1.e6
        for idx in range(n-1):
            if math.fabs(pts[idx,0]) < 1.e-6 and math.fabs(pts[idx,1]) < 1.e-6:  # hack separator "0,0" between regions
                continue
            elif math.fabs(pts[idx+1,0]) < 1.e-6 and math.fabs(pts[idx+1,1]) < 1.e-6:  # hack separator "0,0" between regions
                continue
            xd = xval - pts[idx,0]
            yd = yval - pts[idx,1]
            dist2 = xd*xd + yd*yd
            if dist2 < dist_min:
                dist_min = dist2
            
            # how many intersections with boundaries - even or odd #?
            # l2_p0[0] = pts[idx,0]
            # l2_p0[1] = pts[idx,1]
            # l2_p1[0] = pts[idx+1,0]
            # l2_p1[1] = pts[idx+1,1]
            # ptint = seg_intersect(l1_p0,l1_p1, l2_p0,l2_p1)
            # xmin = min(lseg_p0[0], lseg_p1[0])
            # xmax = max(lseg_p0[0], lseg_p1[0])
            # if ptint[0] >= xmin and ptint[0] <= xmax:
            #     print("--> ",ptint[0],ptint[1])
            # # else:
            #     # print("-- no intersection.")

        dist = math.sqrt(dist_min)
        if signed_dist_flag:
            Z[iy,ix] = -dist
        else:
            Z[iy,ix] = dist

        # 2nd pass: inside or outside a vessel region
        intpts.clear()
        for idx in range(n-1):
        # for idx in range(0,-1):
            if math.fabs(pts[idx,0]) < 1.e-6 and math.fabs(pts[idx,1]) < 1.e-6:  # hack separator "0,0" between regions
                continue
            elif math.fabs(pts[idx+1,0]) < 1.e-6 and math.fabs(pts[idx+1,1]) < 1.e-6:  # hack separator "0,0" between regions
                continue
            # how many intersections with boundaries - even or odd #?
            # if idx < 5:
            if True:
                # check each line seg on boundaries
                l2_p0[0] = pts[idx,0]
                l2_p0[1] = pts[idx,1]
                l2_p1[0] = pts[idx+1,0]
                l2_p1[1] = pts[idx+1,1]
                ptint = seg_intersect(l1_p0,l1_p1, l2_p0,l2_p1)
                # if ptint[0] > -800 and ptint[0] < 600 and ptint[1] > -600 and ptint[1] < 600:
                px0 = l2_p0[0]  # get left and right x values of line seg
                px1 = l2_p1[0]
                if px0 > px1:
                    px0 = l2_p1[0]
                    px1 = l2_p0[0]
                # if ptint[0] > -800 and ptint[0] < 600 and ptint[1] > -600 and ptint[1] < 600:
                # if ptint[0] >= px0 and ptint[0] <= px1 and ptint[1] > -600 and ptint[1] < 600:
                if ptint[0] >= px0 and ptint[0] <= px1 and ptint[1] > yval:
                    # print(idx,": ptint = ",ptint[0], ptint[1])
                    intpts.append(ptint[1])
        # print(iy,ix,intpts)
        if signed_dist_flag and (len(intpts) == 0 or len(intpts)%2 == 0):   # check for outside bdy
            # print("len(intpts)= ",len(intpts))
            Z[iy,ix] = -Z[iy,ix]
        # else:
            # print(intpts)


print("Z.min() = ",Z.min())
print("Z.max() = ",Z.max())
out_file = "Z_dist.dat"
np.savetxt(out_file,Z,delimiter=',')
print("--> ",out_file, " (rename to read as a substrate)")

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
ncontours = 30
CS1 = plt.contourf(X, Y, Z, ncontours, norm=norm, cmap='seismic')
#CS2 = plt.contour(X, Y, Z, [0.0])  # UserWarning: No contour levels were found within the data range.
if signed_dist_flag:
    zval = 10.0  # depends on min(Z),max(Z)
    CS2 = plt.contour(X, Y, Z, [zval])
# CS2.collections[0].set_linewidth(2)
#CS2.collections[0].set_color('black')

plt.colorbar(CS1) 
if signed_dist_flag:
    plt.title("signed dist; grid: " + str(nx_voxels) + "x" + str(ny_voxels) + "; zcont="+str(zval))
else:
    plt.title("unsigned dist; grid: " + str(nx_voxels) + "x" + str(ny_voxels))
plt.show()
