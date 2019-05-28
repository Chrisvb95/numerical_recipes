#a2_5.py
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def NGP(p,N):

    mesh = np.zeros((N,N,N))
    for i in range(len(p[0])):
        x,y,z = np.round(p[:,i])%N
        mesh[int(x)][int(y)][int(z)] += 1
        
    return mesh

def CiC(p,N):

    mesh = np.zeros((N,N,N))
    for i in range(len(p[0])):
        w = np.zeros(8)
        x,y,z = np.round(p[:,i])
        dx,dy,dz = (x-p[0,i]),(y-p[1,i]),(z-p[2,i])
        #print(x,p[0,i],y,p[1,i],z,p[2,i])
        #print('delta:',dx,dy,dz)
        sx,sy,sz = np.sign(dx),np.sign(dy),np.sign(dz)
        dx,dy,dz = np.abs(dx),np.abs(dy),np.abs(dz)
        x,y,z = x%N,y%N,z%N
        # Calculating all of the weights
        w[0] = (1-dx)*(1-dy)*(1-dz)
        w[1] = (dx)*(1-dy)*(1-dz)
        w[2] = (1-dx)*(dy)*(1-dz)
        w[3] = (1-dx)*(1-dy)*(dz)
        w[4] = (dx)*(dy)*(dz-1)
        w[5] = (dx)*(1-dy)*(dz)
        w[6] = (1-dx)*(dy)*(dz)
        w[7] = (dx)*(dy)*(dz) 
        #print(w)       
        # Assigning the weights
        mesh[np.int(x)%N][np.int(y)%N][np.int(z)%N] += w[0]
        mesh[np.int(x-sx)%N][np.int(y)%N][np.int(z)%N] += w[1]
        mesh[np.int(x)%N][np.int(y-sy)%N][np.int(z)%N] += w[2]
        mesh[np.int(x)%N][np.int(y)%N][np.int(z-sz)%N] += w[3]
        mesh[np.int(x-sx)%N][np.int(y-sy)%N][np.int(z)%N] += w[4]
        mesh[np.int(x-sx)%N][np.int(y)%N][np.int(z-sz)%N] += w[5]
        mesh[np.int(x)%N][np.int(y-sy)%N][np.int(z-sz)%N] += w[6]   
        mesh[np.int(x-sx)%N][np.int(y-sy)%N][np.int(z-sz)%N] += w[7]

    return mesh

def CiC_reverse(p,gx,gy,gz,N):

    p_out = np.zeros((p.shape))

    for i in range(len(p[0])):
        w = np.zeros(8)
        x,y,z = np.round(p[:,i])
        dx,dy,dz = (x-p[0,i]),(y-p[1,i]),(z-p[2,i])
        #print(x,p[0,i],y,p[1,i],z,p[2,i])
        #print('delta:',dx,dy,dz)
        sx,sy,sz = np.sign(dx),np.sign(dy),np.sign(dz)
        dx,dy,dz = np.abs(dx),np.abs(dy),np.abs(dz)
        x,y,z = x%N,y%N,z%N
        # Calculating all of the weights
        w[0] = (1-dx)*(1-dy)*(1-dz)
        w[1] = (dx)*(1-dy)*(1-dz)
        w[2] = (1-dx)*(dy)*(1-dz)
        w[3] = (1-dx)*(1-dy)*(dz)
        w[4] = (dx)*(dy)*(dz-1)
        w[5] = (dx)*(1-dy)*(dz)
        w[6] = (1-dx)*(dy)*(dz)
        w[7] = (dx)*(dy)*(dz) 
        #print(w)       
        # Assigning the weights
        # (I am aware of the fact that this is extremely ugly coding, would be the first that I would fix if I had more time to spend on this)
        p_out[:,i] += (gx[np.int(x)%N][np.int(y)%N][np.int(z)%N]* w[0],gy[np.int(x)%N][np.int(y)%N][np.int(z)%N]* w[0],gz[np.int(x)%N][np.int(y)%N][np.int(z)%N]* w[0]) 
        p_out[:,i] += (gx[np.int(x-sx)%N][np.int(y)%N][np.int(z)%N]* w[1],gy[np.int(x-sx)%N][np.int(y)%N][np.int(z)%N]* w[1],gz[np.int(x-sx)%N][np.int(y)%N][np.int(z)%N]* w[1]) 
        p_out[:,i] += (gx[np.int(x)%N][np.int(y-sy)%N][np.int(z)%N]* w[2],gy[np.int(x)%N][np.int(y-sy)%N][np.int(z)%N]* w[2],gz[np.int(x)%N][np.int(y-sy)%N][np.int(z)%N]* w[2]) 
        p_out[:,i] += (gx[np.int(x)%N][np.int(y)%N][np.int(z-sz)%N]* w[3],gy[np.int(x)%N][np.int(y)%N][np.int(z-sz)%N]* w[3],gz[np.int(x)%N][np.int(y)%N][np.int(z-sz)%N]* w[3]) 
        p_out[:,i] += (gx[np.int(x-sx)%N][np.int(y-sy)%N][np.int(z)%N]* w[4],gy[np.int(x-sx)%N][np.int(y-sy)%N][np.int(z)%N]* w[4],gz[np.int(x-sx)%N][np.int(y-sy)%N][np.int(z)%N]* w[4]) 
        p_out[:,i] += (gx[np.int(x-sx)%N][np.int(y)%N][np.int(z-sz)%N]* w[5],gy[np.int(x-sx)%N][np.int(y)%N][np.int(z-sz)%N]* w[5],gz[np.int(x-sx)%N][np.int(y)%N][np.int(z-sz)%N]* w[5]) 
        p_out[:,i] += (gx[np.int(x)%N][np.int(y-sy)%N][np.int(z-sz)%N]* w[6],gy[np.int(x)%N][np.int(y-sy)%N][np.int(z-sz)%N]* w[6],gz[np.int(x)%N][np.int(y-sy)%N][np.int(z-sz)%N]* w[6])   
        p_out[:,i] += (gx[np.int(x-sx)%N][np.int(y-sy)%N][np.int(z-sz)%N]* w[7],gy[np.int(x-sx)%N][np.int(y-sy)%N][np.int(z-sz)%N]* w[7],gz[np.int(x-sx)%N][np.int(y-sy)%N][np.int(z-sz)%N]* w[7]) 

    return p_out


def fft1D(x,Nj,x0=0,step=1,inv=False):
    if inv:
        j2 = 2j
    else:
        j2 = -2j     
    if Nj == 1: 
        #print('Reached bottom',[x[x0]])
        return [x[x0]]
    new_step = step*2
    hNj = Nj//2
    rs = fft1D(x,hNj,x0,new_step,inv=inv)+fft1D(x,hNj,x0+step,new_step,inv=inv)
    rs_new = np.copy(rs)
    for i in range(hNj):
        rs[i],rs[i+hNj]=rs[i]+np.exp(j2*np.pi*i/Nj)*rs[i+hNj],rs[i]-np.exp(j2*np.pi*i/Nj)*rs[i+hNj]
    return rs

def fft2D(x,inv=False):

    x = np.array(x,dtype=complex)
    if len(x.shape) == 2:
        for i in range(x.shape[1]):
            x[:,i] = fft1D(x[:,i],len(x[1]),inv=inv)
        for j in range(x.shape[0]):
            x[j] = fft1D(x[j],len(x[0]),inv=inv)
        return x

def fft3D(x,inv=False):

    x = np.array(x,dtype=complex)
    
    for k in range(x.shape[2]):
        for i in range(x.shape[1]):
            x[k,:,i] = fft1D(x[k,:,i],len(x[1]),inv=inv)
        for j in range(x.shape[0]):
            x[k][j] = fft1D(x[k][j],len(x[0]),inv=inv)
    
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            x[:,i,j] = fft1D(x[:,i,j],len(x[2]),inv=inv)
    
    return x

def central_diff_3D(a):
    # Setting constants and array
    N = len(a[0])
    gradx = np.zeros((a.shape))
    grady = np.zeros((a.shape))
    gradz = np.zeros((a.shape))
    # Running through all values in array
    for i in range(N):
        for j in range(N):
            for k in range(N):
                gradx[i][j][k] = a[(i+1)%N][j][k]-a[(i-1)%N][j][k]
                grady[i][j][k] = a[i][(j+1)%N][k]-a[i][(j-1)%N][k]
                gradz[i][j][k] = a[i][j][(k+1)%N]-a[i][j][(k-1)%N]
    return gradx,grady,gradz



if __name__ == '__main__':
    print('--- Exercise 5 ---')
    
    # --- 5.a --- 
    print('Original seed:',121)
    np.random.seed(121)
    N = 16
    positions = np.random.uniform(low=0,high=16,size=(3,1024))
    # Calculating the mesh
    mesh_ngp = NGP(positions,N)
    vmax = np.max(mesh_ngp)
    # Plotting the mesh 
    fig = plt.figure(1,(30,30))
    grid = AxesGrid(fig, 142,
                    nrows_ncols=(2, 2),
                    axes_pad=(0.15,0.45),
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    )
    im = grid[0].imshow(mesh_ngp[:,:,3],vmin=0, vmax=vmax)
    grid[0].set_title('z = 4')
    im = grid[1].imshow(mesh_ngp[:,:,8],vmin=0, vmax=vmax)
    grid[1].set_title('z = 9')
    im = grid[2].imshow(mesh_ngp[:,:,10],vmin=0, vmax=vmax)
    grid[2].set_title('z = 11')
    im = grid[3].imshow(mesh_ngp[:,:,13],vmin=0, vmax=vmax)
    grid[3].set_title('z = 14')
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
            cax.toggle_label(True)

    fig.suptitle('Four different z-slices of a mesh produced using NGP',x=0.38,y=0.64,fontsize=14)
    fig.tight_layout()
    plt.savefig('./plots/5a.png',bbox_inches='tight',pad_inches = 0.5)
    plt.close()

    # --- 5.b ---
    test_points = np.arange(0,16,0.1)
    cell4 = np.zeros(len(test_points))
    cell0 = np.zeros(len(test_points))

    # 1-D implementation of the NGP method
    for i in range(len(test_points)):
        mesh1d = np.zeros(N)
        x = np.round(test_points[i])%N
        mesh1d[int(x)] += 1
        cell4[i] = mesh1d[4]
        cell0[i] = mesh1d[0]
    # Plotting  
    plt.plot(test_points,cell4,label='cell 4')
    plt.plot(test_points,cell0,label='cell 0')
    plt.xlabel('x-position of a particle')
    plt.ylabel('Value in cell')
    plt.title('X-position of a particle and cell values - NGP')
    plt.legend()
    plt.savefig('./plots/5b.png')
    plt.close()

    # --- 5.c --- 
    # Calculating the mesh
    mesh = CiC(positions,16)
    vmax = np.max(mesh)
    fig = plt.figure(1,(30,30))
    grid = AxesGrid(fig, 142,
                    nrows_ncols=(2, 2),
                    axes_pad=(0.15,0.45),
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    im = grid[0].imshow(mesh[:,:,4])#,vmin=0, vmax=vmax)
    grid[0].set_title('z = 4')
    im = grid[1].imshow(mesh[:,:,9])#,vmin=0, vmax=vmax)
    grid[1].set_title('z = 9')
    im = grid[2].imshow(mesh[:,:,11])#,vmin=0, vmax=vmax)
    grid[2].set_title('z = 11')
    im = grid[3].imshow(mesh[:,:,14])#,vmin=0, vmax=vmax)
    grid[3].set_title('z = 14')
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
            cax.toggle_label(True)
    fig.suptitle('Four different z-slices of a mesh produced using CiC',x=0.38,y=0.64,fontsize=14)
    fig.tight_layout()
    plt.savefig('./plots/5c.png',bbox_inches='tight',pad_inches = 0.5)
    plt.close()

    # 1-D implementation of the CiC method
    cell4 = np.zeros(len(test_points))
    cell0 = np.zeros(len(test_points))
    for i in range(len(test_points)):
        w = np.zeros(2)
        mesh1d = np.zeros(N)
        x = np.round(test_points[i])
        dx = x-test_points[i]
        sx = np.sign(dx)
        dx = np.abs(dx)
        x=x%N
        w[0] = 1-dx
        w[1] = dx
        mesh1d[np.int(x)%N]+=w[0]
        mesh1d[np.int(x-sx)%N]+=w[1]
        cell4[i] = mesh1d[4]
        cell0[i] = mesh1d[0]
        
    plt.plot(test_points,cell4,label='cell 4')
    plt.plot(test_points,cell0,label='cell 0')
    plt.xlabel('x-position of a particle')
    plt.ylabel('Value in cell')
    plt.title('X-position of a particle and cell values - CiC')
    plt.legend()
    plt.savefig('./plots/5d.png')
    plt.close()

    # --- 5.d ---
    f = lambda x: np.cos(x)
    x = np.linspace(-np.pi,np.pi,1024)
    fx = f(x)
    fftself = fft1D(fx,len(fx))
    fftnumpy = np.fft.fft(fx)
    plt.axvline(x=np.pi,color='green')
    plt.axvline(x=-np.pi,color='green',label='Analytical')
    plt.plot(x,np.abs(fftself),label='Self written function')
    plt.plot(x,np.abs(fftnumpy),ls='--',label='Numpy function')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('$\mathcal{F(cos(x))}}$')
    plt.title('1D Fourier transform of $cos(x)$')
    plt.savefig('./plots/5e.png')
    plt.close()

    # --- 5.e --- 
    f2d = lambda x,y: np.cos(x+y)
    N = 16
    x = np.linspace(-3,3,N)
    y = np.linspace(-3,3,N)
    f2dxy = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            f2dxy[i][j] = f2d(x[i],y[j])

    fftself2 = fft2D(f2dxy)
    fftnumpy2 = np.fft.fft2(f2dxy)

    fig = plt.figure(1,(30,30))
    grid = AxesGrid(fig, 142,
                    nrows_ncols=(1, 2),
                    axes_pad=(0.15,0.45),
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    im = grid[0].imshow(np.abs(fftself2))#,vmin=0, vmax=vmax)
    grid[0].set_title('Self written function')
    im = grid[1].imshow(np.abs(fftnumpy2))#,vmin=0, vmax=vmax)
    grid[1].set_title('Numpy function')
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
            cax.toggle_label(True)
    fig.suptitle('2D Fourier transform of $cos(x)$',x=0.38,y=0.58,fontsize=14)
    fig.tight_layout()
    plt.savefig('./plots/5f.png',bbox_inches='tight',pad_inches = 0.5)
    plt.close()

    g3D = lambda x,y,z,mu,sig: 1/(sig*(2*np.pi)**0.5)*np.exp((-(x-mu)**2-(y-mu)**2-(z-mu)**2)/sig**2) 
    N = 16
    x = np.linspace(-3,3,N)
    y = np.linspace(-3,3,N)
    z = np.linspace(-3,3,N)
    f3d = np.zeros((N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                f3d[i][j][k] = g3D(x[i],y[j],z[k],0,1)

    fft_f3d = np.abs(fft3D(f3d))         
                
    vmax = np.max(fft_f3d)
    fig = plt.figure(1,(30,30))
    grid = AxesGrid(fig, 142,
                    nrows_ncols=(1, 3),
                    axes_pad=(0.15,0.45),
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    im = grid[0].imshow(fft_f3d[:,:,7])
    grid[0].set_title('xy')
    im = grid[1].imshow(fft_f3d[:,7,:])
    grid[1].set_title('xz')
    im = grid[2].imshow(fft_f3d[7,:,:])
    grid[2].set_title('yz')
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
            cax.toggle_label(True)
    fig.suptitle('Three centered slices of FFT of a 3D $G(\mu,\sigma)$',x=0.38,y=0.56,fontsize=14)
    fig.tight_layout()
    plt.savefig('./plots/5g.png',bbox_inches='tight',pad_inches = 0.5)
    plt.close()

    # --- 5.f --- 
    # Calculating the gravitational potential
    mean = np.mean(mesh)
    mesh_n = (mesh-mean)/mean
    fft_mesh = fft3D(mesh_n)/N**3
    N = 16
    for l in range(N):
        if l <= (N//2):
            k_z = (l)*2*np.pi/N
        else:
            k_z = (-N+l)*2*np.pi/N
        for j in range(N):
            if i <= (N//2):
                k_y = (j)*2*np.pi/N
            else:
                k_y = (-N+j)*2*np.pi/N
            for i in range(N):
                if i <= (N//2):
                    k_x = (i)*2*np.pi/N
                else:
                    k_x = (-N+i)*2*np.pi/N
                # Calculating k     
                k = (k_x**2+k_y**2+k_z**2)**0.5
                if k == 0:
                    k = 1
                fft_mesh[i][j][l] = fft_mesh[i][j][l]*k**(-2)       
    grav_p = fft3D(fft_mesh,inv=True)
    grav_p = np.abs(grav_p)

    vmax = np.max(grav_p)
    fig = plt.figure(1,(30,30))
    grid = AxesGrid(fig, 142,
                    nrows_ncols=(2, 2),
                    axes_pad=(0.15,0.45),
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    im = grid[0].imshow(grav_p[:,:,0])#,vmin=0, vmax=vmax)
    grid[0].set_title('z = 4')
    im = grid[1].imshow(grav_p[:,:,8])#,vmin=0, vmax=vmax)
    grid[1].set_title('z = 9')
    im = grid[2].imshow(grav_p[:,:,10])#,vmin=0, vmax=vmax)
    grid[2].set_title('z = 11')
    im = grid[3].imshow(grav_p[:,:,13])#,vmin=0, vmax=vmax)
    grid[3].set_title('z = 14')
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
            cax.toggle_label(True)
    fig.suptitle('Four different z-slices of a mesh produced using CiC',x=0.38,y=0.64,fontsize=14)
    fig.tight_layout()
    plt.savefig('./plots/5h.png',bbox_inches='tight',pad_inches = 0.5)
    plt.close()

    fig = plt.figure(1,(30,30))
    grid = AxesGrid(fig, 142,
                    nrows_ncols=(1, 2),
                    axes_pad=(0.15,0.45),
                    share_all=True,
                    label_mode="L",
                    cbar_location="right",
                    cbar_mode="single",
                    )

    im = grid[0].imshow(grav_p[:,7,:])#,vmin=0, vmax=vmax)
    grid[0].set_title('xz')
    im = grid[1].imshow(grav_p[7,:,:])#,vmin=0, vmax=vmax)
    grid[1].set_title('yz')
    grid.cbar_axes[0].colorbar(im)
    for cax in grid.cbar_axes:
            cax.toggle_label(True)
    fig.suptitle('Two centered slices calculated gravitational potential',x=0.38,y=0.57,fontsize=14)
    fig.tight_layout()
    plt.savefig('./plots/5i.png',bbox_inches='tight',pad_inches = 0.5)
    plt.close()

    # --- 5.g ---
    # Calculating the gradients for the 3 dimensions
    gradx,grady,gradz = central_diff_3D(grav_p)
    # Calculating the potential for each position
    positions_grad = CiC_reverse(positions[:,:10],gradx,grady,gradz,16)
    print('Potential gradient output:')
    print('[x,y,z]')
    for i in range(len(positions_grad[0])):
        print(positions_grad[:,i])