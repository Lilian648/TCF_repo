#######################################################################################
# Class for producing box containing randomly located disks/bubbles of a given radius #
#######################################################################################
# Handles periodic boundary conditions
#
# Written for 2D by Jonathan Pritchard (2017) - upgraded to 3D by Adélie Gorce (2018)
#

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, pickle

class RandomBubbles:
    """
    Produce a random box filled with bubbles in 3D or disks in 2D of given size to reach a target filling fraction
    """

    def __init__(self, DIM=256, fillingfraction = 0.3, radius=20., NDIM = 2, nooverlap = False, periodic=True):
        """
        Initialise a DIM ^ NDIM box with random bubbles until given number of bubbles (nb) has been reached
        If 'periodic = True', the box had periodic boundary conditions
        If 'nooverlap = True', bubbles are not allowed to overlap (expect a long computing time if you have a large filling fraction)
        """

        self.NDIM = NDIM  #number of dimensions e.g. 2D or 3D
        self.DIM = DIM   #Number of cells per dimension
        self.nooverlap = nooverlap #Overlapping bubbles allowed or not 
        self.periodic = periodic #Periodic boundary conditions
        self.fillingfraction=fillingfraction #target filling fraction for the box
        self.radius=np.ceil(radius) #Radius (in pixels) of all bubbles in the box xxx

        if not isinstance(radius, int):
            print(f"Warning: radius must be an integer. Rounding {radius} up to {np.ceil(radius)}.")

        print("Initialising a %iD box with %i cells" %(self.NDIM, self.DIM))
        print("Target filling fraction: %.2f with bubbles of radius %i" %(self.fillingfraction,self.radius))
        print("Overlap and periodicity are (%s,%s)" %(np.bool(~nooverlap),periodic))
        print(' ')

        if NDIM == 2:
            self.box = np.zeros([DIM, DIM])
        elif NDIM ==3:
            self.box = np.zeros([DIM, DIM, DIM])
        else:
            raise Exception("NDIM must be 2 or 3" % (NDIM))

        self.bubbles = []

        #Add bubbles to get to target filling fraction
        self.increase_filling_fraction()


    def summary(self):
        """
        Update summary statistics
        """

        self.box[self.box>1.]=1. #avoid pixel value to exceed 1 in overlap zones (pixel value <-> ionisation level)
        
        #Show the slice
        colours=['midnightblue','lightyellow']
        bounds=[0,1]
        cmap = mpl.colors.ListedColormap(colours)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        self.nb=len(self.bubbles)
        self.xhII=self.box.mean()

        # save object in pickle file
        filename='Field_R%i_xhII%.2f_N%i_%iD.pkl' %(self.radius,self.fillingfraction,self.DIM,self.NDIM)
        with open(filename,'wb') as f: 
            pickle.dump(self,f,pickle.HIGHEST_PROTOCOL) 

        # fig = plt.figure(1, (8., 8.))
        # if (self.NDIM==2):
        #     plt.imshow(self.box,cmap=cmap)
        # elif (self.NDIM==3):
        #     plt.imshow(self.box[:,:,self.DIM//2],cmap=cmap)
        # plt.tight_layout()
        # plt.tick_params(axis='both',which='both', bottom=False, left=False,labelbottom=False,labelleft=False)
        # plt.savefig('Bubble_box_xhII%.2f_nooverlap=%s_radius=%i_N%i_%iD.png' %(np.mean(self.box),self.nooverlap,self.radius,self.DIM,self.NDIM))


    def increase_filling_fraction(self):
        """
        Add randomly located bubbles of radius R until reach
        desired filling fraction
        """
        R=self.radius
        if (self.box.mean() > self.fillingfraction):
            print("Box already more ionised than target")
            return
        
        count=10
        while (self.box.mean() < self.fillingfraction):

            #random position for the bubble
            x = npr.randint(self.DIM, size = self.NDIM)

            #Use mask to ID pixels in this ellipse
            bmask = self.bubble_mask(x, R)
            if self.nooverlap:
                if np.any(self.box[bmask.astype(np.bool)]):
                    continue
            #add bubble to whole box
            self.box=np.add(self.box,bmask).astype(np.bool)
            self.box=self.box.astype(int)
            #Store bubbles so can assemble true size PDF
            self.bubbles.append(x)

            if (np.mean(self.box)/self.fillingfraction*100 > count):
                self.loading_verbose(count)
                count=count+10

        self.summary()


    def grow_bubbles(self,R):

        if (R-self.radius<1):
            raise Error('Targer radius smaller than current radius')

        #grow bubbles one by one
        for u,x in enumerate(self.bubbles):
            # draws larger bubble around existing one
            bmask = self.bubble_mask(x, R)
            #add bubble to whole box
            self.box=np.add(self.box,bmask).astype(np.bool)
            self.box=self.box.astype(int)
            sys.stdout.write('\r Growing bubbles... %i%% done' %(u/len(self.bubbles)*100))
            sys.stdout.flush()

        sys.stdout.write('\r Growing bubbles... 100% done\n')
        self.summary() #self.summary(R) <- wan't working for some reason
        
    def bubble_mask(self, x, R):
        #wrapper to handle different dimensionality
        if (self.NDIM == 2):
            return self.disk_mask(x,R)
        elif(self.NDIM == 3):
            return self.sphere_mask(x,R)
        else:
            raise Exception ("NDIM is not 2 or 3")

    def disk_mask(self, pos, R):
        #generates mask corresponding to a 2D ionised disk
        #pos is coordinates of the centre of the bubble
        #R is its radius

        full_struct = np.zeros([self.DIM, self.DIM])

        #Creates a disk at centre of smaller structure to avoid generating another whole box: just enough to contain the disk
        structsize = int(2*R+6)
        #structsize = int(2 * np.ceil(R) + 6)  # Ensure smooth radius changes

        x0 = y0 = int(structsize/2)
        struct = np.zeros((structsize, structsize))
        x, y = np.indices((structsize, structsize))
        mask = (x - structsize/2)**2 + (y - structsize/2)**2 <= R**2 #puts the disk in the middle of new box
        struct[mask] = 1
        
        #Now work out coordinate shift to move centre to pos    
        xmov = [pos[0] - x0, pos[0] + x0]
        ymov = [pos[1] - y0, pos[1] + y0]

        #if struct goes out of the box
        xmin=max(xmov[0],0)
        xmax=min(xmov[1],self.DIM)
        ymin=max(ymov[0],0)
        ymax=min(ymov[1],self.DIM)

        #periodic boundary conditions
        if self.periodic:
            if (xmov[0]<0):
                extra_struct=struct[0:abs(xmov[0]),abs(min(0,ymov[0])):min(structsize,self.DIM-ymov[0])]
                full_struct[self.DIM-abs(xmov[0]):self.DIM,ymin:ymax]=np.add(full_struct[self.DIM-abs(xmov[0]):self.DIM,ymin:ymax],extra_struct)
            if (xmov[1]>self.DIM):
                extra_struct=struct[structsize-(xmov[1]-self.DIM):structsize,abs(min(0,ymov[0])):min(structsize,structsize+self.DIM-ymov[1])]
                full_struct[0:xmov[1]-self.DIM,ymin:ymax]=np.add(full_struct[0:xmov[1]-self.DIM,ymin:ymax],extra_struct)
            if (ymov[0]<0):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,self.DIM-xmov[0]),0:abs(ymov[0])]
                full_struct[xmin:xmax,self.DIM-abs(ymov[0]):self.DIM]=np.add(full_struct[xmin:xmax,self.DIM-abs(ymov[0]):self.DIM],extra_struct)
            if (ymov[1]>self.DIM):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,structsize+self.DIM-xmov[1]),structsize-(ymov[1]-self.DIM):structsize]
                full_struct[xmin:xmax,0:ymov[1]-self.DIM]=np.add(full_struct[xmin:xmax,0:ymov[1]-self.DIM],extra_struct)

        #truncated struct if some part is outside the full struct
        small_struct=struct[abs(xmov[0]-xmin):structsize-abs(xmov[1]-xmax), abs(ymov[0]-ymin):structsize-abs(ymov[1]-ymax)]

        #add to previous box 
        full_struct[xmin:xmax,ymin:ymax] = np.add(full_struct[xmin:xmax,ymin:ymax],small_struct) 
        
        return full_struct


    def sphere_mask(self, pos, R):
        #generates mask corresponding to a 3D ionised sphere
        #pos is coordinates of the centre of the bubble
        #R is its radius

        full_struct = np.zeros([self.DIM, self.DIM, self.DIM])

        #Creates a disk at centre of smaller structure to avoid generating another whole box: just enough to contain the disk
        structsize = int(2*R+6)
        #structsize = int(2 * np.ceil(R) + 6)  # Avoid integer jumps at fractional R

        x0 = y0 = z0 = int(structsize/2)
        struct = np.zeros((structsize, structsize, structsize))
        x, y, z = np.indices((structsize, structsize, structsize))
        mask = (x - structsize/2)**2 + (y - structsize/2)**2 + (z - structsize/2)**2<= R**2
        struct[mask] = 1

        #Now work out coordinate shift to move centre to pos
        xmov = [pos[0] - x0,pos[0]+x0]
        ymov = [pos[1] - y0,pos[1]+y0]
        zmov = [pos[2] - z0,pos[2]+z0]

        #if struct goes out of the box
        xmin=max(xmov[0],0)
        xmax=min(xmov[1],self.DIM)
        ymin=max(ymov[0],0)
        ymax=min(ymov[1],self.DIM)
        zmin=max(zmov[0],0)
        zmax=min(zmov[1],self.DIM)

        #periodic boundary conditions
        if self.periodic:
            if (xmov[0]<0):
                extra_struct=struct[0:abs(xmov[0]),abs(min(0,ymov[0])):min(structsize,self.DIM-ymov[0]),abs(min(0,zmov[0])):min(structsize,self.DIM-zmov[0])]
                full_struct[self.DIM-abs(xmov[0]):self.DIM,ymin:ymax,zmin:zmax]=np.add(full_struct[self.DIM-abs(xmov[0]):self.DIM,ymin:ymax,zmin:zmax],extra_struct)
            if (xmov[1]>self.DIM):
                extra_struct=struct[structsize-(xmov[1]-self.DIM):structsize,abs(min(0,ymov[0])):min(structsize,structsize+self.DIM-ymov[1]),abs(min(0,zmov[0])):min(structsize,structsize+self.DIM-zmov[1])]
                full_struct[0:xmov[1]-self.DIM,ymin:ymax,zmin:zmax]=np.add(full_struct[0:xmov[1]-self.DIM,ymin:ymax,zmin:zmax],extra_struct)
            if (ymov[0]<0):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,self.DIM-xmov[0]),0:abs(ymov[0]),abs(min(0,zmov[0])):min(structsize,self.DIM-zmov[0])]
                full_struct[xmin:xmax,self.DIM-abs(ymov[0]):self.DIM,zmin:zmax]=np.add(full_struct[xmin:xmax,self.DIM-abs(ymov[0]):self.DIM,zmin:zmax],extra_struct)
            if (ymov[1]>self.DIM):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,structsize+self.DIM-xmov[1]),structsize-(ymov[1]-self.DIM):structsize,abs(min(0,zmov[0])):min(structsize,structsize+self.DIM-zmov[1])]
                full_struct[xmin:xmax,0:ymov[1]-self.DIM,zmin:zmax]=np.add(full_struct[xmin:xmax,0:ymov[1]-self.DIM,zmin:zmax],extra_struct)
            if (zmov[0]<0):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,self.DIM-xmov[0]),abs(min(0,ymov[0])):min(structsize,self.DIM-ymov[0]),0:abs(zmov[0])]
                full_struct[xmin:xmax,ymin:ymax,self.DIM-abs(zmov[0]):self.DIM]=np.add(full_struct[xmin:xmax,ymin:ymax,self.DIM-abs(zmov[0]):self.DIM],extra_struct)
            if (zmov[1]>self.DIM):
                extra_struct=struct[abs(min(0,xmov[0])):min(structsize,structsize+self.DIM-xmov[1]),abs(min(0,ymov[0])):min(structsize,structsize+self.DIM-ymov[1]),structsize-(zmov[1]-self.DIM):structsize]
                full_struct[xmin:xmax,ymin:ymax,0:zmov[1]-self.DIM]=np.add(full_struct[xmin:xmax,ymin:ymax,0:zmov[1]-self.DIM],extra_struct)

        #truncated struct if some part is outside the full struct
        small_struct=struct[abs(xmov[0]-xmin):structsize-abs(xmov[1]-xmax), abs(ymov[0]-ymin):structsize-abs(ymov[1]-ymax),abs(zmov[0]-zmin):structsize-abs(zmov[1]-zmax)] #truncated struct if some part is outside the full struct
        #add to full box
        full_struct[xmin:xmax,ymin:ymax,zmin:zmax] = np.add(full_struct[xmin:xmax,ymin:ymax,zmin:zmax],small_struct) #add to previous box in case some intermediate structures overlap

        return full_struct


    def write_ionisation_field(self):
        """ Writes ionisation_field in textfile to be read by TCF algoithm"""

        filechain='Field_%ibinary_bubbles_nooverlap=%s_radius=%i_xhII%.2f_N%i_%iD.txt' %(self.nb,self.nooverlap,self.radius,np.mean(self.box),self.DIM,self.NDIM)
        filechain2='Field_%ibinary_bubbles_nooverlap=%s_radius=%i_xhII%.2f_N%i_%iD_bubble_loc.txt' %(self.nb,self.nooverlap,self.radius,np.mean(self.box),self.DIM,self.NDIM)
        field=self.box
        if self.NDIM==3:
            out=field[:,:,0]
            for i in range(1,self.DIM):
                out=np.r_[out,field[:,:,i]]
            np.savetxt(filechain,out,delimiter=' ',fmt='%-10.4f')
            np.savetxt(filechain2,np.array(self.bubbles),delimiter=' ',fmt='%i')
            print("The filename is:", filechain)

        elif self.NDIM==2:
            np.savetxt(filechain,field,delimiter=' ',fmt='%-10.4f')
            np.savetxt(filechain2,np.array(self.bubbles),delimiter=' ',fmt='%i')
            print("The filename is:", filechain)


    def print_filechain(self):
        """ Returns out filename for future usage"""

        filechain='Field_%ibinary_bubbles_nooverlap=%s_radius=%i_xhII%.2f_N%i_%iD' %(self.nb,self.nooverlap,self.radius,np.mean(self.box),self.DIM,self.NDIM)

        return filechain


    def loading_verbose(self,perc):
        msg = str('%i bubbles, f = %.4f, %i%% done' %(len(self.bubbles),self.box.mean(),perc))
        if (perc>=99.9):
            msg = str('f = %.4f, %i bubbles, 100%% done.\n' %(self.box.mean(),len(self.bubbles)))
        sys.stdout.write('\r'+msg)
        sys.stdout.flush()


