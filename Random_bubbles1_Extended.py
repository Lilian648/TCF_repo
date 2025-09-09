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

    def __init__(
        self,
        DIM=256,
        NDIM=2,
        fillingfraction=0.3,
        params=[20., 0.],
        radius_distribution=0.,
        gaussian_profile=False,
        nooverlap=False,
        periodic=True,
        verbose=False,
        save=False,
    ):
        """
        Parameters
        ----------
            DIM: float
                Number of pixels on each side of your simulation box.
                Default is 512.
            NDIM: int
                Number of dimensions of your simulation box.
                Options are 2 or 3.
                Default is 2.
            fillingfraction: float
                Target filling fraction (average) of the simulation.
                The resulting filling fraction might be slightly different
                from target.
                Computing time increases with filling fraction.
                Default is 0.3.
            params: tuple of 2 floats
                Parameters describing the radius distribution.
                (R, sig) for R the mean radius and sigma the 
                standard deviation. For Dirac distribution, set
                sig=0. Radius must be < DIM/2.
                Default is (20, 0).
            radius_distribution: 0 or 1 or 2
                Integer describing the type of bubble size distribution
                to use to fill the box.
                0 is Dirac (all bubbles have same radius).
                1 is Gaussian with mean R and sigma sig.
                2 is Log-normal with mean R and sigma sig.
                Default is 0.
            gaussian_profile: boolean
                Whether the bubbles have a Gaussian or a step function profile.
                For Gaussian bubbles, the radius is the standard deviation.
                Default is False.
            nooverlap: boolean
                Whether bubbles are allowed to overlap or not.
                Default is False (there is overlap).
            periodic: boolean
                Whether the box has periodic boundary conditions or not.
                Default: True
            verbose: boolean
                Whether to output log.
                Default: False.
            save: boolean
                Whether to save the field and bubble locations to file.
                Default: False.
        """
        # checks
        assert (NDIM == 2) or (NDIM == 3), "NDIM must be 2 or 3"
        if (fillingfraction > 1.) or (fillingfraction < 0.):
            raise ValueError('Filling fraction must be between 0 and 1 (excluded).')

        # initialise attributes
        self.NDIM = int(NDIM)
        self.DIM = int(DIM)
        self.nooverlap = bool(nooverlap)
        self.periodic = bool(periodic)
        self.fillingfraction = float(fillingfraction)
        self.gaussian_profile = bool(gaussian_profile)

        # radius distribution
        self.distribution = float(radius_distribution)
        if self.distribution == 0.:
            self.mean_radius = params[0]
            self.sigma_radius = 0.
        elif self.distribution in [1., 2.]:
            self.mean_radius = params[0]
            self.sigma_radius = params[1]
        else:
            raise ValueError('radius_distribution must be 0, 1, or 2.')
        if (self.mean_radius > DIM / 2.) or (self.sigma_radius > DIM / 2.):
            raise ValueError('Radius larger than half of the cell number.')

        # initialise box
        self.bubble_centres = []
        self.bubble_radii = []
        self.box = np.zeros([DIM for i in range(NDIM)])

        self.verbose = verbose
        self.save = save

        if self.verbose:
            print("Initialising a %iD box with %i cells" % (self.NDIM, self.DIM))
            print(
                "Target filling fraction: %.2f with bubbles of radius %i"
                % (self.fillingfraction, self.radius)
            )
            print(
                "Overlap and periodicity are (%s,%s)" % (not np.bool(nooverlap), self.periodic)
            )
            print(" ")

        # Add bubbles to get to target filling fraction
        self.increase_filling_fraction()

    def summary(self, show_plot=False):
        """
        Update summary statistics
        """

        ###### orignal code ########
        # self.box[
        #     self.box > 1.0
        # ] = 1.0  # avoid pixel value to exceed 1 in overlap zones (pixel value <-> ionisation level)

        # # Show the slice
        # cmap = mpl.colormaps['magma']

        # self.nb = len(self.bubble_radii)
        # self.xhII = self.box.mean()

        # fig, ax = plt.subplots(1, 1, figsize=(8.0, 8.0))
        # if self.NDIM == 2:
        #     ax.imshow(self.box, cmap=cmap)
        # elif self.NDIM == 3:
        #     ax.imshow(self.box[:, :, self.DIM // 2], cmap=cmap)
        # plt.tick_params(
        #     axis="both",
        #     which="both",
        #     bottom=False,
        #     left=False,
        #     labelbottom=False,
        #     labelleft=False,
        # )
        # fig.tight_layout()

        ####### changed version, to only plot figure is show_plot=True #########
        self.box[self.box > 1.0] = 1.0  # Avoid pixel value > 1 in overlap zones

        self.nb = len(self.bubble_radii)
        self.xhII = self.box.mean()
    
        if show_plot:
            # Only show the plot if show_plot is True
            cmap = mpl.colormaps['magma']
            fig, ax = plt.subplots(1, 1, figsize=(8.0, 8.0))
            if self.NDIM == 2:
                ax.imshow(self.box, cmap=cmap)
            elif self.NDIM == 3:
                ax.imshow(self.box[:, :, self.DIM // 2], cmap=cmap)
            plt.tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            fig.tight_layout()

            # Save the figure if save is True
            if self.save:
                fig.savefig(
                    "Bubble_box_xhII%.2f_nooverlap=%s_radius=%i_N%i_%iD.png"
                    % (np.mean(self.box), self.nooverlap, self.mean_radius, self.DIM, self.NDIM)
                )

        #### OLD VERSION ##################################################################
        # if self.save:
        #     # Save object in pickle file
        #     filename = "Field_R%i_xhII%.2f_N%i_%iD.pkl" % (
        #         self.mean_radius,
        #         self.fillingfraction,
        #         self.DIM,
        #         self.NDIM,
        #     )
        ###################################################################################

        #################################### Code to save as a Pickle file ###############################################

        # if self.save:
        #     # Save object in pickle file
        #     filename = "Field_DIM%i_FF%.2f_R%i_S%.2f_N%i_nooverlap%s_periodic%s_RD%i_GP%s.pkl" % (
        #         self.DIM,
        #         self.fillingfraction,
        #         self.mean_radius,         # Assuming radius is stored as mean_radius
        #         self.sigma_radius,        # Assuming sigma is stored as sigma
        #         self.NDIM,
        #         self.nooverlap,           # Assuming nooverlap is boolean, will display as True/False
        #         self.periodic,            # Assuming periodic is boolean, will display as True/False
        #         self.distribution,
        #         self.gaussian_profile     # Assuming gaussian_profile is boolean, will display as True/False
        #     )

        #     with open(filename, "wb") as f:
        #         pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        ###################################################################################

        ################################### Code to save as a txt file ################################################


        if self.save:
            # Define filename with .txt extension
            filename = "Field_DIM%i_FF%.2f_R%i_S%.2f_N%i_nooverlap%s_periodic%s_RD%i_GP%s.txt" % (
                self.DIM,
                self.fillingfraction,
                self.mean_radius,         
                self.sigma_radius,        
                self.NDIM,
                self.nooverlap,           
                self.periodic,            
                self.distribution,
                self.gaussian_profile     
            )
        
            # Save the 2D array to a .txt file
            np.savetxt(filename, self.box, fmt="%.6f")  # Adjust format precision if needed

        ###################################################################################
        

        ########################################################################

        # if self.save:
        #     fig.savefig(
        #         "Bubble_box_xhII%.2f_nooverlap=%s_radius=%i_N%i_%iD.png"
        #         % (np.mean(self.box), self.nooverlap, self.mean_radius, self.DIM, self.NDIM)
        #     )
        #     # save object in pickle file
        #     filename = "Field_R%i_xhII%.2f_N%i_%iD.pkl" % (
        #         self.mean_radius,
        #         self.fillingfraction,
        #         self.DIM,
        #         self.NDIM,
        #     )
        #     with open(filename, "wb") as f:
        #         pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def increase_filling_fraction(self):
        """
        Add randomly located bubbles of radius R until reach
        desired filling fraction
        """
        if self.box.mean() > self.fillingfraction:
            print("Box already more ionised than target")
            return

        count = 10
        while self.box.mean() < self.fillingfraction:
            # random position for the bubble
            x = npr.randint(self.DIM, size=self.NDIM)
            # radius size
            if self.distribution == 0.:
                R = self.mean_radius
            elif self.distribution == 1.:
                R = npr.normal(self.mean_radius, self.sigma_radius, 1)
            else:
                R = npr.lognormal(
                    np.log(self.mean_radius/np.sqrt(1+self.sigma_radius**2/self.mean_radius**2)),
                    np.sqrt( np.log(1+self.sigma_radius**2/self.mean_radius**2) ),
                    1
                )
            if R <= 1.:
                R = 1
                bmask = np.zeros(self.box.shape)
                bmask[x] = 1.
            if R > self.DIM/2.:
                continue
            else:
                # Use mask to ID pixels in this ellipse
                bmask = self.bubble_mask(x, int(R))
            # check overlaps
            if self.nooverlap and np.any(self.box[bmask.astype(bool)]):###(np.bool)]):
                continue
            # add bubble to whole box
            self.box = np.add(self.box, bmask)
            # self.box = self.box.astype(int)
            # Store bubbles so can assemble true size PDF
            self.bubble_centres.append(x)
            self.bubble_radii.append(int(R))

            if self.verbose and (
                np.mean(self.box) / self.fillingfraction * 100 > count
            ):
                self.loading_verbose(count)
                count = count + 10

        self.summary()

    def grow_bubbles(self, R):
        if R - self.radius < 1:
            raise ValueError("Target radius smaller than current radius")

        # grow bubbles one by one
        for u, x in enumerate(self.bubble_centres):
            # draws larger bubble around existing one
            bmask = self.bubble_mask(x, R)
            # add bubble to whole box
            self.box = np.add(self.box, bmask).astype(np.bool)
            self.box = self.box.astype(int)
            sys.stdout.write(
                "\r Growing bubbles... %i%% done" % (u / len(self.bubble_centres) * 100)
            )
            sys.stdout.flush()

        sys.stdout.write("\r Growing bubbles... 100% done\n")
        self.summary(R)

    def bubble_mask(self, x, R):
        # wrapper to handle different dimensionality
        if self.NDIM == 2:
            return self.disk_mask(x, R)
        elif self.NDIM == 3:
            return self.sphere_mask(x, R)

    def disk_mask(self, pos, R):
        # generates mask corresponding to a 2D ionised disk
        # pos is coordinates of the centre of the bubble
        # R is its radius

        full_struct = np.zeros([self.DIM, self.DIM])

        # Creates a disk at centre of smaller structure to avoid generating 
        # another whole box: just enough to contain the disk
        if self.gaussian_profile:
            structsize = int(6 * R)
            x = np.arange(0, structsize, 1)
            y = x[:, np.newaxis]
            x0 = y0 = int(structsize/2)
            struct = np.exp( -1 * ( (x-x0)**2 + (y-y0)**2 ) / (2*R**2) )
            # puts the disk in the middle of new box
        else:
            structsize = int(2 * R + 6)
            x0 = y0 = int(structsize / 2)
            struct = np.zeros((structsize, structsize))
            x, y = np.indices((structsize, structsize))
            mask = (x - structsize / 2) ** 2 + (
                y - structsize / 2
            ) ** 2 <= R**2  # puts the disk in the middle of new box
            struct[mask] = 1

        # Now work out coordinate shift to move centre to pos
        xmov = [pos[0] - x0, pos[0] + x0]
        ymov = [pos[1] - y0, pos[1] + y0]

        # if struct goes out of the box
        xmin = max(xmov[0], 0)
        xmax = min(xmov[1], self.DIM)
        ymin = max(ymov[0], 0)
        ymax = min(ymov[1], self.DIM)

        # periodic boundary conditions
        if self.periodic:
            if xmov[0] < 0:
                extra_struct = struct[
                    0 : abs(xmov[0]),
                    abs(min(0, ymov[0])) : min(structsize, self.DIM - ymov[0]),
                ]
                full_struct[self.DIM - abs(xmov[0]) : self.DIM, ymin:ymax] = np.add(
                    full_struct[self.DIM - abs(xmov[0]) : self.DIM, ymin:ymax],
                    extra_struct,
                )
            if xmov[1] > self.DIM:
                extra_struct = struct[
                    structsize - (xmov[1] - self.DIM) : structsize,
                    abs(min(0, ymov[0])) : min(
                        structsize, structsize + self.DIM - ymov[1]
                    ),
                ]
                full_struct[0 : xmov[1] - self.DIM, ymin:ymax] = np.add(
                    full_struct[0 : xmov[1] - self.DIM, ymin:ymax], extra_struct
                )
            if ymov[0] < 0:
                extra_struct = struct[
                    abs(min(0, xmov[0])) : min(structsize, self.DIM - xmov[0]),
                    0 : abs(ymov[0]),
                ]
                full_struct[xmin:xmax, self.DIM - abs(ymov[0]) : self.DIM] = np.add(
                    full_struct[xmin:xmax, self.DIM - abs(ymov[0]) : self.DIM],
                    extra_struct,
                )
            if ymov[1] > self.DIM:
                extra_struct = struct[
                    abs(min(0, xmov[0])) : min(
                        structsize, structsize + self.DIM - xmov[1]
                    ),
                    structsize - (ymov[1] - self.DIM) : structsize,
                ]
                full_struct[xmin:xmax, 0 : ymov[1] - self.DIM] = np.add(
                    full_struct[xmin:xmax, 0 : ymov[1] - self.DIM], extra_struct
                )

        # truncated struct if some part is outside the full struct
        small_struct = struct[
            abs(xmov[0] - xmin): structsize - abs(xmov[1] - xmax),
            abs(ymov[0] - ymin): structsize - abs(ymov[1] - ymax),
        ]
        # add to previous box
        full_struct[xmin:xmax, ymin:ymax] = np.add(
            full_struct[xmin:xmax, ymin:ymax], small_struct
        )

        return full_struct

    def sphere_mask(self, pos, R):
        # generates mask corresponding to a 3D ionised sphere
        # pos is coordinates of the centre of the bubble
        # R is its radius

        full_struct = np.zeros([self.DIM, self.DIM, self.DIM])

        # Creates a disk at centre of smaller structure to avoid generating
        # another whole box: just enough to contain the disk
        if self.gaussian_profile:
            structsize = int(6 * R)
            x = np.arange(0, structsize, 1)
            y = x[:, np.newaxis]
            z = y[:, :, np.newaxis]
            x0 = y0 = z0 = int(structsize/2)
            struct = np.exp( -1* ( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 ) / (2*R**2) )
            # puts the disk in the middle of new box
        else:
            structsize = int(2 * R + 6)
            x0 = y0 = z0 = int(structsize / 2)
            struct = np.zeros((structsize, structsize, structsize))
            x, y, z = np.indices((structsize, structsize, structsize))
            mask = (x - structsize / 2) ** 2 + (y - structsize / 2) ** 2 + (
                z - structsize / 2
            ) ** 2 <= R**2
            struct[mask] = 1

        # Now work out coordinate shift to move centre to pos
        xmov = [pos[0] - x0, pos[0] + x0]
        ymov = [pos[1] - y0, pos[1] + y0]
        zmov = [pos[2] - z0, pos[2] + z0]

        # if struct goes out of the box
        xmin = max(xmov[0], 0)
        xmax = min(xmov[1], self.DIM)
        ymin = max(ymov[0], 0)
        ymax = min(ymov[1], self.DIM)
        zmin = max(zmov[0], 0)
        zmax = min(zmov[1], self.DIM)

        # periodic boundary conditions
        if self.periodic:
            if xmov[0] < 0:
                extra_struct = struct[
                    0 : abs(xmov[0]),
                    abs(min(0, ymov[0])) : min(structsize, self.DIM - ymov[0]),
                    abs(min(0, zmov[0])) : min(structsize, self.DIM - zmov[0]),
                ]
                full_struct[
                    self.DIM - abs(xmov[0]) : self.DIM, ymin:ymax, zmin:zmax
                ] = np.add(
                    full_struct[
                        self.DIM - abs(xmov[0]) : self.DIM, ymin:ymax, zmin:zmax
                    ],
                    extra_struct,
                )
            if xmov[1] > self.DIM:
                extra_struct = struct[
                    structsize - (xmov[1] - self.DIM) : structsize,
                    abs(min(0, ymov[0])) : min(
                        structsize, structsize + self.DIM - ymov[1]
                    ),
                    abs(min(0, zmov[0])) : min(
                        structsize, structsize + self.DIM - zmov[1]
                    ),
                ]
                full_struct[0 : xmov[1] - self.DIM, ymin:ymax, zmin:zmax] = np.add(
                    full_struct[0 : xmov[1] - self.DIM, ymin:ymax, zmin:zmax],
                    extra_struct,
                )
            if ymov[0] < 0:
                extra_struct = struct[
                    abs(min(0, xmov[0])) : min(structsize, self.DIM - xmov[0]),
                    0 : abs(ymov[0]),
                    abs(min(0, zmov[0])) : min(structsize, self.DIM - zmov[0]),
                ]
                full_struct[
                    xmin:xmax, self.DIM - abs(ymov[0]) : self.DIM, zmin:zmax
                ] = np.add(
                    full_struct[
                        xmin:xmax, self.DIM - abs(ymov[0]) : self.DIM, zmin:zmax
                    ],
                    extra_struct,
                )
            if ymov[1] > self.DIM:
                extra_struct = struct[
                    abs(min(0, xmov[0])) : min(
                        structsize, structsize + self.DIM - xmov[1]
                    ),
                    structsize - (ymov[1] - self.DIM) : structsize,
                    abs(min(0, zmov[0])) : min(
                        structsize, structsize + self.DIM - zmov[1]
                    ),
                ]
                full_struct[xmin:xmax, 0 : ymov[1] - self.DIM, zmin:zmax] = np.add(
                    full_struct[xmin:xmax, 0 : ymov[1] - self.DIM, zmin:zmax],
                    extra_struct,
                )
            if zmov[0] < 0:
                extra_struct = struct[
                    abs(min(0, xmov[0])) : min(structsize, self.DIM - xmov[0]),
                    abs(min(0, ymov[0])) : min(structsize, self.DIM - ymov[0]),
                    0 : abs(zmov[0]),
                ]
                full_struct[
                    xmin:xmax, ymin:ymax, self.DIM - abs(zmov[0]) : self.DIM
                ] = np.add(
                    full_struct[
                        xmin:xmax, ymin:ymax, self.DIM - abs(zmov[0]) : self.DIM
                    ],
                    extra_struct,
                )
            if zmov[1] > self.DIM:
                extra_struct = struct[
                    abs(min(0, xmov[0])) : min(
                        structsize, structsize + self.DIM - xmov[1]
                    ),
                    abs(min(0, ymov[0])) : min(
                        structsize, structsize + self.DIM - ymov[1]
                    ),
                    structsize - (zmov[1] - self.DIM) : structsize,
                ]
                full_struct[xmin:xmax, ymin:ymax, 0 : zmov[1] - self.DIM] = np.add(
                    full_struct[xmin:xmax, ymin:ymax, 0 : zmov[1] - self.DIM],
                    extra_struct,
                )

        # truncated struct if some part is outside the full struct
        small_struct = struct[
            abs(xmov[0] - xmin) : structsize - abs(xmov[1] - xmax),
            abs(ymov[0] - ymin) : structsize - abs(ymov[1] - ymax),
            abs(zmov[0] - zmin) : structsize - abs(zmov[1] - zmax),
        ]  # truncated struct if some part is outside the full struct
        # add to full box
        full_struct[xmin:xmax, ymin:ymax, zmin:zmax] = np.add(
            full_struct[xmin:xmax, ymin:ymax, zmin:zmax], small_struct
        )  # add to previous box in case some intermediate structures overlap

        return full_struct

    def loading_verbose(self, perc):
        msg = str(
            "%i bubbles, f = %.4f, %i%% done"
            % (len(self.bubble_centres), self.box.mean(), perc)
        )
        if perc >= 99.9:
            msg = str(
                "f = %.4f, %i bubbles, 100%% done.\n"
                % (self.box.mean(), len(self.bubble_centres))
            )
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
