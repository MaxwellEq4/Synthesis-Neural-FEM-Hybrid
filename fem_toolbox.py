import numpy as np
from scipy.spatial.distance import cdist
import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
import plotly.figure_factory as ff
from scipy.spatial import Delaunay
import plotly.graph_objects as go
from scipy.interpolate import griddata

class FEM:
    def __init__(self, x0, y0, L1, L2, noelms1, noelms2):
        self.VX, self.VY = self.xy(x0, y0, L1, L2, noelms1, noelms2)
        self.EToV = self.conelmtab(noelms1, noelms2)
        self.bnodes = self.boundarynodes(self.EToV)
        self.beds = self.ConstructBeds(self.VX, self.VY, self.EToV, [x0], [y0])


    def xy(self, x0, y0, L1, L2, noelms1, noelms2):
        """
        Purpose: Compute x & y coordinates of mesh nodes
        Author(s): Mathias R. Jeppesen, s174039
        """
        nonodes = (noelms1 + 1) * (noelms2 + 1)

        VX = np.zeros(nonodes)
        VY = np.zeros(nonodes)

        x_cor = np.linspace(x0, L1 + x0, noelms1 + 1)
        y_cor = np.linspace(y0, L2 + y0, noelms2 + 1)

        index = 0

        for i in range(len(x_cor)):
            for j in range(len(y_cor)):
                VY[index] = y_cor[-j - 1]
                VX[index] = x_cor[i]
                index += 1

        return VX, VY

    def conelmtab(self, noelms1, noelms2):
        """
        Purpose: Construct Element-to-Vertice array to define node connections
        Author(s): Mathias R. Jeppesen, s174039.
        """
        noelms = noelms1 * noelms2 * 2
        EToV = np.zeros((noelms, 3), dtype=int)

        for i in range(noelms // 2):
            EToV[i * 2, :] = np.array([1, noelms2 + 2, 0]) + i + (i // noelms2)


        for i in range(noelms // 2):
            EToV[i * 2 - 1, :] = np.array([0, noelms2 + 2, noelms2 + 1]) + i + (i // noelms2)


        return EToV


    def basfun(self, n, VX, VY, EToV):
        """
        Purpose: Compute a_i, b_i, c_i as well as delta
        Author(s): Mathias R. Jeppesen, s174039.
        """
        

        index = EToV[n, :]
        x = VX[index]
        y = VY[index]

        # eq. (2.1)
        delta = 0.5 * (x[1] * y[2] - y[1] * x[2] - (x[0] * y[2] - y[0] * x[2]) + x[0] * y[1] - y[0] * x[1])
        abc = np.zeros((3, 3))

        # eq. (2.3) and below.
        i, j, k = 0, 1, 2
        abc[i, :] = [x[j] * y[k] - x[k] * y[j], y[j] - y[k], x[k] - x[j]]

        i, j, k = 1, 2, 0
        abc[i, :] = [x[j] * y[k] - x[k] * y[j], y[j] - y[k], x[k] - x[j]]

        i, j, k = 2, 0, 1
        abc[i, :] = [x[j] * y[k] - x[k] * y[j], y[j] - y[k], x[k] - x[j]]
        self.abc, self.delta = abc, delta
        return delta, abc
    

    def outernormal(self, n, k):
        """
        Purpose: Compute outer normal vectors for a given n element and k edge.
        Author(s): Mathias R. Jeppesen, s174039.
        """
        if k == 0:
            edge_n = [2, 0]
        elif k == 1:
            edge_n = [0, 1]
        elif k == 2:
            edge_n = [1, 2]
        else:
            raise ValueError('Unexpected value for k. Expected 1, 2, or 3.')
    
        index = self.EToV[n, edge_n]
        x = self.VX[index]
        y = self.VY[index]
    
        # eq. (2.4)
        delta_x = x[1] - x[0]
        delta_y = y[1] - y[0]
    
        # eq. (2.6)
        n1 = -delta_y / np.sqrt(delta_x ** 2 + delta_y ** 2)
        n2 = delta_x / np.sqrt(delta_x ** 2 + delta_y ** 2)
        self.n1, self.n2 = n1, n2
    
        return n1, n2



    def assembly(self, lam1 = 1, lam2 = 1, qt = None):
        nonodes = len(self.VX)
        noelms = len(self.EToV)
        A = np.zeros((nonodes, nonodes))
        b = np.zeros(nonodes)
    
        for n in range(noelms):
            index = self.EToV[n, :]
            delta, abc = self.basfun(n, self.VX, self.VY, self.EToV)  
    
            q = qt[index]
            qtilde = np.mean(q)
            qtilde_rn = abs(delta) / 3 * qtilde
    
            for r in range(3):
                i = index[r]
                b[i] = b[i] + qtilde_rn
                for s in range(3):
                    b_r = abc[r, 1]
                    b_s = abc[s, 1]
                    c_r = abc[r, 2]
                    c_s = abc[s, 2]
    
                    k_rs = 1 / (4 * abs(delta)) * (lam1 * b_r * b_s + lam2 * c_r * c_s)
    
                    j = index[s]
    
                    A[i, j] = A[i, j] + k_rs
        self.A, self.b = A, b
        return A, b

    
    def boundarynodes(self,EToV):
        nodes = np.unique(EToV)
        elements_encountered = np.bincount(EToV.ravel(), minlength=nodes[-1] + 1)[nodes]
        edges = nodes[elements_encountered < 6]

        bnodes = []
        for bnode_i in edges:
            bnodes.append(bnode_i)
        
        self.bnodes = bnodes
        return bnodes

    def dirbc(self, f, A, b):
        omega_n = []
        
        for index in range(len(b)):
            if not np.any(self.bnodes == index):
                omega_n.append(index)
    
        omega_n = np.array(omega_n)  # convert list to numpy array for efficient computations later
    
        for bnode_index in range(len(self.bnodes)):
            i = self.bnodes[bnode_index]
            A[i, i] = 1
            b[i] = f[bnode_index]
    
            for j in range(len(b)):
                if i == j:
                    continue
    
                A[i, j] = 0
                if np.any(omega_n == j):
                    b[j] = b[j] - A[j, i] * f[bnode_index]
                    A[j, i] = 0
    
        self.A, self.b = A, b
        return A, b



    def ConstructBeds(self, VX, VY, EToV, xallowed=None, yallowed=None):
        if xallowed is None and yallowed is None:
            flag_allow = False
        else:
            flag_allow = True
    
        bnodes = self.boundarynodes(EToV)
    
        beds = []
    
        x = VX[bnodes]
        y = VY[bnodes]
    
        for i_bnode in range(len(bnodes)):
            xi = x[i_bnode]
            yi = y[i_bnode]
    
            dist = cdist(np.array([[xi, yi]]), np.column_stack((x, y)))
            I = np.argpartition(dist, 3)[:, :3][0]
            I = I[I != i_bnode]
    
            for j_bnode in I:
                for n in range(EToV.shape[0]):
                    if bnodes[j_bnode] in EToV[n, :] and bnodes[i_bnode] in EToV[n, :]:
                        r = np.where(EToV[n, :] == bnodes[i_bnode])[0][0]
                        s = np.where(EToV[n, :] == bnodes[j_bnode])[0][0]
    
                        if (s - r) % 3 == 1 and flag_allow:
                            xtest = (x[i_bnode] + x[j_bnode]) / 2
                            ytest = (y[i_bnode] + y[j_bnode]) / 2
                            if (xtest in xallowed or not xallowed) or (ytest in yallowed or not yallowed):
                                beds.append([n, r])
    
                        elif (s - r) % 3 == 1:
                            beds.append([n, r])
        self.beds = np.array(beds)
        return np.array(beds)


    def neubc(self, beds, q, b):
        # Purpose: Algorithm 8. Imposing Neumann boundary conditions by
        # modification of system (2D).
        # Author(s): Mathias R. Jeppesen, s174039

        for p in range(beds.shape[0]):
            # Look up n and r for the p'th edge
            n = beds[p, 0]
            r = beds[p, 1]
            # Determine s and look up i and j and the (x,y) coordinates of the end
            # nodes
            s = (r + 1) % 3

            i = self.EToV[n, r]
            j = self.EToV[n, s]

            xi = self.VX[i]
            yi = self.VY[i]

            xj = self.VX[j]
            yj = self.VY[j]

            # compute q1p and q2p from (2.41);
            qp12 = (q[p] / 2) * np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)  # eq. (2-41)
            b[i] -= qp12
            b[j] -= qp12
        self.b = b
        return b
    def fem_solve(self, qt_function, f_function, bc_type='dirbc', lam1=1, lam2=1):
        """ Compute the solution according to given boundary condition type """
        qt = qt_function(self.VX, self.VY)
        f = f_function(self.VX[self.bnodes], self.VY[self.bnodes])
        
        self.A, self.b = self.assembly(lam1, lam2, qt)
    
        if bc_type == 'dirbc':
            self.A, self.b = self.dirbc(f, self.A, self.b)
        elif bc_type == 'neubc':
            q = np.zeros(len(self.beds))
            self.b = self.neubc(self.beds, q, self.b)
            self.A, self.b = self.dirbc(f, self.A, self.b)
        else:
            raise ValueError("Invalid boundary condition type. Choose either 'dirbc' or 'neubc'.")
        
        u = np.linalg.solve(self.A, self.b)
        
        # plot
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(self.VX, self.VY, u.flatten(), triangles=self.EToV, cmap='viridis')
        ax.set_xlabel('x', fontsize=15)
        ax.set_ylabel('y', fontsize=15)
        ax.set_zlabel('z', fontsize=15)
        ax.set_title('3D visualization of solution with ' + bc_type.capitalize() + ' BC', fontsize=17)
        plt.show()
        # create a grid of points where you want to interpolate
        grid_x, grid_y = np.mgrid[min(self.VX):max(self.VX):100j, min(self.VY):max(self.VY):100j]  
        
        # interpolate the u values onto this grid
        grid_u = griddata((self.VX, self.VY), u.flatten(), (grid_x, grid_y), method='cubic')
    
        # new standalone heatmap figure
        fig, ax = plt.subplots(figsize=(8, 6))
        img = ax.imshow(grid_u.T, extent=(min(self.VX), max(self.VX), min(self.VY), max(self.VY)), 
                         origin='lower', cmap='plasma')
        plt.colorbar(img, ax=ax)
        ax.set_xlabel('x', fontsize=15)
        ax.set_ylabel('y', fontsize=15)
        ax.set_title('Heatmap of solution with ' + bc_type.capitalize() + ' BC', fontsize=17)
        plt.show()
    
        return u

    def set_u_zero(self):
        """
        Initialize the solution vector u with zeros
        """
        if hasattr(self, 'b'):
            self.u = np.zeros_like(self.b)
            return self.u
        else:
            raise ValueError('self.b is not defined. Call the assembly method before calling set_u_zero.')

    def get_h(self):
        """
        Get the current solution vector h
        """
        if hasattr(self, 'u'):
            return self.u
        else:
            raise ValueError('Solution vector u is not defined. Call the fem_solve method before calling get_h.')
    def fem_error(self, analytical_sol):
        """
        Compute the error of the FEM solution against an analytical solution.
        The analytical solution should be a function that takes two parameters (x, y coordinates).
        """
    
        if not hasattr(self, 'u'):
            raise ValueError('Solution vector u is not defined. Call the fem_solve method before calling fem_error.')
    
        # Compute analytical solution at FEM nodes
        analytical_u = analytical_sol(self.VX, self.VY)
    
        # Compute error at FEM nodes
        error = np.abs(analytical_u - self.u)
    
        # plot error
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(self.VX, self.VY, error.flatten(), triangles=self.EToV, cmap='viridis')
        ax.set_xlabel('x', fontsize=15)
        ax.set_ylabel('y', fontsize=15)
        ax.set_zlabel('Error', fontsize=15)
        ax.set_title('3D visualization of error', fontsize=17)
        plt.show()
    
        # return error
        return error
