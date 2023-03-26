import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparselinalg


def circle(x, y, x0, y0, r):
    d2 = (x-x0)**2 + (y-y0)**2
    return (d2 <= r**2)*1.


def ellipsis(x, y, x0, y0, r, factor):
    d2 = (x-x0)**2*factor + (y-y0)**2
    return (d2 <= r**2)*1.


def get_coord(L, n, centers=True):
    if centers:
        return (np.arange(n)+0.5)*(L/n)
    else:
        return (np.arange(n+1))*(L/n)


def get_elliptic_msk(Ly, Lx, shape):
    """ Define an elliptic mask on cell centers"""
    ny, nx = shape
    msk = np.zeros((ny, nx))
    radius = Ly/2
    x = get_coord(Lx, nx, centers=True)
    y = get_coord(Ly, ny, centers=True)
    xx, yy = np.meshgrid(x, y)
    msk = ellipsis(xx, yy, Lx/2, Ly/2, radius, 0.8*(ny/nx)**2)
    msk[msk > 0] = 1
    msk[msk < 0] = 0
    return msk


def get_Gmatrix(msk, xperio=False, yperio=False, centers=True):
    """ compute the G matrix from msk

    msk is at cell centers,
    G is either on centers or on vertices

    """
    ny, nx = msk.shape
    N = msk.sum()

    if centers:
        G = -np.ones((ny, nx), dtype="i")
        G[msk > 0] = np.arange(N)

    else:
        G = -np.ones((ny+1, nx+1), dtype="i")
        I = 0
        j0 = 0 if yperio else 1
        i0 = 0 if xperio else 1
        for j in range(j0, ny):
            for i in range(i0, nx):
                im = (i-1) % nx
                jm = (j-1) % ny
                inside = (msk[jm, im]+msk[jm, i]+msk[j, im]+msk[j, i]) == 4
                if inside:
                    G[j, i] = I
                    I += 1

    return G


def get_laplacian(Ly, Lx, msk,
                  centers=True, BCtype="dirichlet",
                  xperio=False, yperio=False):

    assert BCtype in ["dirichlet", "neumann"]

    ny, nx = msk.shape
    dy, dx = Ly/ny, Lx/nx
    cyy, cxx = 1/dy**2, 1/dx**2

    G = get_Gmatrix(msk, xperio=xperio, yperio=yperio, centers=centers)
    N = 1+G.max()

    maxN = N*5
    # allocating vectors
    data = np.zeros((maxN,))
    row = np.zeros((maxN,), dtype="i")
    col = np.zeros((maxN,), dtype="i")

    # since 'count' is a 'List' and not an 'int'
    # it is *mutable* and *visible* from inside 'add'
    # 'add' can increment 'count' even though 'count' is not
    # an argument of 'add'
    count = [0]

    def add(I, J, value):
        row[count] = I
        col[count] = J
        data[count] = value
        count[0] += 1

    for j, i in np.ndindex(G.shape):
        if G[j, i] > -1:
            diag = 0.
            I = G[j, i]
            im = (i-1) % nx
            ip = (i+1) % nx
            jm = (j-1) % ny
            jp = (j+1) % ny
            Jwest = G[j, im] if (xperio or i > 0) else -1
            Jeast = G[j, ip] if (xperio or i < nx-1) else -1
            Jnorth = G[jp, i] if (yperio or j < ny-1) else -1
            Jsouth = G[jm, i] if (yperio or j > 0) else -1

            if Jwest > -1:
                add(I, Jwest, cxx)
                diag -= cxx
            if Jeast > -1:
                add(I, Jeast, cxx)
                diag -= cxx
            if Jnorth > -1:
                add(I, Jnorth, cyy)
                diag -= cyy
            if Jsouth > -1:
                add(I, Jsouth, cyy)
                diag -= cyy

            if BCtype == "dirichlet":
                diag = -2*(cxx+cyy)

            add(I, I, diag)

    nnz = count[0]
    A = sparse.coo_matrix((data[:nnz], (row[:nnz], col[:nnz])), (N, N))
    return A.tocsr(), G


def solve(A, G, rhs):
    assert G.shape == rhs.shape

    phi = np.zeros_like(rhs)

    phi[G > -1] = sparselinalg.spsolve(A, rhs[G > -1])
    phi[G == -1] = 0.
    return phi
