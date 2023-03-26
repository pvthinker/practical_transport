import numpy as np
import matplotlib.pyplot as plt
import laplacian_tools as lt
from PIL import Image
from weno import weno3, weno5
from numba import njit
from enum import IntEnum

plt.ion()


nx = 100
Lx = 1.
Ly = 1.
tend = 1.

# cfl is the Courant number -> dt is automatically set from it
cfl = .5
flxorder = 2
weno_activated = False
# Asselin's constant only matters with LeapFrog
asselin_cst = 0.1

# all the cases that can be combined
Domain = IntEnum("Domain", ["square", "disk", "disk_with_obstacle"])
Flow = IntEnum("Flow", ["bodyrotation", "vortex", "quadripole"])
Tracer = IntEnum("Tracer", ["square", "gaussian", "tiles", "cow"])
Integrator = IntEnum("Integrator", ["RK3", "LFAM3", "LeapFrog"])

# select a particular combination
domain = Domain.disk
flow = Flow.bodyrotation
tracer = Tracer.gaussian
integrator = Integrator.LeapFrog

# whether or not to integrate backward to initial time
backward = False

# number of iterations between two plots (for the animation)
plotfreq = 10
cmap = "RdBu"

dx = Lx/nx
dy = dx
ny = int(Ly/dx)

assert dx == dy
assert ny % 2 == 0, "use an even number for ny -> easier to enforce symmetric flows"

shape_centers = (ny, nx)
shape_u = (ny, nx+1)
shape_v = (ny+1, nx)
shape_vertices = (ny+1, nx+1)

xc = dx*(np.arange(nx)+0.5)-Lx/2
yc = dy*(np.arange(ny)+0.5)-Ly/2

xe = dx*(np.arange(nx+1))-Lx/2
ye = dy*(np.arange(ny+1))-Ly/2


def get_xy(which):
    """ return 2D arrays of x and y"""

    assert which in ["centers", "u", "v", "vertices"]

    if which == "centers":
        return np.meshgrid(xc, yc)

    elif which == "u":
        return np.meshgrid(xe, yc)

    elif which == "v":
        return np.meshgrid(xc, ye)

    elif which == "vertices":
        return np.meshgrid(xe, ye)


xvert, yvert = get_xy("vertices")
xcent, ycent = get_xy("centers")

assert xvert.shape == shape_vertices
assert yvert.shape == shape_vertices
assert xcent.shape == shape_centers
assert ycent.shape == shape_centers


def get_msk(x, y, domain):
    msk = np.ones(x.shape, dtype="i4")

    if domain == Domain.square:
        pass

    elif domain in [Domain.disk, Domain.disk_with_obstacle]:

        dist = (x**2+y**2)**0.5
        msk[dist > 0.5] = 0

        if domain == Domain.disk_with_obstacle:
            f = 0.2
            msk[(y < -f) & (x < -f)] = 0

            # msk[(y<-f) & (x>f)] = 0
            # msk[(y>f) & (x>f)] = 0
            # msk[(y>f) & (x<-f)] = 0

    else:
        raise NotImplementedError(domain)

    return msk


msk = get_msk(xcent, ycent, domain)
mskv = get_msk(xvert, yvert, domain)


def perpgrad(psi, dx):
    """ compute u = \nabla^\perp psi"""
    assert psi.shape == shape_vertices
    u = np.zeros(shape_u)
    v = np.zeros(shape_v)
    u[:, :] = -(psi[1:, :]-psi[:-1, :])/dx
    v[:, :] = +(psi[:, 1:]-psi[:, :-1])/dx
    return u, v


def get_flow(Lx, Ly, nx, flow):
    dx = Lx/nx
    ny = int(Ly/dx)

    A, G = lt.get_laplacian(Lx, Ly, msk,
                            centers=False,
                            BCtype="dirichlet")

    rhs = np.zeros((ny+1, nx+1))

    if flow == Flow.bodyrotation:
        rhs[:] = 4*np.pi

    elif flow == Flow.vortex:
        x, y = np.meshgrid(xe, ye)
        rhs = 100*lt.circle(x, y, 0., 0., 0.1)

    elif flow == Flow.quadripole:
        x, y = np.meshgrid(xe, ye)
        rhs = lt.circle(x, y, 0.2, 0.2, 0.1)
        rhs += lt.circle(x, y, -0.2, -0.2, 0.1)
        rhs -= lt.circle(x, y, -0.2, 0.2, 0.1)
        rhs -= lt.circle(x, y, 0.2, -0.2, 0.1)
        rhs *= 100

    else:
        raise NotImplementedError(flow)

    # here we solve nabla^2 psi = rhs
    # psi defined at cell vertices
    # with homogeous Dirichlet boundary condition (psi=0)
    psi = lt.solve(A, G, rhs)

    # the (u, v) flow is guaranteed to be incompressible
    u, v = perpgrad(psi, dx)
    u[:, 1:-1] *= msk[:, 1:]*msk[:, :-1]
    v[1:-1, :] *= msk[1:, :]*msk[:-1, :]
    return psi, u, v


def compute_time_step(u, v, dx, cfl):
    um = 0.5*(u[:, 1:]+u[:, :-1])
    vm = 0.5*(v[1:, :]+v[:-1, :])
    umax = np.max(np.abs(um+vm))
    dt = cfl*dx/umax
    return dt


def get_tracer_patch(Lx, Ly, nx, tracer):

    x, y = np.meshgrid(xc, yc)

    if tracer == Tracer.square:
        q = 1.*((0.225*Lx <= x)
                & (x <= 0.275*Lx)
                & (-0.025*Ly <= y)
                & (y <= 0.025*Ly))

    elif tracer == Tracer.gaussian:
        x0, y0 = 0.25, 0.25
        sigma = 0.01
        d2 = (x-x0)**2 + (y-y0)**2
        q = np.exp(-d2/(2*sigma**2))
        amp = (1e-3/np.mean(q**2))**.5
        q *= amp

    elif tracer == Tracer.tiles:
        ntiles = 8
        q = (np.round((x-0.5)*ntiles) %
             2 + np.round((y-0.5)*ntiles) % 2)/2
        q = 0.3*(q-0.5)

    elif tracer == Tracer.cow:
        im = Image.open("cow.png")
        q = np.zeros(x.shape)
        data = im.resize(q.shape)

        q[:] += data.getchannel("R")
        q[:] += data.getchannel("G")
        q[:] += data.getchannel("B")
        q /= 3

        q = ((q/255)-0.5)/3
        q = np.flipud(q)

    else:
        raise NotImplementedError(tracer)

    return q*msk


@njit
def flux1d(q, u, localorder, flx):
    assert q.ndim == 1
    assert u.ndim == 1
    assert localorder.ndim == 1
    assert q.size+1 == localorder.size
    assert u.size == localorder.size
    nx = q.size
    for i in range(1, nx):

        if localorder[i] == 1:
            if u[i] > 0:
                flx[i] = u[i]*q[i-1]
            else:
                flx[i] = u[i]*q[i]

        elif localorder[i] == 2:
            flx[i] = 0.5*u[i]*(q[i-1]+q[i])

        elif localorder[i] == 3:
            if weno_activated:
                if u[i] > 0:
                    flx[i] = u[i]*weno3(q[i-2], q[i-1], q[i])
                else:
                    flx[i] = u[i]*weno3(q[i+1], q[i], q[i-1])
            else:
                if u[i] > 0:
                    flx[i] = u[i]*(-q[i-2]+5*q[i-1]+2*q[i])/6
                else:
                    flx[i] = u[i]*(-q[i+1]+5*q[i]+2*q[i-1])/6

        elif localorder[i] == 4:
            flx[i] = u[i]*(-q[i-2]+7*q[i-1]+7*q[i]-q[i+1])/12

        elif localorder[i] == 5:

            if weno_activated:
                if u[i] > 0:
                    flx[i] = u[i]*weno5(q[i-3], q[i-2], q[i-1], q[i], q[i+1])
                else:
                    flx[i] = u[i]*weno5(q[i+2], q[i+1], q[i], q[i-1], q[i-2])

            else:
                if u[i] > 0:
                    flx[i] = u[i]*(2*q[i-3]-13*q[i-2]+47 *
                                   q[i-1]+27*q[i]-3*q[i+1])/60
                else:
                    flx[i] = u[i]*(2*q[i+2]-13*q[i+1]+47 *
                                   q[i]+27*q[i-1]-3*q[i-2])/60

        else:
            flx[i] = 0.


@njit
def rhs(q, u, v, dx, dq):
    ny, nx = q.shape
    flx = np.zeros((nx+1))
    fly = np.zeros((ny+1))
    dq[:] = 0.
    for j in range(ny):
        flux1d(q[j, :], u[j, :], xorder[j, :], flx)
        dq[j, :] = -np.diff(flx)

    for i in range(nx):
        flux1d(q[:, i], v[:, i], yorder[:, i], fly)
        dq[:, i] -= np.diff(fly)

    dq *= (1/dx)*msk


def update_tracer(q, u, v, dx, dt, integrator, first=False):

    if integrator == Integrator.RK3:

        rhs(q, u, v, dx, dq0)
        q += dt*dq0

        rhs(q, u, v, dx, dq1)
        q += (dt/4)*(dq1-3*dq0)

        rhs(q, u, v, dx, dq2)
        q += (dt/12)*(8*dq2-dq1-dq0)

    elif integrator == Integrator.LFAM3:

        if first:
            rhs(q, u, v, dx, dq0)
            qb[:] = q
            q += dt*dq0

        else:
            rhs(q, u, v, dx, dq0)

            qstar = qb+2*dt*dq0
            qstar = (5/12)*qstar+(8/12)*q-(1/12)*qb
            qb[:] = q

            rhs(qstar, u, v, dx, dq0)

            q += dt*dq0

    elif integrator == Integrator.LeapFrog:

        if first:
            rhs(q, u, v, dx, dq0)
            qb[:] = q
            q += dt*dq0

        else:
            rhs(q, u, v, dx, dq0)

            qstar = qb+2*dt*dq0

            # Asselin filter
            q[:] += asselin_cst * (qstar-2*q+qb)

            qb[:] = q
            q[:] = qstar


psi, u, v = get_flow(Lx, Ly, nx, flow)

q = get_tracer_patch(Lx, Ly, nx, tracer)
qb = np.zeros(q.shape)
dq0 = np.zeros(q.shape)
dq1 = np.zeros(q.shape)
dq2 = np.zeros(q.shape)
dt = compute_time_step(u, v, dx, cfl)


def set_localorder1d(m1d, o):

    assert (flxorder >= 1) & (flxorder <= 5)

    nx = len(m1d)
    for i in range(1, nx):
        m2 = m1d[i-1]*m1d[i]

        if (i-2 >= 0) and (i+1 <= nx-1):
            m4 = m1d[i-2]*m1d[i+1]*m2

        else:
            m4 = 0

        if (i-3 >= 0) and (i+2 <= nx-1):
            m6 = m1d[i-3]*m1d[i+2]*m4

        else:
            m6 = 0

        if flxorder == 1:
            o[i] = max(0, m2)

        elif flxorder == 2:
            o[i] = max(0, 2*m2)

        elif flxorder == 3:
            o[i] = max(0, m2, 3*m4)

        elif flxorder == 4:
            o[i] = max(0, 2*m2, 4*m4)

        elif flxorder == 5:
            o[i] = max(0, m2, 3*m4, 5*m6)

        else:
            raise ValueError


def set_localorder(msk, xord, yord):
    """

    input:
      - msk @ cell centers

    output :
      - xord : order to use for along x-flux at cell x-edge
      - yord : order to use for along y-flux at cell y-edge

    """
    ny, nx = msk.shape
    for j in range(ny):
        set_localorder1d(msk[j, :], xord[j, :])

    for i in range(nx):
        set_localorder1d(msk[:, i], yord[:, i])


xorder = np.zeros((ny, nx+1), dtype="i4")
yorder = np.zeros((ny+1, nx), dtype="i4")
set_localorder(msk, xorder, yorder)


q0 = q.copy()
ite = 0
time = 0

title_template = "time={time:.2f}\n{integrator.name} / {order}-order / weno:{weno_activated}"

infos = {"time": time,
         "integrator": integrator,
         "order": flxorder,
         "weno_activated": weno_activated}

fig, ax = plt.subplots()
im = ax.imshow(q,
               vmin=-.2, vmax=0.2,
               extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
               origin="lower",
               cmap=cmap)

title = ax.set_title(title_template.format(**infos))
ax.contour(xe, ye, psi, colors="k")
plt.colorbar(im)

Qtotal = [np.mean(q)]
Q2total = [np.mean(q**2)]

dt = tend/int(np.floor(tend/dt))

while time < tend:
    update_tracer(q, u, v, dx, dt, integrator, ite == 0)

    Qtotal += [np.mean(q)]
    Q2total += [np.mean(q**2)]

    ite += 1
    time = ite*dt
    if (ite % plotfreq == 0) or (time >= tend):
        im.set_data(q)
        infos["time"] = time
        title.set_text(title_template.format(**infos))
        fig.canvas.draw()
        plt.pause(1e-4)


if backward:
    iteforward = ite
    ite = 0
    u = -u
    v = -v
    while time > 0:
        update_tracer(q, u, v, dx, dt, integrator, ite == 0)

        Qtotal += [np.mean(q)]
        Q2total += [np.mean(q**2)]

        ite += 1
        time -= dt
        if (ite % plotfreq == 0) or (time <= 0):
            im.set_data(q)
            infos["time"] = time
            title.set_text(title_template.format(**infos))
            fig.canvas.draw()
            plt.pause(1e-4)


# Change of q
fig, ax = plt.subplots()
im = ax.imshow(q-q0,
               extent=[-Lx/2, Lx/2, -Ly/2, Ly/2],
               origin="lower",
               cmap="RdBu_r")
plt.title("Final-Initial")
plt.colorbar(im)


# global conservation of q and q**2
fig, ax = plt.subplots(1, 2)
ax[0].plot(Qtotal)
ax[0].set_title("Total amount of Q")
ax[0].set_xlabel("iteration")

ax[1].plot(Q2total)
ax[1].set_title(r"Total amount of $Q^2$")
ax[1].set_xlabel("iteration")
plt.tight_layout()


# histograms of q
plt.figure()
bins = np.linspace(-.4, .4, 101)
plt.hist(q0[msk > 0].ravel(), bins=bins, label="initial time")
plt.hist(q[msk > 0].ravel(), bins=bins, alpha=0.5, label="final time")
ax = plt.gca()
ax.set_yscale("log")
plt.xlabel("q bins")
plt.ylabel("N(q)")
plt.title("histogram of q")
plt.legend()

#plt.show(block=True)
