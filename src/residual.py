from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

RESIDUAL_PATH = "data/output/"

class Residual:
  def __init__(self, model: Model, geom: Geometry, c) -> None:
    self.mdl = model
    self.geom = geom
    self.c = c

    self.homo_model = np.full(
      (self.mdl.nzz, self.mdl.nxx),
      self.mdl.model[0, 0]
    )

    self.seismogram = np.zeros((self.c.nt, self.geom.nrec))
    self.homo_seismogram = np.zeros((self.c.nt, self.geom.nrec))
    self.residual = np.zeros((self.c.nt, self.geom.nrec))

    self.ricker = np.zeros(self.c.nt)

    self.Psrc = np.zeros((self.mdl.nzz, self.mdl.nxx, self.c.nt))

    self.damp2D = np.ones((self.mdl.nzz, self.mdl.nxx))

    self.d2u_dx2 = np.zeros((self.mdl.nzz, self.mdl.nxx))
    self.d2u_dz2 = np.zeros((self.mdl.nzz, self.mdl.nxx))

    self.ix = 0
    self.iz = 0

  def fd_residual(self) -> None:
    
    for isrc in range(len(self.geom.srcxId)):

      self.seismogram.fill(0.0)
      self.homo_seismogram.fill(0.0)
      self.Psrc.fill(0.0)

      self.ix = int(self.geom.srcxId[isrc]) + self.c.nb
      self.iz = int(self.geom.srczId[isrc]) + self.c.nb

      self.fd_inner(self.mdl.model, self.seismogram)

      self.Psrc.fill(0.0)

      self.fd_inner(self.homo_model, self.homo_seismogram)

    if self.c.save_residual:
      self.save_residual()

  def fd_inner(self, model, seis_out) -> None:
    dh2 = self.c.dh * self.c.dh
    arg = self.c.dt * self.c.dt * model * model

    for t in range(1, self.c.nt - 1):

      self.Psrc[self.iz, self.ix, t] += self.ricker[t] / dh2

      lap = laplacian2d(
        self.Psrc[:, :, t],
        self.d2u_dx2,
        self.d2u_dz2,
        self.mdl.nzz,
        self.mdl.nxx,
        dh2
      )

      self.Psrc[:, :, t + 1] = (
        arg * lap
        + 2.0 * self.Psrc[:, :, t]
        - self.Psrc[:, :, t - 1]
      )

      self.Psrc[:, :, t + 1] *= self.damp2D
      self.Psrc[:, :, t] *= self.damp2D

      for irec in range(self.geom.nrec):
        rx = int(self.geom.recx[irec]) + self.c.nb
        rz = int(self.geom.recz[irec]) + self.c.nb
        seis_out[t, irec] = self.Psrc[rz, rx, t]

  def save_residual(self) -> None:
    self.residual = self.seismogram - self.homo_seismogram

    self.residual.astype("float32", order="F").flatten("F").tofile(
      RESIDUAL_PATH +
      f"seismogram_nt{self.c.nt}_dt{self.c.dt}_nrec{self.geom.nrec}.bin"
    )

  def get_ricker(self) -> None:
    fc = self.c.fmax / (3.0 * np.sqrt(np.pi))
    t = np.arange(self.c.nt) * self.c.dt - self.c.tlag

    arg = np.pi * (t * fc * np.pi) ** 2.0 

    self.ricker = (1.0 - 2.0 * arg) * np.exp(-arg)

  def set_damper(self) -> None:
    damp1D = np.zeros(self.c.nb)

    for i in range(self.c.nb):
      damp1D[i] = np.exp(-(self.c.factor*(self.c.nb - i))**2.0)

    for i in range(self.mdl.nzz):
      self.damp2D[i,:self.c.nb] *= damp1D
      self.damp2D[i,-self.c.nb:] *= damp1D[::-1]

    for j in range(self.mdl.nxx):
      self.damp2D[:self.c.nb,j] *= damp1D
      self.damp2D[-self.c.nb:,j] *= damp1D[::-1]

  def plot(self, seismogram, title=None):
    tloc = np.linspace(0, self.c.nt - 1, 11, dtype=int)
    tlab = np.around(tloc * self.c.dt, decimals=1)

    xloc = np.linspace(0, self.geom.nrec - 1, 9)
    xlab = np.array(self.c.dh * xloc, dtype=int)

    scale_min = np.percentile(seismogram, 100 - self.c.perc)
    scale_max = np.percentile(seismogram, self.c.perc)

    fig, ax = plt.subplots(figsize=(10, 8))

    img = ax.imshow(seismogram, aspect="auto", cmap="Greys",
                    vmin=scale_min, vmax=scale_max)

    ax.set_yticks(tloc)
    ax.set_yticklabels(tlab)

    ax.set_xticks(xloc)
    ax.set_xticklabels(xlab)

    ax.set_xlabel("Offset (m)", fontsize=13)
    ax.set_ylabel("TWT (s)", fontsize=13)

    if title:
      ax.set_title(title, fontsize=13)

    plt.show()

class Model:
  def __init__(self, c) -> None:
    self.c = c

    self.nxx = 2*self.c.nb + self.c.nx
    self.nzz = 2*self.c.nb + self.c.nz

    self.model = np.zeros((self.c.nz, self.c.nx))

  def load(self) -> None:
    self.model = np.fromfile(
      self.c.model_path, dtype=np.float32, count=self.c.nx*self.c.nz
    ).reshape([self.c.nz, self.c.nx], order='F')

  def set_boundary(self) -> None:
    model_ext = np.zeros((self.nzz, self.nxx))

    for j in range(self.c.nx):
      for i in range(self.c.nz):
        model_ext[i+self.c.nb, j+self.c.nb] = self.model[i, j]

    for j in range(self.c.nb, self.c.nx+self.c.nb):
      for i in range(self.c.nb):
        model_ext[i, j] = model_ext[self.c.nb, j]
        model_ext[self.c.nz+self.c.nb+i, j] = model_ext[self.c.nz+self.c.nb-1, j]

    for i in range(self.nzz):
      for j in range(self.c.nb):
        model_ext[i, j] = model_ext[i, self.c.nb]
        model_ext[i, self.c.nx+self.c.nb+j] = model_ext[i, self.c.nx+self.c.nb-1]

    self.model = model_ext

class Geometry:
  def __init__(self, c) -> None:
    self.c = c

    self.recx, self.recz     = np.array([]), np.array([])
    self.srcxId, self.srczId = np.array([]), np.array([])

    self.nrec = 0

    self.dt_canditates = np.array([])

    self.max_dt = 0.0

  def get(self) -> None:
    receivers = np.loadtxt(self.c.receivers, delimiter=',', skiprows=1)

    if receivers.ndim == 1:
      self.recx = np.array([receivers[1]])
      self.recz = np.array([receivers[2]])
    else:
      self.recx = receivers[:, 1]
      self.recz = receivers[:, 2]

    sources = np.loadtxt(self.c.sources, delimiter=',', skiprows=1)

    if sources.ndim == 1:
      self.srcxId = np.array([sources[1]])
      self.srczId = np.array([sources[2]])
    else:
      self.srcxId = sources[:, 1]
      self.srczId = sources[:, 2]

    self.nrec = len(self.recx)

@njit(parallel=True)
def laplacian2d(
    upre, d2u_dx2, d2u_dz2, 
    nzz, nxx, dh2,
) -> None:
  inv_dh2 = 1.0 / (5040.0 * dh2)

  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      d2u_dx2[i, j] = (
          -9   * upre[i-4, j] + 128   * upre[i-3, j] - 1008 * upre[i-2, j] +
          8064 * upre[i-1, j] - 14350 * upre[i,   j] + 8064 * upre[i+1, j] -
          1008 * upre[i+2, j] + 128   * upre[i+3, j] - 9    * upre[i+4, j]
      ) * inv_dh2

      d2u_dz2[i, j] = (
          -9   * upre[i, j-4] + 128   * upre[i, j-3] - 1008 * upre[i, j-2] +
          8064 * upre[i, j-1] - 14350 * upre[i, j]   + 8064 * upre[i, j+1] -
          1008 * upre[i, j+2] + 128   * upre[i, j+3] - 9    * upre[i, j+4]
      ) * inv_dh2

  return d2u_dx2 + d2u_dz2
