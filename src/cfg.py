import tomllib
from dataclasses import dataclass

@dataclass
class Parameters:
  debug = False
  
  receivers = ""
  sources = ""

  offset = 0.0

  model_path = ""

  nx = 0
  nz = 0

  nb = 0
  factor = 0.015

  dh = 0

  tlag = 0.0
  
  perc = 99

  save_residual = False

  nt   = 0
  dt   = 0.0

  fmax = 0

class Config(Parameters):
  def __init__(self, toml_path: str):
    super().__init__()

    self.toml_path = toml_path 

  def load(self):
    with open(self.toml_path, "rb") as f:
      data = tomllib.load(f)

      if self.debug: print(data)

    for key, value in data.items():
      setattr(self, key, value)

    return self


