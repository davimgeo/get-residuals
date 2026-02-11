from src import *

PATH = "config/parameters.toml"

def main():
    cfg = Config(PATH).load()

    model = Model(cfg)
    model.load()
    model.set_boundary()

    geom = Geometry(cfg)
    geom.get()

    res = Residual(model, geom, cfg)
    res.get_ricker()
    res.set_damper()
    res.fd_residual()

    return res

if __name__ == "__main__":
    res = main()

    res.plot(res.residual)