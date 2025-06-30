import torch
from typing import Any


class FickResidual:
    def __init__(self, P, D, T, P0) -> None:
        self.D = D
        self.T = T
        self.P0 = P0
        self.P = P

    def __call__(self, X, *args: Any, **kwds: Any) -> Any:
        X.requires_grad_(True)
        r = X[:, 0].view(-1, 1)
        p_tensor = self.P(X)
        dP_d = torch.autograd.grad(
            p_tensor, X, grad_outputs=torch.ones_like(p_tensor), create_graph=True
        )[0]
        dP_dr = dP_d[:, 0].view(-1, 1)
        dP_dt = dP_d[:, 1].view(-1, 1)
        # dP_dd = torch.autograd.grad(
        #     dP_d,
        #     X,
        #     grad_outputs=torch.ones_like(dP_d),
        #     create_graph=True,
        # )[0]
        # dP_drr = dP_dd[:, 0].view(-1, 1)
        D_tensor = self.D(r)
        P0_tensor = self.P0(r)
        T_tensor = self.T(r)
        nabla_D_nablaP = torch.autograd.grad(
            D_tensor * dP_dr,
            r,
            grad_outputs=torch.ones_like(D_tensor),
            create_graph=True,
        )[0]
        return dP_dt - nabla_D_nablaP + ((p_tensor - P0_tensor) / T_tensor)


params = {
    "D_f": 5 * 10 ** (-16),
    "D_j": 2.5807 * 10 ** (-16),
    "T_1": 20,
    "T_B": 3,
    "P0_f": 1,
    "P0_j": 200,
    "R_bis": 225 * 10 ** (-9),
    "def_t": 20,
    "rayon_initialisation": 100 * 10 ** (-9),
    "name": "Sim3",
}

if __name__ == "__main__":
    from model import G_MLP

    G = G_MLP(nb_layer=2, hidden_layer=10)
    # P = P_from_G(G=G)
    D = params["D_f"]
    T = params["T_1"]
    P0 = params["P0_f"]
    fick_res = FickResidual(D=D, T=T, P0=P0)
    # print(fick_res(P, X))
