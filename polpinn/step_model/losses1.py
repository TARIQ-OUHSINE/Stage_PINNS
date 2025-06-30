import torch
from torch.nn import MSELoss

from polpinn.step_model.physics1 import FickResidual

mse = MSELoss()

# Perte sur l'équation de la physique (EDP)
# Appliquée sur tout le domaine (solide + liquide) jusqu'à R' (system_radius)
class PDEFickLoss:
    def __init__(self, P, D, T, P0, system_radius, t_max, n_samples):
        self.fick_residual = FickResidual(P=P, D=D, T=T, P0=P0)
        self.n_samples = n_samples
        self.system_radius = system_radius
        self.t_max = t_max

    def __call__(self):
        # Points de collocation dans tout le volume de 0 à system_radius
        r_collocation = torch.sqrt(torch.rand(self.n_samples, 1)) * self.system_radius
        t_collocation = torch.rand(self.n_samples, 1) * self.t_max
        
        X = torch.cat([r_collocation, t_collocation], dim=1)
        X.requires_grad_(True)
        
        res = self.fick_residual(X=X)
        return mse(res, torch.zeros_like(res))

# LA NOUVELLE LOSS CLÉ : L_yz du poster
# Elle lie G(R', t) aux données du solide et du liquide via la loi de conservation
class ConservationDataLoss:
    def __init__(self, G, data_in, data_out, radius, system_radius):
        # On suppose que data_in et data_out ont les mêmes 'times' grâce à l'augmentation
        self.times = data_in['times'].view(-1, 1)
        self.values_in = data_in['values'].view(-1, 1)   # fy(t)
        self.values_out = data_out['values'].view(-1, 1) # fz(t)
        
        self.R = radius
        self.R_prime = system_radius
        self.G = G
        
        # Coordonnées pour l'évaluation : sur la surface externe r = R'
        self.X = torch.cat(
            [torch.full_like(self.times, fill_value=self.R_prime), self.times],
            dim=1,
        )

    def __call__(self):
        # Prédiction du modèle pour G(R', t)
        G_R_prime_predicted = self.G(self.X)
        
        # Valeur cible théorique basée sur la conservation de la polarisation
        vol_ratio_solid = (self.R**3) / (self.R_prime**3)
        vol_ratio_liquid = 1.0 - vol_ratio_solid

        G_R_prime_target = vol_ratio_liquid * self.values_out + vol_ratio_solid * self.values_in
        
        return mse(G_R_prime_predicted, G_R_prime_target)

# Perte sur la condition initiale P(r, 0) = 0
class InitialConditionLoss:
    def __init__(self, P, n_samples, system_radius):
        self.n_samples = n_samples
        self.system_radius = system_radius
        self.P = P

    def __call__(self):
        r_tensor = torch.linspace(0, self.system_radius, self.n_samples, requires_grad=True).view(-1, 1)
        X = torch.cat([r_tensor, torch.zeros_like(r_tensor)], dim=1)
        p_tensor = self.P(X)
        return mse(p_tensor, torch.zeros_like(p_tensor))

# Perte sur la condition de symétrie au centre (gradient de P nul à r=0)
class BoundarySymmetryLoss:
    def __init__(self, P, n_samples, t_max):
        self.n_samples = n_samples
        self.t_max = t_max
        self.P = P

    def __call__(self):
        t_tensor = torch.linspace(0, self.t_max, self.n_samples, requires_grad=True).view(-1, 1)
        X = torch.cat([torch.zeros_like(t_tensor), t_tensor], dim=1)
        p_tensor = self.P(X)
        dP = torch.autograd.grad(
            p_tensor, X, grad_outputs=torch.ones_like(p_tensor), create_graph=True
        )[0]
        dP_dr = dP[:, 0].view(-1, 1)
        return mse(dP_dr, torch.zeros_like(dP_dr))

# FONCTION DE COUT PRINCIPALE
def compute_loss(
    model_G,
    model_P,
    data_in,
    data_out,
    R,
    system_radius,
    t_max,
    T,
    P0,
    D,
    n_samples
    ):
    losses = {}

    # L_pde: Résidu de l'EDP sur tout le domaine
    losses["pde"] = PDEFickLoss(
        P=model_P, D=D, T=T, P0=P0, system_radius=system_radius, t_max=t_max, n_samples=n_samples
    )()
    
    # L_yz: Perte de conservation sur les données
    losses["conservation_data"] = ConservationDataLoss(
        G=model_G, data_in=data_in, data_out=data_out, radius=R, system_radius=system_radius
    )()
    
    # L_init: P(r,0) = 0
    losses["init"] = InitialConditionLoss(
        P=model_P, n_samples=n_samples, system_radius=system_radius
    )()

    # L_bnd: Symétrie à r=0
    losses["symmetry"] = BoundarySymmetryLoss(
        P=model_P, t_max=t_max, n_samples=n_samples,
    )()
    
    losses["total"] = sum(losses.values())
    return losses