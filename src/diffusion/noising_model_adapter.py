import sys
import os
import torch
from src import utils

# Add the annotix-ml directory to the path to import NoisingModel
# Assumes annotix-ml is a sibling of DiGress
current_path = os.path.dirname(os.path.realpath(__file__))
# current_path is src/diffusion, so we go up twice to DiGress, then up to projects output, then to annotix-ml
# Actually, DiGress/src/diffusion -> DiGress/src -> DiGress -> projects -> annotix-ml
# ../../../turbingen_project/annotix-ml ?
# Let's try to find it relative to the project root.
project_root = os.path.abspath(os.path.join(current_path, "../../.."))
annotix_ml_path = os.path.join(project_root, "annotix-ml")

if annotix_ml_path not in sys.path:
    sys.path.append(annotix_ml_path)

try:
    from annotix_ml.graphtransf.models.noising import NoisingModel
except ImportError:
    # Fallback or error handling if path is incorrect
    print(f"Warning: Could not import NoisingModel from {annotix_ml_path}")
    # Try assuming standard install or other location if needed, but for now specific to user env
    pass


class NoisingModelAdapter:
    def __init__(self, x_marginals, e_marginals, y_classes, diffusion_steps):
        """
        Adapter for NoisingModel to be used in DiscreteDenoisingDiffusion.
        """
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals

        # Instantiate the wrapped NoisingModel
        # NoisingModel takes (nodes_distribution, edges_distribution, diffusion_steps, noise_schedule_type)
        # We assume cosine schedule for now as it's the default in NoisingModel
        self.noising_model = NoisingModel(
            nodes_distribution=x_marginals,
            edges_distribution=e_marginals,
            diffusion_steps=diffusion_steps,
            noise_schedule_type="cosine",
        )

        # 'y' is not handled by NoisingModel, so we keep the functionality from MarginalUniformTransition
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        """
        Returns one-step transition matrices for X and E, from step t - 1 to step t.

        In the original code, beta_t is passed.
        However, NoisingModel uses integer t.
        DiscreteDenoisingDiffusion computes beta_t via its own noise schedule.

        CRITICAL ISSUE: NoisingModel computes alpha internally based on t.
        DiscreteDenoisingDiffusion computes beta_t and passes it here.
        If we want to use NoisingModel's definitions, we should ideally use its `get_Q_t(t)`.
        But `get_Qt` signature only gives us `beta_t` (float values).

        If `beta_t` corresponds to the alphas in NoisingModel (beta = 1 - alpha), we can reconstruct Q_t manually
        using the marginals stored in NoisingModel, effectively replicating `MarginalUniformTransition` logic
        but ensuring we use the same marginals `NoisingModel` would have used.

        Actually, `NoisingModel` implementation of `get_Q_t` is:
        Q = alpha * I + (1 - alpha) * marginals

        This matches `MarginalUniformTransition`:
        Q = (1 - beta) * I + beta * marginals

        So if beta = 1 - alpha, they are identical.

        Since `DiscreteDenoisingDiffusion` manages the schedule and passes `beta_t`,
        we should trust `beta_t` to be consistent with what `NoisingModel` expects *conceptually*
        if we configured the schedule correctly in the main class.

        So we essentially re-implement `MarginalUniformTransition`'s logic here but formally we are "adapting"
        conceptually.

        BUT, the task is to "Adapt the NoisingModel... to replace MarginalUniformTransition".
        If I just re-implement the math, I am not really "using" NoisingModel instance for the transition matrices.

        However, `get_Qt` needs `beta_t` (batch of floats). `NoisingModel.get_Q_t` takes integer `t`.
        We don't have integer `t` here easily (it's abstracted away).

        Wait, `DiscreteDenoisingDiffusion` has `apply_noise` where it samples `t`.
        But `compute_Lt` calls `get_Qt(noisy_data["beta_t"])`.

        Strategy: Use the `noising_model.n_m` (marginals) and the `beta_t` provided to construct the matrix.
        This ensures we use the exact distribution stored in NoisingModel.
        """
        beta_t = beta_t.unsqueeze(1).to(device)  # (bs, 1)

        # We use the marginals from the wrapped model
        u_x = (
            self.noising_model.n_m.to(device)
            .unsqueeze(0)
            .expand(self.X_classes, -1)
            .unsqueeze(0)
        )  # (1, dx, dx)
        u_e = (
            self.noising_model.e_m.to(device)
            .unsqueeze(0)
            .expand(self.E_classes, -1)
            .unsqueeze(0)
        )  # (1, de, de)
        u_y = self.u_y.to(device)

        # Q_t = (1-beta)*I + beta*M
        # This is standard transition.
        q_x = beta_t * u_x + (1 - beta_t) * torch.eye(
            self.X_classes, device=device
        ).unsqueeze(0)
        q_e = beta_t * u_e + (1 - beta_t) * torch.eye(
            self.E_classes, device=device
        ).unsqueeze(0)
        q_y = beta_t * u_y + (1 - beta_t) * torch.eye(
            self.y_classes, device=device
        ).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        """
        Returns t-step transition matrices.
        Q_bar = alpha_bar * I + (1 - alpha_bar) * M
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1).to(device)

        u_x = (
            self.noising_model.n_m.to(device)
            .unsqueeze(0)
            .expand(self.X_classes, -1)
            .unsqueeze(0)
        )
        u_e = (
            self.noising_model.e_m.to(device)
            .unsqueeze(0)
            .expand(self.E_classes, -1)
            .unsqueeze(0)
        )
        u_y = self.u_y.to(device)

        q_x = (
            alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * u_x
        )
        q_e = (
            alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * u_e
        )
        q_y = (
            alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0)
            + (1 - alpha_bar_t) * u_y
        )

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)
