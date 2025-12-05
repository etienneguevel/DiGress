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

    def get_Qt(self, beta_t, device, t=None):
        """
        Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Uses NoisingModel.get_Q_t(t).
        """
        if t is None:
            # Fallback to manual construction if t is not provided (should not happen with new call sites)
            # Or raise error to be strict.
            # For now, let's try to support the legacy way if needed, or just crash if t is missing.
            # Given the user context, we should use get_Q_t.
            raise ValueError("t (timestep) is required for NoisingModelAdapter.get_Qt")

        # Implementation:
        t = t.squeeze()  # (bs,) or scalar
        if t.ndim == 0:
            t = t.unsqueeze(0)

        bs = t.shape[0]
        device = self.noising_model.alphas.device  # ensure device match

        # Loop implementation to be safe and compliant
        q_x_list = []
        q_e_list = []
        q_y_list = []

        for i in range(bs):
            idx = t[i].item()
            idx = max(0, min(idx, self.noising_model.T - 1))

            qx, qe = self.noising_model.get_Q_t(idx)
            q_x_list.append(qx)
            q_e_list.append(qe)

            # y transition
            # qt_y = alpha I + (1-alpha) u_y ...
            alpha = self.noising_model.alphas[idx]
            qy = (
                alpha * torch.eye(self.y_classes, device=device)
                + (1 - alpha) * self.u_y
            )
            q_y_list.append(qy)

        q_x = torch.stack(q_x_list).to(device)
        q_e = torch.stack(q_e_list).to(device)
        q_y = torch.stack(q_y_list).to(device)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device, t=None):
        """
        Returns t-step transition matrices.
        Uses NoisingModel.get_Q_bar_t(t).
        """
        if t is None:
            raise ValueError(
                "t (timestep) is required for NoisingModelAdapter.get_Qt_bar"
            )

        t = t.squeeze()
        if t.ndim == 0:
            t = t.unsqueeze(0)

        bs = t.shape[0]
        # device = self.noising_model.alphas_bar.device

        q_x_list = []
        q_e_list = []
        q_y_list = []

        for i in range(bs):
            idx = t[i].item()
            idx = max(0, min(idx, self.noising_model.T - 1))

            qx, qe = self.noising_model.get_Q_bar_t(idx)
            q_x_list.append(qx)
            q_e_list.append(qe)

            # y transition
            alpha_bar = self.noising_model.alphas_bar[idx]
            qy = (
                alpha_bar * torch.eye(self.y_classes, device=device)
                + (1 - alpha_bar) * self.u_y
            )
            q_y_list.append(qy)

        q_x = torch.stack(q_x_list).to(device)
        q_e = torch.stack(q_e_list).to(device)
        q_y = torch.stack(q_y_list).to(device)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)
