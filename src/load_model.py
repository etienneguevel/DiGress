from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from diffusion_model_discrete import DiscreteDenoisingDiffusion


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in ["sbm", "comm20", "planar"]:
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import (
            PlanarSamplingMetrics,
            SBMSamplingMetrics,
            Comm20SamplingMetrics,
        )
        from analysis.visualization import NonMolecularVisualization

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config["name"] == "sbm":
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config["name"] == "comm20":
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = (
            TrainAbstractMetricsDiscrete()
            if cfg.model.type == "discrete"
            else TrainAbstractMetrics()
        )
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos
            )
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": train_metrics,
            "sampling_metrics": sampling_metrics,
            "visualization_tools": visualization_tools,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }

    elif dataset_config["name"] in ["qm9", "guacamol", "moses"]:
        from metrics.molecular_metrics import (
            TrainMolecularMetrics,
            SamplingMolecularMetrics,
        )
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if dataset_config["name"] == "qm9":
            from datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(
                cfg=cfg,
                train_dataloader=datamodule.train_dataloader(),
                dataset_infos=dataset_infos,
                evaluate_dataset=False,
            )
        elif dataset_config["name"] == "guacamol":
            from datasets import guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == "moses":
            from datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == "discrete" and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(
                cfg.model.extra_features, dataset_info=dataset_infos
            )
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        if cfg.model.type == "discrete":
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(
            cfg.dataset.remove_h, dataset_infos=dataset_infos
        )

        model_kwargs = {
            "dataset_infos": dataset_infos,
            "train_metrics": train_metrics,
            "sampling_metrics": sampling_metrics,
            "visualization_tools": visualization_tools,
            "extra_features": extra_features,
            "domain_features": domain_features,
        }
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if hasattr(cfg, "model_path") and cfg.model_path:
        model_path = cfg.model_path
    else:
        raise ValueError(
            "Please provide a model_path via CLI: +model_path=/path/to/ckpt"
        )

    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        model_path,
        weights_only=False,
        **model_kwargs,
    )

    print(model)
    print(type(model.model))
    print(model.model)

    if hasattr(cfg, "saving_path") and cfg.saving_path:
        saving_path = cfg.saving_path
    else:
        raise ValueError(
            "Please provide a saving_path via CLI: +saving_path=output.pth"
        )

    torch.save(model.model.gnn, Path(__file__).parent.parent / saving_path)


if __name__ == "__main__":
    main()
