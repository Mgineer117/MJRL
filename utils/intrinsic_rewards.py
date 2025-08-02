import glob
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from extractor.base.mlp import NeuralNet
from utils.rl import call_env
from utils.sampler import OnlineSampler


class IntrinsicRewardFunctions(nn.Module):
    def __init__(self, logger, writer, args):
        super(IntrinsicRewardFunctions, self).__init__()

        # === Parameter saving === #
        self.num_trials = 2_000

        self.extractor_env = call_env(
            deepcopy(args), verbose=False, spawn_agent_random=True
        )
        self.logger = logger
        self.writer = writer
        self.args = args

        self.current_timesteps = 0
        self.loss_dict = {}

        # === MAKE ENV === #
        self.num_rewards = self.args.num_options
        self.extractor_mode = "ALLO"

        print(f"[INFO] Using {args.extractor_mode} for intrinsic rewards.")
        if args.extractor_mode == "FE":
            self.define_fe_extractor()
            self.define_fe_eigenvectors()
        elif args.extractor_mode == "ALLO":
            self.define_allo_extractor()
            self.define_allo_eigenvectors()
        else:
            raise NotImplementedError(
                f"Extractor mode {args.extractor_mode} is not implemented."
            )
        print(
            f"[INFO] {self.extractor_mode} Extractor defined with feature dimension: {self.args.feature_dim}"
        )

        print(f"[INFO] Eigenvectors defined with {len(self.eigenvectors)} vectors.")
        # normalizer is not good for off-policy learning
        # self.define_intrinsic_reward_normalizer()

    def forward(
        self, states: torch.Tensor, next_states: torch.Tensor, i: int
    ) -> torch.Tensor:
        with torch.no_grad():
            feature, _ = self.extractor(states)
            next_feature, _ = self.extractor(next_states)
            difference = next_feature - feature

            eigenvector_sign = self.eigenvector_signs[i]
            eigenvector = self.eigenvectors[i]

            intrinsic_rewards = eigenvector_sign * (
                difference @ eigenvector.unsqueeze(-1)
            )

        # === INTRINSIC REWARD NORMALIZATION === #
        if hasattr(self, "reward_rms"):
            # drnd has its own normalizer in itself
            self.reward_rms[i].update(intrinsic_rewards.cpu().numpy())
            var_tensor = torch.as_tensor(
                self.reward_rms[i].var,
                device=intrinsic_rewards.device,
                dtype=intrinsic_rewards.dtype,
            )
            intrinsic_rewards = intrinsic_rewards / (torch.sqrt(var_tensor) + 1e-8)

        return intrinsic_rewards

    def define_fe_extractor(self):
        from extractor.base.cnn import CNN
        from extractor.extractor import Extractor
        from trainer.extractor_trainer import ExtractorTrainer
        from utils.cnn_architecture import get_cnn_architecture

        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists(f"model/FE/{self.args.env_name}"):
            os.makedirs(f"model/FE/{self.args.env_name}")

        # === CREATE FEATURE EXTRACTOR === #
        encoder_architecture, decoder_architecture = get_cnn_architecture(self.args)

        feature_network = CNN(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            feature_dim=self.args.feature_dim,
            encoder_architecture=encoder_architecture,
            decoder_architecture=decoder_architecture,
            device=self.args.device,
        )

        # === DEFINE LEARNING METHOD FOR EXTRACTOR === #
        extractor = Extractor(
            network=feature_network,
            extractor_lr=self.args.extractor_lr,
            epochs=self.args.fe_extractor_epochs,
            batch_size=1024,
            device=self.args.device,
        )

        # Step 1: Search for .pth files in the directory
        model_dir = f"model/FE/{self.args.env_name}/"
        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))

        if not pth_files:
            print(
                f"[INFO] No existing model found in {model_dir}. Training from scratch."
            )
            epochs = 0
            model_path = os.path.join(
                model_dir,
                f"FE_{self.args.fe_extractor_epochs}.pth",
            )
        else:
            print(f"[INFO] Found {len(pth_files)} .pth files in {model_dir}")
            epochs = []
            valid_files = []

            for pth_file in pth_files:
                filename = os.path.basename(pth_file)
                parts = filename.replace(".pth", "").split("_")
                if len(parts) != 2:
                    print(f"[WARNING] Skipping malformed file: {filename}")
                    continue

                _, epoch_str = parts
                try:
                    epoch = int(epoch_str)
                    epochs.append(epoch)
                    valid_files.append(filename)
                except ValueError:
                    print(f"[WARNING] Failed to parse file: {filename}")
                    continue

            matching = [(e, filename) for e, filename in zip(epochs, valid_files)]

            max_epoch, _ = max(matching, key=lambda x: x[0])
            idx = epochs.index(max_epoch)
            filename = matching[idx][-1]
            model_path = os.path.join(model_dir, filename)
            print(f"[INFO] Loading model from: {model_path} (epoch {max_epoch})")

            extractor.load_state_dict(
                torch.load(model_path, map_location=self.args.device)
            )
            extractor.to(self.args.device)
            epochs = max_epoch  # set current epoch

        if epochs < self.args.fe_extractor_epochs:
            self.collect_samples()
            trainer = ExtractorTrainer(
                extractor=extractor,
                logger=self.logger,
                writer=self.writer,
                epochs=self.args.fe_extractor_epochs - epochs,
                seed=42,  # The result of ALLO should be seed invariant
            )

            final_timesteps = trainer.train(self.batch)
            self.current_timesteps += final_timesteps

            torch.save(extractor.state_dict(), model_path)

        self.extractor = extractor

    def define_allo_extractor(self):
        from extractor.extractor import ALLO
        from policy.layers.building_blocks import MLP
        from trainer.extractor_trainer import ExtractorTrainer

        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists(f"model/ALLO/{self.args.env_name}"):
            os.makedirs(f"model/ALLO/{self.args.env_name}")

        # === CREATE FEATURE EXTRACTOR === #
        input_dim = (
            np.prod(self.args.state_dim)
            if self.args.state_mask is None
            else len(self.args.state_mask)
        )
        feature_network = MLP(
            input_dim=input_dim,
            hidden_dims=[256, 256, 256, 256],
            output_dim=self.args.feature_dim,
            activation=nn.ReLU(),
        )

        # === DEFINE LEARNING METHOD FOR EXTRACTOR === #
        extractor = ALLO(
            network=feature_network,
            extractor_lr=self.args.extractor_lr,
            epochs=self.args.allo_extractor_epochs,
            batch_size=1024,
            discount_sampling_factor=self.args.discount_sampling_factor,
            state_mask=self.args.state_mask,
            device=self.args.device,
        )

        # Step 1: Search for .pth files in the directory
        model_dir = f"model/ALLO/{self.args.env_name}/"
        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))

        if not pth_files:
            print(
                f"[INFO] No existing model found in {model_dir}. Training from scratch."
            )
            epochs = 0
            model_path = os.path.join(
                model_dir,
                f"ALLO_{self.args.allo_extractor_epochs}_{self.args.discount_sampling_factor}.pth",
            )
        else:
            print(f"[INFO] Found {len(pth_files)} .pth files in {model_dir}")
            epochs = []
            discount_factors = []
            valid_files = []

            for pth_file in pth_files:
                filename = os.path.basename(pth_file)
                parts = filename.replace(".pth", "").split("_")
                if len(parts) != 3:
                    print(f"[WARNING] Skipping malformed file: {filename}")
                    continue

                _, epoch_str, discount_str = parts
                try:
                    epoch = int(epoch_str)
                    discount = float(discount_str)
                    epochs.append(epoch)
                    discount_factors.append(discount)
                    valid_files.append(filename)
                except ValueError:
                    print(f"[WARNING] Failed to parse file: {filename}")
                    continue

            if self.args.discount_sampling_factor not in discount_factors:
                print(
                    f"[INFO] No model with discount factor {self.args.discount_sampling_factor} found. Starting fresh."
                )
                epochs = 0
                model_path = os.path.join(
                    model_dir,
                    f"ALLO_{self.args.allo_extractor_epochs}_{self.args.discount_sampling_factor}.pth",
                )
            else:
                matching = [
                    (e, f, filename)
                    for e, f, filename in zip(epochs, discount_factors, valid_files)
                    if f == self.args.discount_sampling_factor
                ]

                max_epoch, _, _ = max(matching, key=lambda x: x[0])
                idx = epochs.index(max_epoch)
                filename = matching[idx][-1]
                model_path = os.path.join(model_dir, filename)
                print(
                    f"[INFO] Loading model from: {model_path} (epoch {max_epoch}, discount {self.args.discount_sampling_factor})"
                )

                extractor.load_state_dict(
                    torch.load(model_path, map_location=self.args.device)
                )
                extractor.to(self.args.device)
                epochs = max_epoch  # set current epoch

        if epochs < self.args.allo_extractor_epochs:
            self.collect_samples()
            trainer = ExtractorTrainer(
                extractor=extractor,
                logger=self.logger,
                writer=self.writer,
                epochs=self.args.allo_extractor_epochs - epochs,
                seed=42,  # The result of ALLO should be seed invariant
            )

            final_timesteps = trainer.train(self.batch)
            self.current_timesteps += final_timesteps

            torch.save(extractor.state_dict(), model_path)

        self.extractor = extractor

    def define_allo_eigenvectors(self):
        # === Define eigenvectors === #
        # ALLO does not have explicit eigenvectors.
        # Instead, we make list that contains the eigenvector index and sign
        if self.args.num_options % 2 == 1:
            print(
                "[Warning] The num_options should be an even number, otherwise this may result in corrupt eigenvectors."
            )

        # create a onehot_vector of [1, 0, 0,] using self.args.num_option
        self.eigenvectors = [
            F.one_hot(torch.tensor(n // 2), num_classes=self.args.feature_dim).float()
            for n in range(2, self.args.num_options + 2)
        ]

        for i, eig_vec in enumerate(self.eigenvectors):
            self.eigenvectors[i] = eig_vec.to(self.args.device)
        self.eigenvector_signs = [2 * (n % 2) - 1 for n in range(self.args.num_options)]

        if self.args.env_name in ("FourRooms-v0", "Maze-v0", "Maze-v1"):
            heatmaps = self.extractor_env.get_rewards_heatmap(
                self.extractor, self.eigenvector_signs, self.eigenvectors
            )
            self.logger.write_images(
                step=self.current_timesteps, images=heatmaps, logdir="Image/Heatmaps"
            )

    def define_fe_eigenvectors(self):
        # === Define eigenvectors === #
        # ALLO does not have explicit eigenvectors.
        # Instead, we make list that contains the eigenvector index and sign
        if self.args.num_options % 2 == 1:
            print(
                "[Warning] The num_options should be an even number, otherwise this may result in corrupt eigenvectors."
            )

        # get features
        if not hasattr(self, "batch"):
            self.collect_samples()
        with torch.no_grad():
            features, _ = self.extractor(self.batch["states"])
        _, _, Vt = torch.linalg.svd(features, full_matrices=False)
        Vt = Vt.to(self.args.device)

        if self.args.option_method == "top":
            # create a onehot_vector of [1, 0, 0,] using self.args.num_option
            self.eigenvectors = [
                Vt[n // 2, :].to(self.args.device) for n in range(self.args.num_options)
            ]
        elif self.args.option_method == "cvs":
            # cluster Vt
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=self.args.num_options)
            kmeans.fit(Vt.cpu().detach().numpy())
            cluster_centers = kmeans.cluster_centers_
            self.eigenvectors = [
                torch.tensor(cluster_centers[n // 2, :]).to(self.args.device)
                for n in range(self.args.num_options)
            ]
        elif self.args.option_method == "crs":
            from sklearn.cluster import KMeans

            with torch.no_grad():
                next_features, _ = self.extractor(self.batch["next_states"])
            difference = next_features - features
            intrinsic_rewards = difference @ Vt.T

            kmeans = KMeans(n_clusters=self.args.num_options)
            kmeans.fit(intrinsic_rewards.cpu().detach().numpy())
            cluster_centers = kmeans.cluster_centers_
            cluster_labels = kmeans.labels_

            self.eigenvectors = []
            for n in range(self.args.num_options):
                self.eigenvectors.append(
                    torch.mean(Vt[cluster_labels == (n // 2)], axis=0).to(
                        self.args.device
                    )
                )

        elif self.args.option_method == "trs":
            if not hasattr(self, "batch"):
                self.collect_samples()

            # Round down to ensure integer count
            num_top = int(0.25 * self.args.num_options)
            num_crs = self.args.num_options - num_top

            # Optional: Make sure both are even numbers (if required for later logic)
            if num_top % 2 != 0:
                num_top -= 1
                num_crs += 1  # preserve total count

            if num_crs % 2 != 0:
                num_crs -= 1
                num_top += 1  # preserve total count

            # Top: one-hot
            top_vectors = [Vt[n // 2, :].to(self.args.device) for n in range(num_top)]

            # CRS: SVD-based
            with torch.no_grad():
                next_features, _ = self.extractor(self.batch["next_states"])
            difference = next_features - features
            intrinsic_rewards = difference @ Vt.T

            kmeans = KMeans(n_clusters=num_crs)
            kmeans.fit(intrinsic_rewards.cpu().detach().numpy())
            cluster_centers = kmeans.cluster_centers_
            cluster_labels = kmeans.labels_

            crs_vectors = []
            for n in range(num_crs):
                crs_vectors.append(
                    torch.mean(Vt[cluster_labels == (n // 2)], axis=0).to(
                        self.args.device
                    )
                )

            self.eigenvectors = top_vectors + crs_vectors

        elif self.args.option_method == "uniform":
            # Uniformly spaced one-hot vectors over feature_dim
            indices = torch.linspace(
                0, self.args.feature_dim - 1, steps=self.args.num_options // 2
            ).long()

            self.eigenvectors = [
                F.one_hot(i, num_classes=self.args.feature_dim).float() for i in indices
            ]

        for i, eig_vec in enumerate(self.eigenvectors):
            self.eigenvectors[i] = eig_vec.to(self.args.device)
        self.eigenvector_signs = [2 * (n % 2) - 1 for n in range(self.args.num_options)]

        if self.args.env_name in ("FourRooms-v0", "Maze-v0", "Maze-v1"):
            heatmaps = self.extractor_env.get_rewards_heatmap(
                self.extractor, self.eigenvector_signs, self.eigenvectors
            )
            self.logger.write_images(
                step=self.current_timesteps, images=heatmaps, logdir="Image/Heatmaps"
            )

    def define_intrinsic_reward_normalizer(self):
        from utils.wrapper import RunningMeanStd

        self.reward_rms = []
        for _ in range(self.args.num_options):
            self.reward_rms.append(RunningMeanStd(shape=(1,)))

    def collect_samples(self):
        from policy.elementary_policy.uniform_random import UniformRandom

        uniform_random_policy = UniformRandom(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            is_discrete=self.args.is_discrete,
            device=self.args.device,
        )
        sampler = OnlineSampler(
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            episode_len=self.args.episode_len,
            batch_size=self.num_trials * self.args.episode_len,
            verbose=False,
        )
        self.batch, _ = sampler.collect_samples(
            env=self.extractor_env,
            policy=uniform_random_policy,
            seed=self.args.seed,
        )
