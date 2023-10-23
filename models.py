from typing import Tuple
from tqdm import tqdm

import torch
from torch import tensor
from torch.utils.data import DataLoader
import timm

import numpy as np
from sklearn.metrics import roc_auc_score

from utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params
import cv2
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path  # Import Path from pathlib to create a folder

class KNNExtractor(torch.nn.Module):
	def __init__(
		self,
		backbone_name : str = "resnet18",
		out_indices : Tuple = None,
		pool_last : bool = False,
	):
		super().__init__()

		self.feature_extractor = timm.create_model(
			backbone_name,
			out_indices=out_indices,
			features_only=True,
			pretrained=True,
		)
		for param in self.feature_extractor.parameters():
			param.requires_grad = False
		self.feature_extractor.eval()
		
		self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
		self.backbone_name = backbone_name # for results metadata
		self.out_indices = out_indices

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		print("Device used:", self.device)
		self.feature_extractor = self.feature_extractor.to(self.device)
			
	def __call__(self, x: tensor):
		with torch.no_grad():
			feature_maps = self.feature_extractor(x.to(self.device))
		feature_maps = [fmap.to("cpu") for fmap in feature_maps]
		if self.pool:
			# spit into fmaps and z
			return feature_maps[:-1], self.pool(feature_maps[-1])
		else:
			return feature_maps

	def fit(self, _: DataLoader):
		raise NotImplementedError

	def predict(self, _: tensor):
		raise NotImplementedError

	def evaluate(self, test_dl: DataLoader, dataset_name: str) -> Tuple[float, float]:
		"""Calls predict step for each test sample."""
		image_preds = []
		image_labels = []
		image_ids = []  # New list to store image IDs
		pixel_preds = []
		pixel_labels = []

		for idx, (sample, mask, label) in enumerate(tqdm(test_dl, **get_tqdm_params())):  # Added idx
			z_score, fmap = self.predict(sample)
			
			image_preds.append(z_score.numpy())
			image_labels.append(label)
			image_ids.append(idx)  # Store the image ID (in this case, the index)

			pixel_preds.extend(fmap.flatten().numpy())
			pixel_labels.extend(mask.flatten().numpy())
			
		image_labels = np.stack(image_labels)
		image_preds = np.stack(image_preds)
		# Create a folder with the dataset name if it doesn't exist
		folder_path = Path(f"./{dataset_name}_least_anomalous_images")
		folder_path.mkdir(parents=True, exist_ok=True)
		sorted_indices = np.argsort(image_preds)[:5]  # Sort in ascending order and take the bottom 5

		sorted_indices = np.argsort(image_preds)[:5]
		for i in sorted_indices:
			image_id = image_ids[i]
			sample_image = test_dl.dataset[image_id][0]  # Assuming the image is the first element in your dataset
			img_pil = transforms.ToPILImage()(sample_image.squeeze(0))
			img_pil.save(folder_path / f"least_anomalous_image_{image_id}_zscore_{image_preds[i]:.2f}.png")
		bottom_5_image_ids = [image_ids[i] for i in sorted_indices]
		print("Top 5 image IDs with lowest z-scores:", bottom_5_image_ids)
		image_rocauc = roc_auc_score(image_labels, image_preds)
		pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)

		return image_rocauc, pixel_rocauc

	def get_parameters(self, extra_params : dict = None) -> dict:
		return {
			"backbone_name": self.backbone_name,
			"out_indices": self.out_indices,
			**extra_params,
		}

class PatchCore(KNNExtractor):
	def __init__(
		self,
		f_coreset: float = 0.01, # fraction the number of training samples
		backbone_name : str = "resnet18",
		coreset_eps: float = 0.90, # sparse projection parameter
	):
		super().__init__(
			backbone_name=backbone_name,
			out_indices=(2,3),
		)
		self.f_coreset = f_coreset
		self.coreset_eps = coreset_eps
		self.image_size = 224
		self.average = torch.nn.AvgPool2d(3, stride=1)
		self.blur = GaussianBlur(4)
		self.n_reweight = 3

		self.patch_lib = []
		self.resize = None

	def fit(self, train_dl):
		for sample, _ in tqdm(train_dl, **get_tqdm_params()):
			feature_maps = self(sample)

			if self.resize is None:
				largest_fmap_size = feature_maps[0].shape[-2:]
				self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
			resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
			patch = torch.cat(resized_maps, 1)
			patch = patch.reshape(patch.shape[1], -1).T

			self.patch_lib.append(patch)

		self.patch_lib = torch.cat(self.patch_lib, 0)

		if self.f_coreset < 1:
			self.coreset_idx = get_coreset_idx_randomp(
				self.patch_lib,
				n=int(self.f_coreset * self.patch_lib.shape[0]),
				eps=self.coreset_eps,
			)
			self.patch_lib = self.patch_lib[self.coreset_idx]

	def predict(self, sample):		
		feature_maps = self(sample)
		resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
		patch = torch.cat(resized_maps, 1)
		patch = patch.reshape(patch.shape[1], -1).T

		dist = torch.cdist(patch, self.patch_lib)
		min_val, min_idx = torch.min(dist, dim=1)
		s_idx = torch.argmax(min_val)
		s_star = torch.max(min_val)

		# reweighting
		m_test = patch[s_idx].unsqueeze(0) # anomalous patch
		m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0) # closest neighbour
		w_dist = torch.cdist(m_star, self.patch_lib) # find knn to m_star pt.1
		_, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False) # pt.2
		# equation 7 from the paper
		m_star_knn = torch.linalg.norm(m_test-self.patch_lib[nn_idx[0,1:]], dim=1)
		# Softmax normalization trick as in transformers.
		# As the patch vectors grow larger, their norm might differ a lot.
		# exp(norm) can give infinities.
		D = torch.sqrt(torch.tensor(patch.shape[1]))
		w = 1-(torch.exp(s_star/D)/(torch.sum(torch.exp(m_star_knn/D))))
		s = w*s_star

		# segmentation map
		s_map = min_val.view(1,1,*feature_maps[0].shape[-2:])
		s_map = torch.nn.functional.interpolate(
			s_map, size=(self.image_size,self.image_size), mode='bilinear'
		)
		s_map = self.blur(s_map)

		return s, s_map
	
	def get_parameters(self):
		return super().get_parameters({
			"f_coreset": self.f_coreset,
			"n_reweight": self.n_reweight,
		})

