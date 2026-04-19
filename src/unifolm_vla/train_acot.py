from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from torchvision import transforms as T

from unifolm_vla.datasets.acot_adapter import ACOTAdapterConfig, ACOTToUnifoLMAdapter
from unifolm_vla.datasets.acot_dataset import ACOTIterableDataset, TransformedIterableDataset
from unifolm_vla.datasets.acot_batch_transform import ACOTBatchTransform


def collate_fn(batch):
 # Minimal collate_fn for testing - just return the batch
 # The real collate_fn is handled by the training script
 return batch


def build_dataset(task_roots):
 adapter = ACOTToUnifoLMAdapter(
 ACOTAdapterConfig(
 use_left_wrist=False,
 use_right_wrist=True,
 project_state_to_21=True,
 project_action_to_21=True,
 )
 )

 dataset = ACOTIterableDataset(
 task_roots=task_roots,
 adapter=adapter,
 )
 return dataset


def main():
 task_roots = [
 "/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth/open_door",
 ]

 # For testing with dummy processor - replace with real Qwen processor
 # in the actual training script
 from transformers import AutoProcessor
 processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat")

 dataset = build_dataset(task_roots)
 batch_transform = ACOTBatchTransform(
 processor=processor,
 use_wrist_image=True,
 use_proprio=True,
 )
 transformed_dataset = TransformedIterableDataset(dataset, batch_transform)

 loader = DataLoader(
 transformed_dataset,
 batch_size=4,
 num_workers=0,
 collate_fn=collate_fn,
 )

 print("Testing data loading...")
 for i, batch in enumerate(loader):
 print(f"\nBatch {i}:")
 print(f"  Sample keys: {list(batch[0].keys())}")
 if "input_ids" in batch[0]:
 print(f"  input_ids shape: {batch[0]['input_ids'].shape}")
 if "pixel_values" in batch[0]:
 print(f"  pixel_values shape: {batch[0]['pixel_values'].shape}")
 if "actions" in batch[0]:
 print(f"  actions shape: {batch[0]['actions'].shape}")
 if "proprio" in batch[0]:
 print(f"  proprio shape: {batch[0]['proprio'].shape}")
 break

 print("\n✅ Data loading test passed!")
 print(f"Total tasks: {len(task_roots)}")
 print("Ready to connect to the main training loop.")


if __name__ == "__main__":
 main()
