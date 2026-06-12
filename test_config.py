from src.utils.config_loader import ConfigLoader

config = ConfigLoader.load("configs/train_config.yaml")

print("\n===== CONFIG LOADED =====\n")

print(config)

print("\n===== ACCESS TEST =====\n")

print("Project Name:", config.project_name)
print("Batch Size:", config.training.batch_size)
print("Learning Rate:", config.training.learning_rate)
print("Sequence Length:", config.dataset.sequence_length)
print("Backbone:", config.model.backbone)