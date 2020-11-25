from transformers import AlbertConfig, AlbertModel

# Initializing an ALBERT-xxlarge style configuration
albert_xxlarge_configuration = AlbertConfig()

model = AlbertModel(albert_xxlarge_configuration)

# Accessing the model configuration
configuration = model.config

print(configuration)