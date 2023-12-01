
# ! KEY CODE
import datasets as huggingface_datasets
import torchvision

dataset = huggingface_datasets.load_dataset("zh-plus/tiny-imagenet")
training_dataset, validation_dataset = dataset['train'], dataset['valid']

transformations_list = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

def transform_datasets(examples):
    examples["pixel_values"] = []
    
    for image in examples['image']:
        transformed_image = transformations_list(image)
        examples['pixel_values'].append(transformed_image)
    
    return examples
 
training_dataset.set_transform(transform_datasets)
validation_dataset.set_transform(transform_datasets)