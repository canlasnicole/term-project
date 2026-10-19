"""Module 2.3 - Image classification with pretrained ResNet-18 (ImageNet, 1000 classes).
Based on the professor's CNN_YOLO.ipynb (Reference A). No training: inference only."""
import threading

import requests
import torch
from torchvision import models, transforms

from .config import HTTP_TIMEOUT

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
DOG_IDS = range(151, 269)      # ImageNet class 151-268 = dog breeds
CAT_IDS = range(281, 286)      # ImageNet class 281-285 = domestic cats

device = torch.device("cpu")   # free servers have no GPU
_model, _categories = None, None
_lock = threading.Lock()

# Same preprocessing as the notebook: 224x224 + ImageNet mean / std
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load():
    """Lazy-load: the model is loaded on the first request only, then reused."""
    global _model, _categories
    with _lock:
        if _model is None:
            weights = models.ResNet18_Weights.DEFAULT
            _model = models.resnet18(weights=weights).to(device)
            _model.eval()                                   # inference mode
            try:   # class names from the professor's URL
                r = requests.get(LABELS_URL, timeout=HTTP_TIMEOUT)
                r.raise_for_status()
                _categories = [line.strip() for line in r.text.splitlines()]
            except Exception:  # fallback: names stored inside torchvision
                _categories = list(weights.meta.get("categories",
                                                    [f"Class {i}" for i in range(1000)]))
    return _model, _categories


def classify(img) -> dict:
    model, categories = _load()
    batch = preprocess(img).unsqueeze(0).to(device)       # shape [1, 3, 224, 224]
    with torch.no_grad():                                  # no gradients = faster, less memory
        output = model(batch)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_p, top_id = torch.topk(probs, 5)

    top5 = [{"class_id": int(i), "class_name": categories[int(i)], "probability": float(p)}
            for p, i in zip(top_p, top_id)]
    best = top5[0]

    # Use class-ID ranges, not keywords ("cat" would also match "catamaran")
    animal = "dog" if best["class_id"] in DOG_IDS else "cat" if best["class_id"] in CAT_IDS else None
    head = f"Prediction: {best['class_name']} ({best['probability'] * 100:.1f}%)"
    msg = f"{head} -> It looks like a DOG or a CAT!" if animal else \
          f"{head} -> Classified as another class."
    return {**best, "top5": top5, "is_dog_or_cat": animal is not None,
            "animal": animal, "message": msg}
