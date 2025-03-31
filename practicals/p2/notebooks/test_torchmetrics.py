# %%
import torch
import torchmetrics
import torchmetrics.functional.classification
import torchmetrics.functional.segmentation

# %%
dice_score = torchmetrics.functional.segmentation.dice_score
f1_score = torchmetrics.functional.classification.multiclass_f1_score
precision = torchmetrics.functional.classification.multiclass_precision
recall = torchmetrics.functional.classification.multiclass_recall

output = torch.tensor([[[1, 1, 2, 0, 1, 2]]])
target = torch.tensor([[[0, 1, 2, 1, 1, 1]]])

print(f"Dice Score: {dice_score(output, target, num_classes=3, average='macro', input_format='index')}")
print(f"F1 Score: {f1_score(output, target, num_classes=3, average='macro')}")
print(f"Precision: {precision(output, target, num_classes=3, average='macro')}")
print(f"Recall: {recall(output, target, num_classes=3, average='macro')}")

# %%
output = torch.tensor([[[1, 0]],[[2, 1]]])
target = torch.tensor([[[1, 0]],[[1, 2]]])

print(output.shape)
print(target.shape)

print(f"Dice Score: {dice_score(output, target, num_classes=3, average='macro', input_format='index', include_background=False).nanmean(-1)}")
print(f"F1 Score: {f1_score(output, target, num_classes=3, average='macro', ignore_index=0)}")
print(f"Precision: {precision(output, target, num_classes=3, average='macro', ignore_index=0)}")
print(f"Recall: {recall(output, target, num_classes=3, average='macro', ignore_index=0)}")


