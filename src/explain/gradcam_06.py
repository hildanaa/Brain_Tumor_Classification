import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.h1.remove()
        self.h2.remove()

    def generate(self, x, class_idx=None):
        """
        x: tensor [1,3,H,W]
        returns heatmap [H,W] in [0,1]
        """
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        # activations: [1,C,h,w], gradients: [1,C,h,w]
        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True) # [1,1,h,w]
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.detach()