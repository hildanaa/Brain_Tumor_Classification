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
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients            # [1,C,h,w]
        acts = self.activations           # [1,C,h,w]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.detach()

    def generate_smooth(self, x, class_idx=None, n_samples=8, noise_sigma=0.10):
        """
        Smooth Grad-CAM: average CAM over noisy inputs.
        n_samples: 8–16 works well
        noise_sigma: 0.05–0.15 typical
        """
        cams = []
        for _ in range(n_samples):
            noise = torch.randn_like(x) * noise_sigma
            cam = self.generate(x + noise, class_idx=class_idx)
            cams.append(cam)
        cam_avg = torch.stack(cams, dim=0).mean(dim=0)
        cam_avg -= cam_avg.min()
        cam_avg /= (cam_avg.max() + 1e-8)
        return cam_avg.detach()

    def generate_pp(self, x, class_idx=None):
        """
        Grad-CAM++: usually more localized than standard Grad-CAM.
        """
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients          # [1,C,h,w]
        acts  = self.activations        # [1,C,h,w]

        grads_pow2 = grads ** 2
        grads_pow3 = grads ** 3
        eps = 1e-8

        # sum over spatial dims
        sum_acts_grads_pow3 = (acts * grads_pow3).sum(dim=(2,3), keepdim=True)

        # alpha numerator/denominator
        alpha = grads_pow2 / (2 * grads_pow2 + sum_acts_grads_pow3 + eps)  # [1,C,h,w]

        # weights: sum of alpha * relu(grads) over spatial dims
        weights = (alpha * F.relu(grads)).sum(dim=(2,3), keepdim=True)     # [1,C,1,1]

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.detach()