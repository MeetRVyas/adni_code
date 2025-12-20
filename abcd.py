import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class NativeGradCAM:
    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model.eval()
        self.reshape_transform = reshape_transform
        self.activations = None
        self.gradients = None
        self.hooks = []
        
        # 1. Register Hooks (Equivalent to creating the multi-output Keras model)
        self.hooks.append(target_layer.register_forward_hook(self.save_activation))
        self.hooks.append(target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        pred_index = output.argmax(dim=1)
        score = output[:, pred_index]
        score.backward()
        grads = self.gradients
        fmaps = self.activations
        if self.reshape_transform:
            grads = self.reshape_transform(grads)
            fmaps = self.reshape_transform(fmaps)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * fmaps, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()[0, 0]

    def remove_hooks(self):
        for h in self.hooks: h.remove()

def generate_gradcam_plot(model, input_tensor, original_img_np, target_layer, reshape_transform=None, alpha=0.4):
    """
    Visualization logic matching your cv2 snippet.
    """
    cam_engine = NativeGradCAM(model, target_layer, reshape_transform)
    
    try:
        # Get Heatmap (0 to 1 float)
        heatmap = cam_engine(input_tensor)
        
        # Resize to match original image
        heatmap = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
        
        # Scale to 0-255 and ColorMap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay (cv2.addWeighted)
        # Ensure original image is uint8 (0-255)
        if original_img_np.max() <= 1.0:
            original_img_np = np.uint8(255 * original_img_np)
        else:
            original_img_np = np.uint8(original_img_np)
            
        superimposed_img = cv2.addWeighted(original_img_np, 1, heatmap_colored, alpha, 0)
        
        return superimposed_img
        
    finally:
        cam_engine.remove_hooks()