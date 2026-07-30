import torch
import numpy as np
import plotly.graph_objects as go

def random_direction(model):
    """Generates filter-normalized random directions."""
    d = [torch.randn_like(p) for p in model.parameters()]
    for di, p in zip(d, model.parameters()):
        if di.dim() <= 1:          
            di.zero_()
        else:
            for df, pf in zip(di, p):   
                df.mul_(pf.norm() / (df.norm() + 1e-10))
    return d

@torch.no_grad()
def eval_loss(model, loss_fn, loader, device, max_batches=15):
    """Evaluates the loss over a subset of the loader for speed."""
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: break
        x, y = x.to(device), y.to(device)
        total += loss_fn(model(x), y).item() * x.size(0)
        n += x.size(0)
    return total / n

def loss_landscape(model, loss_fn, loader, device, d1, d2, resolution=15, span=1.0):
    """Computes the 3D loss landscape for a specific state using predefined directions."""
    orig = [p.clone() for p in model.parameters()]
    alphas = np.linspace(-span, span, resolution)
    betas = alphas
    Z = np.zeros((resolution, resolution))

    def set_point(a, b):
        with torch.no_grad():
            for p, o, x1, x2 in zip(model.parameters(), orig, d1, d2):
                p.copy_(o).add_(x1, alpha=a).add_(x2, alpha=b)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            set_point(a, b)
            Z[i, j] = eval_loss(model, loss_fn, loader, device)
            
    with torch.no_grad():
        for p, o in zip(model.parameters(), orig): p.copy_(o)
        
    return alphas, betas, Z

def generate_interactive_landscape(clf, snapshots: dict, val_loader, device, out_path, resolution=15):
    """
    Master function: Assumes clf.model is currently loaded with PEAK weights.
    Generates the interactive Plotly HTML file across all provided snapshots.
    
    Args:
        snapshots: Dictionary mapping string labels (e.g. "Phase 1", "Epoch 10") to state_dicts.
    """
    if not snapshots:
        return False
        
    print(f"Generating interactive loss landscape ({len(snapshots)} frames) -> {out_path}")
    
    # 1. Generate reference directions from the currently loaded PEAK model
    d1_ref = random_direction(clf.model)
    d2_ref = random_direction(clf.model)
    
    Z_list = []
    frame_names = list(snapshots.keys())
    
    # 2. Iterate through snapshots and collect landscapes
    for name, state_dict in snapshots.items():
        clf.model.load_state_dict(state_dict)
        clf.model.to(device)
        
        alphas, betas, Z = loss_landscape(
            clf.model, clf.compute_loss, val_loader, device,
            d1=d1_ref, d2=d2_ref, resolution=resolution
        )
        Z_list.append(Z)

    # 3. Restore the peak model weights
    final_key = list(snapshots.keys())[-1]
    clf.model.load_state_dict(snapshots[final_key]) 
    
    # 4. Plotly Visualization
    z_min, z_max = min([Z.min() for Z in Z_list]), max([Z.max() for Z in Z_list])
    fig = go.Figure(data=[go.Surface(z=Z_list[0], x=alphas, y=betas, cmin=z_min, cmax=z_max, colorscale='viridis')])
    
    frames = [go.Frame(data=[go.Surface(z=Z, cmin=z_min, cmax=z_max)], name=name) for Z, name in zip(Z_list, frame_names)]
    fig.frames = frames

    sliders = [{
        "active": 0, "currentvalue": {"prefix": ""}, "pad": {"t": 50},
        "steps": [{"args": [[f.name], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
                   "label": f.name, "method": "animate"} for f in frames]
    }]

    fig.update_layout(
        title="Interactive Loss Landscape Evolution",
        scene=dict(
            xaxis_title="Alpha (d1)", yaxis_title="Beta (d2)", zaxis_title="Loss",
            zaxis=dict(range=[z_min, z_max])
        ),
        updatemenus=[{"buttons": [
            {"args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}], "label": "Play", "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}], "label": "Pause", "method": "animate"}
        ], "direction": "left", "pad": {"r": 10, "t": 87}, "showactive": False, "type": "buttons", "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"}],
        sliders=sliders, width=800, height=800
    )
    
    fig.write_html(out_path)
    return True