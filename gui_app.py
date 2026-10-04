#!/usr/bin/env python3
"""
CNN comparison GUI: open one image and see the prediction of each of the five trained networks.

Run on a LOCAL machine (needs a screen). Does NOT train anything: it only loads the saved
weights produced by the Colab training notebook.

Folder layout (next to this file):
    gui_app.py
    models/
        LeNet-5.pt
        AlexNet-lite.pt
        VGG-16-lite.pt
        PlacesNet-lite.pt
        ResNet-18-lite.pt
        meta.json            <- must be INSIDE models/

Install:   pip install torch torchvision pillow
           (Linux only: sudo apt install python3-tk)
Run:       python gui_app.py
           python gui_app.py --models path/to/models
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

IMG = 32
tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# =============================================================================
# Network definitions (identical to the training notebook)
# =============================================================================
def make_lenet(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, 6, 5), nn.Tanh(),
        nn.AvgPool2d(2),
        nn.Conv2d(6, 16, 5), nn.Tanh(),
        nn.AvgPool2d(2),
        nn.Conv2d(16, 120, 5), nn.Tanh(),
        nn.Flatten(),
        nn.Linear(120, 84), nn.Tanh(),
        nn.Linear(84, num_classes))


def make_alexnet_lite(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(),
        nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(),
        nn.Conv2d(96, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Dropout(0.5), nn.Linear(1024, 256), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, num_classes))


def vgg_block(c_in, c_out, n_convs, bn=False):
    layers = []
    for i in range(n_convs):
        layers.append(nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1))
        if bn:
            layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


def make_vgg_lite(num_classes=10):
    return nn.Sequential(
        vgg_block(3, 16, 2, bn=True), vgg_block(16, 32, 2, bn=True), vgg_block(32, 64, 3, bn=True),
        vgg_block(64, 128, 3, bn=True), vgg_block(128, 128, 3, bn=True),
        nn.Flatten(),
        nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, num_classes))


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                                          nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def res_stages(base):
    layers, c_in = [], base
    for mult, stride in [(1, 1), (2, 2), (4, 2), (8, 2)]:
        c_out = base * mult
        layers += [BasicBlock(c_in, c_out, stride), BasicBlock(c_out, c_out)]
        c_in = c_out
    return layers


def make_resnet_lite(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
        *res_stages(16),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes))


def builders(K):
    """PlacesNet-lite = AlexNet-lite body with a 365-way output (labels 0..K-1 only)."""
    return {
        "LeNet-5":        lambda: make_lenet(K),
        "AlexNet-lite":   lambda: make_alexnet_lite(K),
        "VGG-16-lite":    lambda: make_vgg_lite(K),
        "PlacesNet-lite": lambda: make_alexnet_lite(365),
        "ResNet-18-lite": lambda: make_resnet_lite(K),
    }


# =============================================================================
# Loading the trained networks
# =============================================================================
def load_networks(models_dir):
    meta_path = os.path.join(models_dir, "meta.json")
    if not os.path.exists(meta_path):
        sys.exit(f"Cannot find '{meta_path}'.\n"
                 "Put the downloaded 'models' folder next to gui_app.py, with meta.json INSIDE it.")
    with open(meta_path) as f:
        classes = json.load(f)["classes"]
    K = len(classes)

    nets = {}
    for name, build in builders(K).items():
        path = os.path.join(models_dir, name.replace(" ", "_") + ".pt")
        if not os.path.exists(path):
            sys.exit(f"Missing weights file: {path}\nAll five .pt files are needed.")
        model = build()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        nets[name] = model
    return classes, nets


def predict_all(img, classes, nets):
    """img: PIL RGB image. Returns [(network, label, confidence)] and the majority vote."""
    K = len(classes)
    x = tf(img).unsqueeze(0)
    rows, votes = [], {}
    with torch.no_grad():
        for name, model in nets.items():
            p = F.softmax(model(x), dim=1)[0]
            idx = int(p.argmax())
            label = classes[idx] if idx < K else f"(unused output {idx})"
            rows.append((name, label, p[idx].item()))
            if idx < K:
                votes[label] = votes.get(label, 0) + 1
    top = max(votes, key=votes.get) if votes else None
    return rows, top, (votes[top] if top else 0)


# =============================================================================
# The window
# =============================================================================
def run_gui(models_dir):
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import Image, ImageTk
    except ImportError as e:
        sys.exit(f"{e}\nInstall Pillow (pip install pillow) and Tkinter "
                 "(Linux: sudo apt install python3-tk).")

    classes, nets = load_networks(models_dir)

    root = tk.Tk()
    root.title("CNN comparison - one image, five networks")
    root.geometry("760x420")

    left = ttk.Frame(root, padding=10)
    left.pack(side="left", fill="y")
    right = ttk.Frame(root, padding=10)
    right.pack(side="left", fill="both", expand=True)

    preview = ttk.Label(left, text="No image loaded", width=34, anchor="center", relief="groove")
    preview.pack(pady=(0, 8), ipady=100)
    path_var = tk.StringVar(value="")
    ttk.Label(left, textvariable=path_var, wraplength=260).pack()

    cols = ("network", "prediction", "confidence")
    table = ttk.Treeview(right, columns=cols, show="headings", height=len(nets))
    for c, w in zip(cols, (140, 170, 90)):
        table.heading(c, text=c.capitalize())
        table.column(c, width=w, anchor="w")
    table.pack(fill="x")
    verdict = tk.StringVar(value="")
    ttk.Label(right, textvariable=verdict, font=("TkDefaultFont", 11, "bold"),
              wraplength=420).pack(pady=12, anchor="w")
    ttk.Label(right, text="PlacesNet-lite picks the largest of its 365 outputs; if one of the "
                          "unused outputs wins, it is shown as 'unused output'.",
              wraplength=420, foreground="grey").pack(anchor="w")

    state = {"photo": None}

    def open_image():
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Cannot open image", str(e))
            return
        thumb = img.copy()
        thumb.thumbnail((260, 260))
        state["photo"] = ImageTk.PhotoImage(thumb)
        preview.configure(image=state["photo"], text="")
        path_var.set(os.path.basename(path))

        rows, top, n = predict_all(img, classes, nets)
        for row in table.get_children():
            table.delete(row)
        for name, label, conf in rows:
            table.insert("", "end", values=(name, label, f"{conf * 100:.1f}%"))
        if top:
            verdict.set(f"Majority vote: {top}  ({n} of {len(nets)} networks)")
        else:
            verdict.set("No network predicted one of your classes.")

    ttk.Button(left, text="Open image...", command=open_image).pack(pady=10)
    ttk.Button(left, text="Quit", command=root.destroy).pack()
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="models", help="folder with the .pt files and meta.json")
    # parse_known_args ignores the extra "-f kernel.json" that Jupyter adds
    run_gui(ap.parse_known_args()[0].models)