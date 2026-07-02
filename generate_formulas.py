import os
import matplotlib.pyplot as plt

def render_formula(latex_str, save_path):
    # Setup figure with solid white background to make it visible on both GitHub light and dark modes
    fig = plt.figure(figsize=(12, 1.5), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.text(0.5, 0.5, f"${latex_str}$", size=22, color='black', ha='center', va='center')
    ax.axis('off')
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save with solid white background
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Generated formula image at: {save_path}")

# Formula 1: Backbone
f1 = r"w_k \propto \left(\frac{n_k}{N}\right)^\beta \times \text{Reliability}_k \times \text{SampleConfidence}_k \times \text{RareFactor}_k"
render_formula(f1, "pictures/formula_backbone.png")

# Formula 2: Classifier Head
f2 = r"\alpha_{k,c} \propto \text{Reliability}_{k,c} \times \text{ClassConfidence}_{k,c} \times \text{RiskWeight}_c"
render_formula(f2, "pictures/formula_classifier.png")
