import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_astra_architecture():
    # Setup Canvas
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off') # Hide axes

    # --- STYLE CONFIG ---
    server_color = '#F5F5F5'
    client_color = '#DAE8FC'
    teacher_color = '#FFE6CC'
    student_color = '#D5E8D4' # Greenish/Blue
    border_color = '#333333'

    # ==========================================
    # 1. SERVER LAYER (Top)
    # ==========================================
    # Cloud/Server Box
    server_box = patches.FancyBboxPatch((2, 9), 6, 2.5, boxstyle="round,pad=0.1", 
                                        linewidth=2, edgecolor=border_color, facecolor=server_color)
    ax.add_patch(server_box)
    ax.text(5, 11.1, "FL SERVER (Cloud)", ha='center', fontsize=12, fontweight='bold')

    # Aggregator Process
    agg_box = patches.Rectangle((3, 10), 4, 0.8, linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(agg_box)
    ax.text(5, 10.4, "FedAvg\nAggregation", ha='center', va='center', fontsize=9)

    # Curriculum Scheduler (Diamond)
    # Drawing a diamond using Polygon
    diamond = patches.Polygon([[5, 10], [6, 9.5], [5, 9], [4, 9.5]], 
                              closed=True, linewidth=1, edgecolor='purple', facecolor='#E1D5E7')
    ax.add_patch(diamond)
    ax.text(5, 9.5, "Scheduler\n(Check t)", ha='center', va='center', fontsize=8, color='purple')

    # ==========================================
    # 2. NETWORK CONNECTIONS (Middle)
    # ==========================================
    # Downlink (Global Model + Gate Signal)
    ax.arrow(3.5, 9, 0, -1.8, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.text(2.5, 8.2, "Global Model ($w_t$)", fontsize=9, ha='right')
    
    # Gate Signal Arrow (Purple)
    ax.annotate("", xy=(5, 7.2), xytext=(5, 9),
                arrowprops=dict(arrowstyle="->", color='purple', lw=2, linestyle='--'))
    ax.text(5.1, 8.2, "Gate Signal\n($I_{gate}$)", fontsize=9, color='purple', ha='left')

    # Uplink (Local Updates)
    ax.arrow(6.5, 7.2, 0, 1.8, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.text(7.5, 8.2, "Local Update\n($\Delta w_k$)", fontsize=9, ha='left')

    # ==========================================
    # 3. CLIENT LAYER (Bottom)
    # ==========================================
    # Client Container
    client_box = patches.Rectangle((1.5, 0.5), 7, 6.7, linewidth=2, edgecolor=border_color, facecolor=client_color)
    ax.add_patch(client_box)
    ax.text(5, 6.8, "ASTRA CLIENT (Edge Device)", ha='center', fontsize=12, fontweight='bold')

    # Data Input
    data_cyl = patches.Ellipse((2.5, 3.5), 1, 0.6, linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(data_cyl)
    ax.text(2.5, 3.5, "Non-IID\nData", ha='center', va='center', fontsize=8)

    # --- MODELS ---
    
    # Student (Trainable)
    student_box = patches.Rectangle((4, 4.5), 2.5, 1.2, linewidth=2, edgecolor='#2E7D32', facecolor=student_color)
    ax.add_patch(student_box)
    ax.text(5.25, 5.1, "Student Model\n(Active Training)", ha='center', va='center', fontsize=9, fontweight='bold')

    # Teacher (Frozen) - Dashed
    teacher_box = patches.Rectangle((4, 2.0), 2.5, 1.2, linewidth=2, linestyle='--', edgecolor='#D79B00', facecolor=teacher_color)
    ax.add_patch(teacher_box)
    ax.text(5.25, 2.6, "Teacher Model\n(Frozen Global)", ha='center', va='center', fontsize=9, color='#D79B00')

    # --- THE GATE & LOSS ---

    # Gate Switch (Circle on the Teacher Output)
    gate_circle = patches.Circle((7.5, 2.6), 0.3, linewidth=2, edgecolor='purple', facecolor='white')
    ax.add_patch(gate_circle)
    ax.text(7.5, 2.6, "X", ha='center', va='center', fontsize=12, color='purple', fontweight='bold')
    ax.text(7.5, 1.9, "Curriculum\nGate", ha='center', fontsize=8, color='purple')

    # Connecting Gate Signal to Switch
    ax.annotate("", xy=(7.5, 2.9), xytext=(5, 7.2), # From top to switch
                arrowprops=dict(arrowstyle="->", color='purple', lw=1, linestyle='--'))

    # Loss Calculation (Summation)
    sum_circle = patches.Circle((7.5, 4.5), 0.4, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(sum_circle)
    ax.text(7.5, 4.5, "$\Sigma$", ha='center', va='center', fontsize=14)
    ax.text(8.2, 4.5, "Total\nLoss", ha='left', va='center', fontsize=10)

    # Arrows inside Client
    # Data -> Student
    ax.arrow(3.0, 3.8, 1.0, 1.0, head_width=0.1, color='black')
    # Data -> Teacher
    ax.arrow(3.0, 3.2, 1.0, -0.8, head_width=0.1, color='black')

    # Student -> Loss
    ax.arrow(6.5, 5.1, 0.6, -0.4, head_width=0.1, color='black')
    ax.text(6.8, 5.2, "$z_s$", fontsize=8)

    # Teacher -> Gate -> Loss
    ax.arrow(6.5, 2.6, 0.7, 0, head_width=0.1, color='black') # To Gate
    ax.arrow(7.5, 2.9, 0, 1.2, head_width=0.1, color='purple') # Gate to Loss
    ax.text(7.6, 3.5, "$\mathcal{L}_{KD}$", fontsize=9, color='purple')

    # Proximal Loss Arrow (Self-loop idea)
    ax.text(5.25, 6.0, "$\mathcal{L}_{Prox}$", ha='center', fontsize=9)
    ax.annotate("", xy=(5.25, 5.7), xytext=(5.25, 5.9), arrowprops=dict(arrowstyle="->"))

    plt.tight_layout()
    plt.savefig('astra_architecture_reference.png', dpi=150)
    print("✅ Reference Image Created: astra_architecture_reference.png")

if __name__ == "__main__":
    draw_astra_architecture()