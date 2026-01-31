import graphviz

# --- CONFIGURATION (Academic Style) ---
dot = graphviz.Digraph(
    'astra_curriculum',
    comment='ASTRA Curriculum Logic',
    format='png'
)

# Set global graph attributes for a clean, professional look
dot.attr(
    rankdir='TB',       # Top-to-Bottom layout
    splines='ortho',    # Orthogonal lines (cleaner)
    fontname='Helvetica',
    fontsize='10'
)

# Set default node and edge styles
dot.attr('node', shape='box', style='rounded,filled', fillcolor='#FFFFFF', color='#000000', fontname='Helvetica', fontsize='10')
dot.attr('edge', fontname='Helvetica', fontsize='9', color='#000000', arrowhead='vee')

# --- NODES ---
# Main Inputs
dot.node('Input', 'Input Batch\n(x, y)', shape='parallelogram', fillcolor='#E0E0E0')
dot.node('GlobalModel', 'Global Model\n(w_t)', shape='cylinder', fillcolor='#E0E0E0')

# The Gatekeeper (Switching Logic)
dot.node('Curriculum', 'Curriculum Gate\n(Is Teacher Active?)', shape='diamond', fillcolor='#F0F0F0')

# Teacher Branch (Conditional)
dot.node('Teacher', 'Frozen Teacher\n(w_teach)', style='dashed,filled', fillcolor='#F8F8F8')
dot.node('KD_Loss', 'Distillation Loss\n(L_KD)', shape='ellipse')

# Student Branch (Always Active)
dot.node('Student', 'Active Student\n(w_k)')
dot.node('CE_Loss', 'CE Loss\n(L_CE)', shape='ellipse')
dot.node('Prox_Loss', 'Proximal Loss\n(L_Prox)', shape='ellipse')

# Final Loss & Update
dot.node('TotalLoss', 'Total Loss\n(L_ASTRA)', shape='box', style='filled', fillcolor='#D0D0D0')
dot.node('Update', 'Weight Update\n(w_k ← w_k - η∇L)', shape='box', style='filled', fillcolor='#D0D0D0')

# --- EDGES ---
# Flow from Inputs
dot.edge('Input', 'Student', label='x')
dot.edge('GlobalModel', 'Curriculum', label='Round t')

# Student Path (Always happens)
dot.edge('GlobalModel', 'Student', label='Initialize w_k', style='dashed')
dot.edge('Student', 'CE_Loss', label='Logits (z_s)')
dot.edge('Input', 'CE_Loss', label='Labels (y)')
dot.edge('GlobalModel', 'Prox_Loss', label='Reference (w_t)')
dot.edge('Student', 'Prox_Loss', label='Weights (w_k)')

# Teacher Path (Conditional)
dot.edge('Curriculum', 'Teacher', label='Yes (Bootcamp/Correction)', style='bold')
dot.edge('GlobalModel', 'Teacher', label='Initialize w_teach', style='dashed')
dot.edge('Input', 'Teacher', label='x')
dot.edge('Teacher', 'KD_Loss', label='Logits (z_t)')
dot.edge('Student', 'KD_Loss', label='Logits (z_s)')

# Combining Losses
dot.edge('CE_Loss', 'TotalLoss')
dot.edge('Prox_Loss', 'TotalLoss')
dot.edge('KD_Loss', 'TotalLoss', label='α_t * KL')
dot.edge('TotalLoss', 'Update')

# --- LEGEND (Crucial for academic clarity) ---
with dot.subgraph(name='cluster_legend') as c:
    c.attr(label='Legend', fontsize='9', style='rounded')
    c.node('L_Active', 'Active Path', shape='none', width='0', height='0')
    c.node('L_Conditional', 'Conditional Path', shape='none', width='0', height='0')
    c.edge('L_Active', 'L_Conditional', label='Standard Flow', style='solid')
    c.edge('L_Active', 'L_Conditional', label='Switchable Flow', style='bold')

# Render the image
output_path = 'astra_curriculum_diagram'
dot.render(output_path, view=False)
print(f"✅ Academic diagram saved: {output_path}.png")