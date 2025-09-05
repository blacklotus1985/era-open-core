import networkx as nx
import matplotlib.pyplot as plt

def build_lineage(models):
    G = nx.DiGraph()
    for m in models:
        G.add_node(m['name'], role=m.get('role','candidate'), kind=m.get('kind','fine-tuned'))
        if m.get('parent'):
            G.add_edge(m['parent'], m['name'], relation='fine-tune')
    return G

def save_graph(G, path: str, title: str = "ERA Lineage"):
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="#cce5ff")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'relation'))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
