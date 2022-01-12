from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import networkx as nx
import numpy as np

def build_similarity(vectors):
    """
    builds similarity matrix of vectors using cosine similarity
    @input: vector embeddings of items
    @returns similarity matrix

    """
    sims = cosine_similarity(vectors, vectors)
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i <= j :
                sims[i, j] = False

    return sims


def build_graph(sims, items, searches=[]):
    """
    builds knowledge graph of ingredients
    @input sims: similarity matrix
    """

    idxs = np.argwhere(sims > 0.55)

    # Build a graph with edge between two items if they're similar
    G = nx.Graph()

    for index in idxs:
        G.add_edge(
            items[index[0]], items[index[1]],
            weight=sims[index[0], index[1]]
        )

    # spring_layout in short, keeps everthing that are related are close together
    # and everything that are disimilar are pulled away
    positions = nx.spring_layout(G)
    nx.set_node_attributes(G, name='position', values=positions)
    weight_values = nx.get_edge_attributes(G, 'weight')
    
    edge_x = []
    edge_y = []
    weights = []
    ave_x, ave_y = [], []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['position']
        x1, y1 = G.nodes[edge[1]]['position']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        ave_x.append(np.mean([x0, x1]))
        ave_y.append(np.mean([y0, y1]))
        weights.append(f'{edge[0]}, {edge[1]}: {weight_values[(edge[0], edge[1])]}')

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        opacity=0.7,
        line=dict(width=2, color='White'),
        hoverinfo='text',
        mode='lines')

    edge_trace.text = weights


    node_x = []
    node_y = []
    sizes = []
    for node in G.nodes():
        x, y = G.nodes[node]['position']
        node_x.append(x)
        node_y.append(y)
        if node in searches:
            sizes.append(50)
        else:
            sizes.append(15)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            line=dict(color='White'),
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Picnic',
            reversescale=False,
            color=[],
            opacity=0.9,
            size=sizes,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    invisible_similarity_trace = go.Scatter(
        x=ave_x, y=ave_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=[],
            opacity=0,
        )
    )

    invisible_similarity_trace.text=weights

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(adjacencies[0])

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
    data=[edge_trace, node_trace, invisible_similarity_trace],
    layout=go.Layout(
        title='Knowledge Graph of Recipe Ingredients',
        template='plotly_white',
        titlefont_size=20,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(
                text="Created By: <a href='https://github.com/gborn'> Glad Nayak</a>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) 
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))

    return fig
