from src.build_graph import build_similarity, build_graph
import streamlit as st
import numpy as np
import os


PAGE_CONFIG = {"page_title":"App by Glad Nayak","page_icon":":smiley:","layout":"wide"}
st.set_page_config(**PAGE_CONFIG)

def main():
    """
    Render UI on web app, build and display knowledge graph
    """

    st.title('Knowledge Graph of Food Ingredients')
    # load items, and their vector embeddings
    items = np.loadtxt('models/items.txt', dtype=str)
    vectors = np.load('models/vectors.npy')

    # select ingredients
    selected_items = st.sidebar.multiselect('Select Ingredients', options=list(items))
    if selected_items:
        sims = build_similarity(vectors)
        fig = build_graph(sims, items, selected_items)   
        fig.update_layout(title_text="title", margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=800)
        st.plotly_chart(fig, use_container_width=True, )

    else:
        sims = build_similarity(vectors)
        fig = build_graph(sims, items,  '')
        fig.update_layout(title_text="title", margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=800)
        st.plotly_chart(fig, use_container_width=True, )

    
    # select vector weights
    selected_weights = st.sidebar.radio('Choose weights', options=['Word2Vec', 'FastText'])
    if selected_weights:
        vectors = np.load('models/ft_vectors.npy')

    
    st.sidebar.markdown("## How it works? :tomato:")
    st.sidebar.write(
        "Search ingredients and select weights to see similar ingredients, or explore existing clusters of food ingredients."
    )

if __name__ == '__main__':
	main()