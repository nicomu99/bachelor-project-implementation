import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def plot_3D_data(data, xyz_titles=['X', 'Y', 'Z'], ax_range=[-40, 40], labels=None, true_labels=None):
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scatter3d"}]],
        column_widths=[1],
        subplot_titles=[
            '3D', 
        ],
    )

    # if true labels is not none, get rid of cluster -1
    if true_labels is not None:
        data = data.loc[true_labels != -1]
        # true_labels = true_labels[true_labels != -1]
        labels = labels[true_labels != -1]
    print(np.unique(labels, return_counts=True))

    # --------------- 3D scatter plot -------------------
    if labels is None:
        trace_3d = go.Scatter3d(
            x=data.loc[:, xyz_titles[0]], y=data.loc[:, xyz_titles[1]], z=data.loc[:, xyz_titles[2]],
            mode='markers',
            marker=dict(size=5, color='red'),
            hoverinfo='none',
            showlegend=False,
        )
        fig.add_trace(trace_3d, row=1, col=1)
    else:
        for l_i in np.unique(labels):
            if l_i != -1:
                trace_3d = go.Scatter3d(
                    x=data.loc[labels==l_i, xyz_titles[0]], y=data.loc[labels==l_i, xyz_titles[1]], z=data.loc[labels==l_i, xyz_titles[2]],
                    mode='markers',
                    marker=dict(size=5),
                    hoverinfo='none',
                    showlegend=True,
                    name=f'Cluster {l_i}'
                )
                fig.add_trace(trace_3d, row=1, col=1)
    
    # 3d position
    plt_kwargs = dict(showbackground=False, showline=False, zeroline=True, zerolinecolor='grey', zerolinewidth=2, 
                      showgrid=True, showticklabels=True, color='black',
                      linecolor='black', linewidth=1,  gridcolor='rgba(100,100,100,0.5)')

    xaxis=dict(**plt_kwargs, title=xyz_titles[0], range=ax_range)
    yaxis=dict(**plt_kwargs, title=xyz_titles[1], range=ax_range)
    zaxis=dict(**plt_kwargs, title=xyz_titles[2], range=ax_range)

    # Finalize layout
    fig.update_layout(
        title="",
        #width=800,
        #height=800,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend= dict(itemsizing='constant', font_color='black'),
        # 3D plot
        scene=dict(
            xaxis=dict(xaxis),
            yaxis=dict(yaxis),
            zaxis=dict(zaxis)
        )
    )
    fig.write_html(f"/home/nico/Desktop/evaluation_data/simulated_cluster_{'sigma' if true_labels is not None else 'true'}.html")


def scatter_plot_matrix(data, labels, cols):

    # create a scatterplot matrix in matplotlib
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    for i in range(3):
        for j in range(3):
            axes[i, j].scatter(data[cols[j]], data[cols[i]], s=1, c=labels, cmap='tab10', alpha=0.5)

            axes[i, j].set_xlim(-1000, 1000)
            axes[i, j].set_ylim(-1000, 1000)
            # get rid of right and top spines
            # only show ticks on the most left and bottom axes
            if i == 2:
                axes[i, j].set_xlabel(cols[j])
                axes[i, j].set_xticks([-1000, -500, 0, 500, 1000])
            else:
                axes[i, j].set_xticks([])
            if j == 0:
                axes[i, j].set_ylabel(cols[i])
                axes[i, j].set_yticks([-1000, -500, 0, 500, 1000])
            else:
                axes[i, j].set_yticks([])
    # fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    plt.show()
    fig.savefig('/home/nico/Desktop/evaluation_data/scatterplot_matrix.png')