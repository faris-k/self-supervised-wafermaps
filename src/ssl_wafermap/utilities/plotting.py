import seaborn as sns


# Modified from https://github.com/sparks-baird/mat_discover/blob/73b33bcf8a8e897d8d6dc0f334508ef935e8fc96/mat_discover/utils/plotting.py#L254
def matplotlibify(fig, font_size=24, width_inches=5.5, height_inches=3.5, dpi=142):
    # make it look more like matplotlib
    # modified from: https://medium.com/swlh/formatting-a-plotly-figure-with-matplotlib-style-fa56ddd97539)
    font_dict = dict(family="Arial", size=font_size, color="black")

    fig.update_layout(
        font=font_dict,
        # plot_bgcolor="white",
        width=width_inches * dpi,
        height=height_inches * dpi,
        margin=dict(r=40, t=20, b=10),
    )

    fig.update_yaxes(
        showline=True,  # add line at x=0
        linecolor="black",  # line color
        linewidth=2.4,  # line size
        ticks="inside",  # ticks outside axis
        tickfont=font_dict,  # tick label font
        mirror="allticks",  # add ticks to top/right axes
        tickwidth=2.4,  # tick width
        tickcolor="black",  # tick color
    )

    fig.update_xaxes(
        showline=True,
        showticklabels=True,
        linecolor="black",
        linewidth=2.4,
        ticks="inside",
        tickfont=font_dict,
        mirror="allticks",
        tickwidth=2.4,
        tickcolor="black",
    )

    width_default_px = fig.layout.width
    targ_dpi = 300
    scale = width_inches / (width_default_px / dpi) * (targ_dpi / dpi)

    return fig, scale


def init_seaborn_style():
    sns.set_style(
        "darkgrid",
        {
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.bottom": True,
            "ytick.left": True,
            "axes.linewidth": "2.4",
            "axes.edgecolor": "black",
        },
    )

    sns.set_context("paper", font_scale=1.75)
