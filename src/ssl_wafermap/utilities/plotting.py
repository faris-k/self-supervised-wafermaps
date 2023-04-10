import matplotlib.pyplot as plt
import seaborn as sns


# Modified from https://github.com/sparks-baird/mat_discover/blob/73b33bcf8a8e897d8d6dc0f334508ef935e8fc96/mat_discover/utils/plotting.py#L254
# Which itself is modified from: https://medium.com/swlh/formatting-a-plotly-figure-with-matplotlib-style-fa56ddd97539)
def matplotlibify(fig, font_size=24, width_inches=5.5, height_inches=3.5, dpi=142):
    """Make a plotly figure look like a matplotlib figure

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The plotly figure to modify
    font_size : int, optional
        Font size of tick labels, by default 24
    width_inches : float, optional
        Width of figure in inches, by default 5.5
    height_inches : float, optional
        Height of figure in inches, by default 3.5
    dpi : int, optional
        Dots per inch of figure, by default 142

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The modified plotly figure
    """
    font_dict = dict(family="Arial", size=font_size, color="black")

    fig.update_layout(
        font=font_dict,
        template="seaborn",
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
    """Initialize a custom seaborn style for plotting"""
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


def ax_thiccify(ax, width=1.5):
    """Make the axes lines thicker

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes to thiccify
    width : float, optional
        Line width of axes, by default 1.5
    """
    ax.tick_params(width=width)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(width)


def legend_thiccify(
    ax, line_width=4, legend_title=None, bbox_to_anchor=(1.0, 1.0), **kwargs
):
    """Make the legend lines of a seaborn plot thicker
    and move it outside the plot

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes to modify
    line_width : float, optional
        Line width of the legend lines, by default 4
    legend_title : str, optional
        Title of the legend, by default None
    bbox_to_anchor : tuple, optional
        Bounding box of the legend, by default (1.0, 1.0)
    **kwargs
        Additional keyword arguments to pass to sns.move_legend
    """
    legend = plt.legend()
    for line in legend.get_lines():
        line.set_linewidth(line_width)
    sns.move_legend(
        ax, "upper left", bbox_to_anchor=bbox_to_anchor, title=legend_title, **kwargs
    )
