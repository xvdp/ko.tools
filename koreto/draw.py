
"""
basic plotly utility
"""
from typing import Tuple, Optional
import numpy as np
import plotly.graph_objs as go


def new_fig(fig: Optional[go.Figure] = None) -> go.Figure:
    """creates new figure if none exists"""
    if fig is None:
        fig = go.Figure(layout = go.Layout(scene=dict( aspectmode='data')))
    return fig


def draw_points(points: np.ndarray,
                fig: Optional[go.Figure] = None,
                size: float = 1,
                number: bool = False,
                name: Optional[str] = None,
                show: bool = False) -> go.Figure:
    """ scatter points
    Args
        points  (ndarray (N,3))
    """
    if number:
        kw = {'mode':'markers+text', 'text':np.arange(len(points)).tolist()}
    else:
        kw = {'mode':'markers'}
    if name is not None:
        kw['name'] = name

    fig = new_fig(fig)
    if points.shape[-1] != 3:
        x,y,z = points[0], points[1], points[2]
    else:
        x,y,z = points[:, 0], points[:, 1], points[:, 2]
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
        **kw, marker=dict(size=size),))

    if show:
        fig.show()
    return fig

def draw_axis(matrix: Optional[np.ndarray] = None,
              origin: Optional[np.ndarray] = None,
              fig: Optional[go.Figure] = None,
              norm: Optional[float] = 1.,
              show: bool = False,
              **kwargs) -> go.Figure:
    """ draw 3x3 matrix, default itentity
    Args:
        matrix  (ndarray (3,3) [None]) : default identity
        origin  (ndarray (3,) [None]) : default 0
        fig     (Figure [None]) if None: create new
        norm    (float [1.]) norm and scale, if None, dont normalize
        show    (bool [False]) if true show figure

    """
    fig = new_fig(fig)

    if origin is None:
        origin = np.array([0,0,0.], dtype=np.float32)
    if matrix is None:
        matrix = np.array([[1.,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
    assert origin.shape == (3,) and matrix.shape == (3,3)

    if norm is not None:
        for i in range(3):
            matrix[i] = norm*matrix[i]/np.linalg.norm(matrix[i])
    matrix = matrix + origin
    ox,oy,oz = origin
    x1,x2,x3,y1,y2,y3,z1,z2,z3=matrix.reshape(-1)

    fig.add_trace(go.Scatter3d(x=[ox, x1], y=[oy, x2], z=[oz, x3],
                               mode='lines+text', text="X",name="X",
                               surfacecolor='green'), **kwargs)
    fig.add_trace(go.Scatter3d(x=[ox, y1], y=[oy, y2], z=[oz, y3],
                               mode='lines+text', text="Y",name="Y",
                               surfacecolor='red'), **kwargs)
    fig.add_trace(go.Scatter3d(x=[ox, z1], y=[oy, z2], z=[oz, z3],
                               mode='lines+text', text="Z",name="Z",
                               surfacecolor='blue'), **kwargs)
    if show:
        fig.show()
    return fig
#showlegend=False

def draw_vector(vector: np.ndarray,
                origin: Optional[np.ndarray] = None,
                fig: Optional[go.Figure] = None,
                name: Optional[str] = None,
                show: bool = False,
                **kwargs) -> go.Figure:
    """ draw and name vector
    """
    if origin is None:
        origin = np.array([0,0,0.], dtype=np.float32)
    vector = vector + origin
    fig = new_fig(fig)
    kwg = {'mode':'lines'} if name is None else {'mode':'lines+text', 'text':name, 'name':name}
    kwg.update(kwargs)
    fig.add_trace(go.Scatter3d(x=[origin[0], vector[0]],
                               y=[origin[1], vector[1]],
                               z=[origin[2], vector[2]],
                               **kwg))
    if show:
        fig.show()
    return fig



def sort_points_by_distance(points:np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate the squared Euclidean distances
    dist_sq = np.sum((points - center) ** 2, axis=1)
    sorted_indices = np.argsort(dist_sq)
    return sorted_indices, dist_sq

def zscore_keep(dist: np.ndarray, stds: float = 2) -> np.ndarray:
    """ threshold by std """
    return np.where(np.abs((dist - dist.mean())/dist.std()) < stds)[0]

def irq_keep(distances: np.ndarray) -> np.ndarray:
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # outliers = np.where((distances < lower_bound) | (distances > upper_bound))
    return np.where((distances > lower_bound) & (distances < upper_bound))[0]
