import plotly.graph_objects as go
from mc_dagprop import EventTimestamp, GenericDelayGenerator, SimActivity, SimContext, SimEvent, Simulator


def make_context() -> SimContext:
    events = [SimEvent("0", EventTimestamp(0.0, 0.0, 0.0)), SimEvent("1", EventTimestamp(0.0, 0.0, 0.0))]
    activities = {(0, 1): SimActivity(60.0, 1)}
    precedence_list = [(1, [(0, 0)])]
    return SimContext(events, activities, precedence_list, max_delay=0.0)


def sample_realized(generator: GenericDelayGenerator, n_samples: int = 10_000) -> list[float]:
    ctx = make_context()
    sim = Simulator(ctx, generator)
    seeds = list(range(n_samples))
    results = sim.run_many(seeds)
    # extract realized timestamp of event "1"
    return [res.realized[1] for res in results]


def visualize_constant(factor: float, n_samples: int = 10_000) -> go.Figure:
    gen = GenericDelayGenerator()
    gen.add_constant(activity_type=1, factor=factor)
    data = sample_realized(gen, n_samples)

    fig = go.Figure(go.Histogram(x=data, nbinsx=50, name=f"constant * {factor}"))
    fig.update_layout(
        title=f"Constant Delay (factor={factor})", xaxis_title="Realized time of node 1", yaxis_title="Count"
    )
    return fig


def visualize_exponential(lambd: float, max_scale: float, n_samples: int = 10_000) -> go.Figure:
    gen = GenericDelayGenerator()
    gen.add_exponential(1, lambd, max_scale)
    data = sample_realized(gen, n_samples)

    fig = go.Figure(go.Histogram(x=data, nbinsx=50, name=f"exp(lambda={lambd}), max={max_scale}"))
    fig.update_layout(
        title=f"Exponential Delay (labda={lambd}, max_scale={max_scale})",
        xaxis_title="Realized time of node 1",
        yaxis_title="Count",
    )
    return fig


def visualize_gamma(shape: float, scale: float, n_samples: int = 10_000) -> go.Figure:
    gen = GenericDelayGenerator()
    gen.add_gamma(1, shape, scale)
    data = sample_realized(gen, n_samples)

    fig = go.Figure(go.Histogram(x=data, nbinsx=50, name=f"gamma(k={shape}, m={scale})"))
    fig.update_layout(
        title=f"Gamma Delay (shape={shape}, scale={scale})", xaxis_title="Realized time of node 1", yaxis_title="Count"
    )
    return fig


if __name__ == "__main__":
    # Render each in sequence (in Jupyter they'd display inline;when run as script they'll open your browser):
    fig1 = visualize_constant(factor=0.5)
    fig1.show()

    fig2 = visualize_exponential(lambd=0.1, max_scale=0.3)
    fig2.show()

    fig3 = visualize_gamma(shape=2.0, scale=0.5)
    fig3.show()
