# model_viz — Design Notes

A flexible, modular visualizer for ML models (LLMs first, image models later)
built on PyQt. The framework is model-agnostic; each model family plugs in via
an adapter, and each visualization plugs in as a self-contained folder.

The idea is to allow the user to visualize any model more easily. These wouldn't
just be limited to transformers, but any future model that can arise. Thus the design
is for extendability for any future models. 

---

## Guiding principles

- **Functional where reasonable.** Pure helpers, dataclasses, closures for
  hooks. Classes only where Qt or adapter contracts naturally call for them.
- **No god files.** Every file has a single, named responsibility.
- **Plug-and-play.** Adding a new model = one adapter file. Adding a new
  visualization = one folder under `viz/visualizers/`. No edits to `core/`.
- **Model-agnostic core.** `core/` must not assume text, sequences, or any
  specific architecture.

---

## Folder layout

model_viz/
│
├── core/                              # Framework contracts (model-agnostic)
│   ├── layer.py                       # LayerLike, VisualizableLayer, LayerGroup
│   ├── adapter.py                     # ModelAdapter protocol; declares accepted_inputs
│   ├── registry.py                    # Global registries: models, visualizers, inputs
│   │
│   ├── input/                         # Input wrappers
│   │   ├── base.py                    # InputBase protocol
│   │   ├── text_input.py
│   │   └── image_input.py
│   │
│   └── gui/                           # GUI pieces shared across all models
│       │
│       ├── sidebar/                   # Left-hand navigation
│       │   ├── sidebar.py             # Top-level Sidebar QWidget; emits signals
│       │   ├── model_selector.py      # "Select a model" collapsable menu
│       │   ├── layer_tree.py          # Recursive LayerLike → nested collapsable menus
│       │   └── viz_list.py            # Visualizers compatible with selected layer
│       │
│       └── tabs/                      # Right-hand visualization area
│           ├── tab_area.py            # QTabWidget host; open/close/switch
│           ├── tab.py                 # Single tab: renders one visualizer, or empty state
│           └── empty_state.py         # "Select a model" placeholder widget
│
├── adapters/                          # One file/folder per model; plug-and-play
│   └── transformer_adapter.py         # accepted_inputs = (TextInput,)
│
├── viz/                               # Visualization layer
│   ├── visualizer_base.py             # VisualizerBase; compatible_with(layer)
│   ├── main_window.py                 # Thin: wires sidebar ↔ tab_area
│   │
│   ├── components/                    # Reusable widgets (any visualizer may import)
│   │   ├── heatmap.py
│   │   ├── pca_plot.py
│   │   └── tensor_grid.py
│   │
│   └── visualizers/                   # Each visualizer is a folder
│       └── attention_viz/
│           ├── viz.py                 # The QWidget rendered inside a tab
│           └── components/            # Private to this visualizer
│
├── examples/
│   └── xor_transformer/               # End-to-end demo model
│
└── main.py                            # Entry point

---

---

## Core concepts

| Concept         | Lives in              | Declares                                                  |
| --------------- | --------------------- | --------------------------------------------------------- |
| `LayerLike`     | `core/layer.py`       | `prev/next/curr`, `weights()`, `outputs()`, `children()`  |
| `LayerGroup`    | `core/layer.py`       | Same API as a layer; recursive: `OneOrMore(Layer \| LayerGroup)` |
| `InputBase`     | `core/input/base.py`  | How a raw user input becomes a model-ready tensor         |
| `ModelAdapter`  | `adapters/`           | `accepted_inputs: tuple[type[InputBase], ...]`            |
| `VisualizerBase`| `viz/visualizer_base.py` | `compatible_with(layer) -> bool`                       |
| `Sidebar`       | `core/gui/sidebar/`   | Recursively renders `LayerLike` as nested collapsable menus       |

### Layer protocol

```python
# core/layer.py
class LayerLike(Protocol):
    name: str
    next_layer: Optional[LayerLike]
    prev_layer: Optional[LayerLike]

    @property
    def curr_layer(self) -> nn.Module: ...
    def weights(self) -> Dict[str, torch.Tensor]: ...
    def outputs(self) -> Any: ...
    def children(self) -> list[LayerLike]: ...   # [] for leaves
```

VisualizableLayer.children() returns []. LayerGroup.children() returns
its sublayers. The sidebar uses children() to recurse — no special-casing
of "group vs leaf".

Adapter protocol

```
# core/adapter.py
class ModelAdapter(Protocol):
    name: str
    accepted_inputs: tuple[type[InputBase], ...]

    def layers(self) -> list[LayerLike]: ...      # flat, linked
    def groups(self) -> list[LayerLike]: ...      # top-level grouping
    def forward(self, inputs: Any) -> Any: ...    # populates activation store
```

Visualizer protocol

```
# viz/visualizer_base.py
class VisualizerBase(QWidget):
    @classmethod
    def compatible_with(cls, layer: LayerLike) -> bool: ...
    def set_layer(self, layer: LayerLike) -> None: ...
    def refresh(self) -> None: ...
```

GUI responsibilities (one line each)
File	Owns
sidebar/sidebar.py	Composes selector + tree + viz list; emits visualizer_requested
sidebar/model_selector.py	Collapsable menus of registered models; emits model_selected(adapter)
sidebar/layer_tree.py	Recursive nested collapsable menus built from LayerLike.children()
sidebar/viz_list.py	Filters registry by viz.compatible_with(layer); emits viz_chosen
tabs/tab_area.py	Holds QTabWidget; open_tab(layer, viz_cls); handles close
tabs/tab.py	Wraps one visualizer instance; if none → renders EmptyState
tabs/empty_state.py	Centered label: "Select a model"
main_window.py	QMainWindow with Sidebar left + TabArea right; connects signals
UX flow

When the program opens with two registered models:

    The visualizer area shows an empty tab: "Select a model".
    The sidebar's model collapsable menu lists the two models.
    Clicking a model expands a collapsable menu of that model's layers / layer groups.
    Clicking a layer expands a list of only the visualizers compatible with that layer (via compatible_with).
    Clicking a visualizer opens a new closeable tab containing it.
    Multiple tabs let the user switch between visualizations without losing state. Closing the last tab returns to the empty state.

Signal wiring (lives only in main_window.py)

scss
ModelSelector.model_selected   → LayerTree.set_root(adapter.groups())
LayerTree.layer_selected       → VizList.set_layer(layer)
VizList.viz_chosen             → TabArea.open_tab(layer, viz_cls)
TabArea (close button)         → remove tab; if empty → show EmptyState

Each component only knows about its immediate neighbors via signals, so every
file stays small and independently testable.
Compatibility declarations

Two natural places to declare what fits with what:

python
# adapters/transformer_adapter.py
class TransformerAdapter:
    accepted_inputs = (TextInput,)        # what user inputs make sense

# viz/visualizers/attention_viz/viz.py
class AttentionVisualizer(VisualizerBase):
    @classmethod
    def compatible_with(cls, layer: LayerLike) -> bool:
        return "attn" in layer.name.lower() and _has_attn_weights(layer.outputs())

The MainWindow then:

    shows input widgets allowed by adapter.accepted_inputs
    on layer-select, filters registered visualizers by compatible_with(layer)

Extending the framework
Adding a new model

    Create adapters/<model>_adapter.py.
    Implement the ModelAdapter protocol.
    Register it in core/registry.py.

No edits to core/ or viz/ required.
Adding a new visualizer

    Create viz/visualizers/<name>/viz.py containing a VisualizerBase subclass.
    Optionally add viz/visualizers/<name>/components/ for private widgets.
    Register it in core/registry.py (or rely on auto-discovery via viz/visualizers/__init__.py).

Adding a new input type

    Create core/input/<type>_input.py implementing InputBase.
    Adapters opt in by listing it in accepted_inputs.

Image model support

The core abstractions are architecture-agnostic and already cover image
models. Adding image support means:

    A new CNNAdapter / ViTAdapter that walks conv blocks instead of transformer blocks, with accepted_inputs = (ImageInput,).
    Image-appropriate visualizers (feature maps, conv kernels, Grad-CAM), each gating itself via compatible_with.

No changes to core/ are required.
Deferred — not in scope for v1
1. Layer-group inheritance of visualizers

Currently viz_list only shows visualizers whose compatible_with accepts
the selected layer directly. A LayerGroup does not inherit the
visualizers of its children, because rendering "the same viz across N
children at once" needs a separate composition story (grid? overlay?
side-by-side?). Revisit once we have ≥2 visualizers and real usage.
2. Multi-pane layout inside a single tab

Today: 1 tab = 1 visualizer. Future: a tab could host a splitter with
multiple visualizers on the same or different layers. This likely means
promoting tab.py from "wraps a QWidget" to "wraps a layout tree of
QWidgets" — keep tab.py small now so that swap is cheap.
3. Multi-page workspaces

Allow grouping sets of tabs into pages (like browser windows) so a user can
keep, e.g., an "attention exploration" page separate from a "weights
inspection" page.
4. Auto-discovery of visualizers and adapters

Walk viz/visualizers/*/viz.py and adapters/*.py at startup and register
automatically, so core/registry.py doesn't need manual edits.
Quality bars

    Type hints everywhere; from __future__ import annotations at the top of every module.
    No global mutable state outside core/registry.py.
    The activation store is owned by the adapter and passed in explicitly.
    Every file begins with a short docstring describing its role.
    The app must run end-to-end on the examples/xor_transformer/ demo with no errors.
    The app must not hardcode anything for xor 

