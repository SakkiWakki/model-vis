"""Sidebar: composes Model+Dataset selectors + LayerTree + VizList; builds adapters."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from model_viz.core.adapter import ModelAdapter
from model_viz.core import registry
from model_viz.core.layer import LayerLike
from model_viz.core.gui.sidebar.dataset_selector import DatasetSelector
from model_viz.core.gui.sidebar.hf.loader_dialog import HFLoaderDialog
from model_viz.core.gui.sidebar.hf.ollama_picker import OllamaPickerDialog
from model_viz.core.gui.sidebar.model_selector import ModelSelector
from model_viz.core.gui.sidebar.layer_tree import LayerTree
from model_viz.core.gui.sidebar.viz_list import VizList


class Sidebar(QWidget):
    visualizer_requested = pyqtSignal(object, object)  # (layer, viz_cls)
    model_selected = pyqtSignal(object)  # emits ModelAdapter

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(220)
        self.setMaximumWidth(340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.model_selector = ModelSelector()
        layout.addWidget(self.model_selector)

        self.dataset_selector = DatasetSelector()
        layout.addWidget(self.dataset_selector)

        self._load_hf_btn = QPushButton("Load HF model…")
        self._load_hf_btn.clicked.connect(self._on_load_hf_clicked)
        layout.addWidget(self._load_hf_btn)

        self._load_ollama_btn = QPushButton("Load Ollama model…")
        self._load_ollama_btn.clicked.connect(self._on_load_ollama_clicked)
        layout.addWidget(self._load_ollama_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        self.layer_tree = LayerTree()
        layout.addWidget(self.layer_tree)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)

        self.viz_list = VizList()
        layout.addWidget(self.viz_list)
        layout.addStretch(1)

        # Wire internal signals.
        self.model_selector.model_selected.connect(self._on_model_selected)
        self.dataset_selector.dataset_selected.connect(self._on_dataset_selected)
        self.layer_tree.layer_selected.connect(self.viz_list.set_layer)
        self.viz_list.viz_chosen.connect(self.visualizer_requested)

        self._selected_model_name: Optional[str] = None
        # Datasets are duck-typed via the capability protocols in
        # ``data.base``; the sidebar itself doesn't care which capabilities
        # are present, the chosen factory does the narrowing.
        self._selected_dataset: Optional[object] = None

    def _on_model_selected(self, model_name: str) -> None:
        self._selected_model_name = model_name
        self._maybe_build_adapter()

    def _on_dataset_selected(self, dataset: object) -> None:
        self._selected_dataset = dataset
        self._maybe_build_adapter()

    def _maybe_build_adapter(self) -> None:
        if self._selected_model_name is None or self._selected_dataset is None:
            return
        factories = registry.get_model_factories()
        factory = factories.get(self._selected_model_name)
        if factory is None:
            return
        adapter = factory(self._selected_dataset)
        self._present_adapter(adapter)

    def _present_adapter(self, adapter: ModelAdapter) -> None:
        self.layer_tree.set_root(adapter.groups())
        self.viz_list.set_adapter(adapter)
        self.model_selected.emit(adapter)

    def _on_load_hf_clicked(self) -> None:
        self._present_bundle_dialog(HFLoaderDialog(self))

    def _on_load_ollama_clicked(self) -> None:
        self._present_bundle_dialog(OllamaPickerDialog(self))

    def _present_bundle_dialog(self, dlg) -> None:
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        bundle = dlg.bundle
        if bundle is None:
            return
        adapter = self._adapter_from_bundle(bundle)
        if adapter is not None:
            self._present_adapter(adapter)

    def _adapter_from_bundle(self, bundle) -> Optional[ModelAdapter]:
        # Dispatch on which loader produced the bundle.  HF bundles carry a
        # ``model`` (nn.Module); llama.cpp bundles carry a ``llama`` runtime.
        if hasattr(bundle, "llama"):
            from model_viz.adapters.llamacpp import LlamaCppAdapter
            return LlamaCppAdapter(name=bundle.name, llama=bundle.llama)
        if hasattr(bundle, "model"):
            from model_viz.adapters.hf import HFCausalLMAdapter
            return HFCausalLMAdapter(
                name=bundle.name, model=bundle.model, tokenizer=bundle.tokenizer,
            )
        return None
