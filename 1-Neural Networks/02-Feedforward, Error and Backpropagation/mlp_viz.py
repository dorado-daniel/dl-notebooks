"""
mlp_viz.py — Mini visualizador de MLP para PyTorch (capas nn.Linear)

Uso rápido:
    from mlp_viz import draw, draw_dense, draw_layer
    # 1) Capa sola
    draw_dense(layer1)             # o más corto: draw(layer1)

    # 2) Capa + activación (sin declarar Sequential)
    draw([layer1, "ReLU"])         # construye Sequential al vuelo
    draw_layer(layer1, act=True)   # atajo ReLU
    draw_layer(layer1, act="ReLU") # explícito

    # 3) Modelo con varias capas (detecta activaciones reales)
    model = nn.Sequential(layer1, nn.ReLU(), layer2, nn.Sigmoid())
    draw(model)                    # capas + pesos + bias + activaciones

    # 4) Valores de nodos para una muestra
    draw_dense(model, sample=X[0], show_node_values=True)

Características:
- Lee pesos reales (azul = +, rojo = −; grosor ∝ |peso|).
- Muestra activación que sigue a cada Linear (ReLU, Sigmoid, Tanh, ...).
- Activación por neurona opcional con 'act_overrides'.
- Valores de nodos (si pasas 'sample' y 'show_node_values=True').
- Soporta nn.Sequential, nn.Module con children simples o una nn.Linear suelta.
- CPU/GPU: maneja detach()/cpu() internamente para graficar.
"""

import math
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


__all__ = ["draw", "draw_dense", "draw_layer"]


# ----------------------------- API pública ------------------------------------
def draw_dense(
    model_or_layer: Union[nn.Module, nn.Linear],
    sample: Optional[torch.Tensor] = None,
    feature_names: Optional[List[str]] = None,
    title: str = "MLP visualizer",
    show_weights: bool = True,
    show_bias: bool = True,
    show_activation_names: bool = True,
    show_node_values: bool = False,
    scale: float = 2.0,
    act_overrides: Optional[List[Union[str, List[str]]]] = None,
) -> None:
    """
    Visualiza redes densas (nn.Linear + activaciones) con pesos reales.

    Args:
        model_or_layer: nn.Sequential / nn.Module con capas nn.Linear, o una sola nn.Linear.
        sample: (opcional) tensor (in_features,) o (1, in_features) para mostrar valores de nodos.
        feature_names: etiquetas de entrada (por defecto ['x1','x2',...]).
        title: título del gráfico.
        show_weights: muestra etiquetas numéricas en las conexiones.
        show_bias: muestra bias junto a cada neurona.
        show_activation_names: anota la activación tras cada Linear.
        show_node_values: si True y hay sample, escribe el valor activado de cada nodo.
        scale: factor del grosor de líneas ∝ |peso|.
        act_overrides: lista por capa (str) o por neurona (lista de str) para forzar anotación de activación.
    """
    if not isinstance(model_or_layer, nn.Module):
        raise ValueError("model_or_layer debe ser nn.Module o nn.Linear.")

    layers, activations = _extract_dense_layers_and_acts(model_or_layer)
    if not layers:
        raise ValueError("No se encontraron capas nn.Linear en el módulo dado.")

    # Activación por neurona a partir de 'act_overrides' o la activación detectada por capa.
    per_neuron_acts: List[Optional[List[Optional[str]]]] = []
    if act_overrides is not None:
        for li, layer in enumerate(layers):
            if li < len(act_overrides):
                item = act_overrides[li]
                if isinstance(item, str) or item is None:
                    per_neuron_acts.append([item] * layer.out_features)
                elif isinstance(item, list):
                    row = (item + [item[-1]] * layer.out_features)[:layer.out_features] if item else [None]*layer.out_features
                    per_neuron_acts.append(row)
                else:
                    per_neuron_acts.append([None] * layer.out_features)
            else:
                per_neuron_acts.append([activations[li]] * layer.out_features)
    else:
        for li, layer in enumerate(layers):
            per_neuron_acts.append([activations[li]] * layer.out_features)

    # Nombres de features
    in_features = layers[0].in_features
    if feature_names is None:
        feature_names = [f"x{j+1}" for j in range(in_features)]

    # Valores de nodos (z/a) si se solicita
    node_values = None
    if sample is not None and show_node_values:
        node_values = _compute_node_values(model_or_layer, sample, layers, activations)

    _plot_network(
        layers=layers,
        per_neuron_acts=per_neuron_acts,
        feature_names=feature_names,
        node_values=node_values,
        title=title,
        show_weights=show_weights,
        show_bias=show_bias,
        show_activation_names=show_activation_names,
        scale=scale,
    )


def draw_layer(
    layer: nn.Linear,
    act: Optional[Union[str, List[str], bool]] = None,
    **kwargs,
) -> None:
    """
    Visualiza una sola capa Linear con (opcional) una activación.

    Ejemplos:
        draw_layer(layer1)                 → sin activación
        draw_layer(layer1, act=True)       → ReLU por defecto (atajo)
        draw_layer(layer1, act="ReLU")     → capa + ReLU
        draw_layer(layer1, act=["ReLU","Tanh"]) → etiquetar por neurona (solo visual)
    """
    if not isinstance(layer, nn.Linear):
        raise TypeError("draw_layer espera una nn.Linear")
    if act is True:  # alias cómodo
        act = "ReLU"

    act_module = _make_activation_module(act) if isinstance(act, str) else nn.Identity()
    model = nn.Sequential(layer, act_module)

    act_overrides = None
    if isinstance(act, list):
        act_overrides = [act]  # una sola capa: lista por neurona

    draw_dense(
        model_or_layer=model,
        act_overrides=act_overrides,
        **kwargs,
    )


def draw(obj: Union[nn.Module, nn.Linear, List[Union[str, nn.Module]], Tuple[Union[str, nn.Module], ...]], **kwargs):
    """
    Atajo ultra simple:
      - draw(model)                 → igual que draw_dense(model)
      - draw(layer)                 → igual que draw_dense(layer)
      - draw([layer, "ReLU"])       → construye Sequential(layer, ReLU) en vuelo y lo dibuja
      - draw((layer1, "ReLU", layer2, "Sigmoid")) → idem con varias capas
    """
    if isinstance(obj, (list, tuple)):
        parts = []
        for p in obj:
            if isinstance(p, str):
                parts.append(_make_activation_module(p))
            elif isinstance(p, nn.Module):
                parts.append(p)
            else:
                raise TypeError(f"Elemento no soportado en lista/tupla: {type(p)}")
        model = nn.Sequential(*parts)
        return draw_dense(model, **kwargs)
    elif isinstance(obj, nn.Module):
        return draw_dense(obj, **kwargs)
    else:
        raise TypeError("Argumento no soportado para draw(). Pasa un módulo, capa o lista/tupla de partes.")


# ------------------------------ Internos --------------------------------------
def _extract_dense_layers_and_acts(module: nn.Module) -> Tuple[List[nn.Linear], List[Optional[str]]]:
    """
    Devuelve dos listas paralelas:
      - layers: nn.Linear en orden de forward
      - activations: nombre de la activación inmediata que sigue a cada Linear (o None)
    """
    modules = list(_iter_modules_in_order(module))
    linear_layers: List[nn.Linear] = []
    activations: List[Optional[str]] = []

    for i, m in enumerate(modules):
        if isinstance(m, nn.Linear):
            linear_layers.append(m)
            act_name = None
            if i + 1 < len(modules):
                act_name = _activation_name(modules[i + 1])
            activations.append(act_name)

    return linear_layers, activations


def _iter_modules_in_order(module: nn.Module) -> Iterable[nn.Module]:
    """
    Recorre módulos en un orden razonable para modelos tipo Sequential o módulos simples.
    """
    if isinstance(module, nn.Sequential):
        for m in module:
            if isinstance(m, nn.Sequential):
                for sm in _iter_modules_in_order(m):
                    yield sm
            else:
                yield m
        return

    if isinstance(module, nn.Linear):
        yield module
        return

    for m in module.children():
        yield m


def _activation_name(m: nn.Module) -> Optional[str]:
    mapping = {
        nn.ReLU: "ReLU",
        nn.Sigmoid: "Sigmoid",
        nn.Tanh: "Tanh",
        nn.LeakyReLU: "LeakyReLU",
        nn.Softmax: "Softmax",
        nn.Identity: "Identity",
        nn.SELU: "SELU",
        nn.GELU: "GELU",
        nn.SiLU: "SiLU",
        nn.Hardsigmoid: "HardSigmoid",
        nn.Hardtanh: "Hardtanh",
        nn.ELU: "ELU",
    }
    for k, v in mapping.items():
        if isinstance(m, k):
            return v
    return None


def _compute_node_values(
    model: nn.Module,
    sample: torch.Tensor,
    layers: List[nn.Linear],
    activations: List[Optional[str]],
):
    """
    Ejecuta forward y captura valores por capa (z tras Linear, a tras activación inmediata).
    Devuelve: [{'z': tensor(1,out), 'a': tensor(1,out)}, ...]
    """
    model_device = next(layers[0].parameters()).device
    x = sample.detach()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(model_device)

    values = []
    modules = list(_iter_modules_in_order(model))
    a = x

    for i, m in enumerate(modules):
        if isinstance(m, nn.Linear):
            z = m(a)
            a_next = z
            if i + 1 < len(modules):
                act_name = _activation_name(modules[i + 1])
                if act_name is not None:
                    a_next = _apply_activation(act_name, z)
            values.append({"z": z.detach().cpu(), "a": a_next.detach().cpu()})
            a = a_next
        else:
            act_name = _activation_name(m)
            if act_name is None:
                a = m(a)

    return values


def _apply_activation(name: str, x: torch.Tensor) -> torch.Tensor:
    name = name.lower()
    if "relu" in name and "leaky" not in name:
        return torch.relu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "tanh":
        return torch.tanh(x)
    if "leakyrelu" in name:
        return torch.nn.functional.leaky_relu(x)
    if "softmax" in name:
        return torch.nn.functional.softmax(x, dim=-1)
    if "identity" in name:
        return x
    return x


def _plot_network(
    layers: List[nn.Linear],
    per_neuron_acts: List[List[Optional[str]]],
    feature_names: List[str],
    node_values,
    title: str,
    show_weights: bool,
    show_bias: bool,
    show_activation_names: bool,
    scale: float,
):
    L = len(layers)
    xs = np.linspace(0.0, float(L), L + 1)  # x=0 inputs, x=1..L salidas de cada Linear

    def layer_ys(n):
        if n == 1:
            return [0.0]
        step = 1.0 / (n - 1)
        return [0.5 - i * step for i in range(n)]

    # Entradas
    in_features = layers[0].in_features
    y_prev = layer_ys(in_features)
    x_prev = xs[0]

    plt.figure(figsize=(max(7, 3 * L), 4))
    ax = plt.gca()
    ax.axis("off")
    plt.title(title)

    # Nodos de entrada
    ax.scatter([x_prev] * in_features, y_prev, s=600, color="gold", edgecolor="black", zorder=3)
    for i, y in enumerate(y_prev):
        ax.text(x_prev - 0.08, y, feature_names[i], fontsize=11, va="center", ha="right")

    # Por cada capa Linear
    for li, layer in enumerate(layers):
        W = layer.weight.detach().cpu().numpy()  # (out, in)
        b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

        out_features = layer.out_features
        x_curr = xs[li + 1]
        y_curr = layer_ys(out_features)

        # Conexiones prev->curr
        for i in range(len(y_prev)):
            for j in range(len(y_curr)):
                w = W[j, i]
                lw = max(0.5, abs(w) * scale)
                color = "royalblue" if w >= 0 else "crimson"
                ax.plot([x_prev, x_curr], [y_prev[i], y_curr[j]], color=color, linewidth=lw, alpha=0.75)

                if show_weights:
                    # Posición a lo largo + offset perpendicular alternado (evita solapamientos)
                    t = 0.35 if ((i + j) % 2 == 0) else 0.65
                    perp = 0.03 if ((i - j) % 2 == 0) else -0.03
                    _label_along(ax, x_prev, y_prev[i], x_curr, y_curr[j], f"{w:.2f}", t, perp)

        # === Nodos actuales (con sombreado activo/apagado si hay sample) ===
        if node_values is not None and li < len(node_values):
            activations_np = node_values[li]["a"].numpy()[0]
            colors = ["lightgray" if a == 0 else "teal" for a in activations_np]
        else:
            colors = ["teal"] * out_features

        ax.scatter(
            [x_curr] * out_features,
            y_curr,
            s=600,
            color=colors,
            edgecolor="black",
            zorder=3
        )

        # === Etiquetas, activaciones y valores ===
        for j, y in enumerate(y_curr):
            # Etiqueta del nodo: h{j+1} o ŷ{j+1} si es la última capa
            is_output_layer = (li == L - 1)
            node_name = (f"ŷ{j+1}" if is_output_layer else f"h{j+1}")
            bias_str = f"  b={b[j]:.2f}" if (show_bias and b is not None) else ""
            ax.text(x_curr + 0.08, y, f"{node_name}{bias_str}", fontsize=11, va="center", ha="left")

            if show_activation_names:
                act_name = per_neuron_acts[li][j]
                if act_name is not None:
                    ax.text(x_curr, y - 0.1, f"{act_name}", fontsize=9, va="top", ha="center", color="dimgray")

            if node_values is not None and li < len(node_values):
                z = float(node_values[li]["z"][0, j])
                a = float(node_values[li]["a"][0, j])
                # Valor activado dentro del nodo
                ax.text(x_curr, y, f"{a:.2f}", fontsize=9, color="white", ha="center", va="center")
                # Valor de preactivación justo debajo en gris
                ax.text(x_curr, y - 0.2, f"z={z:.2f}", fontsize=8, color="dimgray", ha="center", va="top")

        # siguiente capa
        y_prev, x_prev = y_curr, x_curr

    plt.xlim(xs[0] - 0.3, xs[-1] + 0.3)
    plt.ylim(-0.6, 0.6)
    plt.show()


def _label_along(ax, x0, y0, x1, y1, text, t=0.5, perp=0.03):
    xm = x0 + t * (x1 - x0)
    ym = y0 + t * (y1 - y0)
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) if (dx or dy) else 1.0
    nx, ny = -dy / L, dx / L  # normal unitaria
    ax.text(
        xm + nx * perp,
        ym + ny * perp,
        text,
        fontsize=8,
        ha="center",
        va="center",
        color="black",
        bbox=dict(facecolor="white", alpha=0.65, lw=0),
    )


def _make_activation_module(name: Optional[Union[str, bool]]) -> nn.Module:
    if name is None or name is False:
        return nn.Identity()
    if name is True:
        return nn.ReLU()
    name_low = str(name).lower()
    if name_low == "relu":
        return nn.ReLU()
    if name_low == "sigmoid":
        return nn.Sigmoid()
    if name_low == "tanh":
        return nn.Tanh()
    if name_low == "leakyrelu":
        return nn.LeakyReLU()
    if name_low == "gelu":
        return nn.GELU()
    if name_low == "silu":
        return nn.SiLU()
    if name_low == "selu":
        return nn.SELU()
    if name_low == "elu":
        return nn.ELU()
    if name_low == "softmax":
        return nn.Softmax(dim=-1)
    return nn.Identity()
