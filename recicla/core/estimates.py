"""
Estimativa de peso e valor.

IMPORTANTE: uma câmera RGB não mede peso. Aqui só se multiplica a CONTAGEM pelo
PESO MÉDIO configurado por categoria e pelo PREÇO/kg configurado. É uma estimativa
estatística: funciona razoavelmente para muitos itens, mas um item isolado pode
pesar muito mais ou menos que a média.
"""
from __future__ import annotations

from ..config import Material


def item_estimate(m: Material) -> tuple[float, float]:
    """(peso em kg, valor em moeda) de UM item médio da categoria."""
    kg = m.avg_weight_g / 1000.0
    return kg, kg * m.price_per_kg


def group_estimate(m: Material, count: int) -> tuple[float, float]:
    kg, value = item_estimate(m)
    return kg * count, value * count


def fmt_money(value: float, currency: str = "R$") -> str:
    s = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{currency} {s}"


def fmt_weight(kg: float) -> str:
    if kg < 1:
        return f"{kg * 1000:.0f} g".replace(".", ",")
    return f"{kg:.2f} kg".replace(".", ",")
