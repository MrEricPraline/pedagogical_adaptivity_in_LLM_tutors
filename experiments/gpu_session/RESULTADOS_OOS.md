# Resultados — Test de generalización OOS (base vs greedy r=8)

**Qué se hizo:** correr el modelo base (Qwen3-32B) y el LoRA **greedy r=8** sobre
**200 casos out-of-sample** (elegidos del corpus, 0 solapamiento con los 589 de
entrenamiento) y medir PAI + diversidad de la secuencia. greedy_r8 se sirvió en
vLLM+LoRA en una H100; el base se puntuó reparseando sus propias salidas.

## Resultado

| arm | PAI | repetición | entropía | % monótono | n |
|---|---|---|---|---|---|
| base (Qwen3-32B) | 0.301 | 0.50 | 1.53 | 0% | 199/200 |
| **greedy_r8** | **0.433** | **1.00** | **0.00** | **100%** | 200/200 |

Contraste con el entrenamiento (589 casos), ya conocido:

| | base PAI | base %mono | greedy PAI | greedy %mono |
|---|---|---|---|---|
| train (589) | 0.185 | 0% | 0.413 | 100% |
| **OOS (200)** | **0.301** | **0%** | **0.433** | **100%** |

## Hallazgo

**greedy_r8 sube el PAI (+0.13 sobre base) pero colapsa la diversidad de la
secuencia a cero** (entropía 1.53→0, monótono 0%→100%): diseña **una sola actividad
y la repite 5 veces**. Sigue adaptando la elección al caso (varía entre alumnos),
pero no genera secuencia.

Como esto ocurre **fuera del conjunto de entrenamiento** y **reproduce clavado el
patrón del train**, la conclusión es:

> **El colapso generaliza.** La ganancia de métrica del adapter va inseparablemente
> unida al colapso de diversidad, también en casos nunca vistos.

## Notas de método

- Prompt y scoring PAI **idénticos** al pipeline original (`pai_lib.py`, verificado
  contra `src/stage5_scoring`).
- El adapter greedy_r8 requirió renombrar la capa de salida `unembed_tokens` →
  `lm_head` (nomenclatura Tinker vs HuggingFace); pesos intactos. Que el resultado
  cuadre con el train confirma que el renombrado fue correcto.
- base: 1/200 no reparseó (formato JSON degenerado del modelo); despreciable.

## Pendiente (no hecho)

- **Test causal RQ3.2** (brazos `greedy_parallel` / `greedy_orthogonal`): esos
  adapters **no existen** todavía ni hay receta de construcción. Es la otra mitad
  del plan del advisor y queda para cuando llegue la receta del colaborador.

## Archivos

- `results_oos_FINAL.zip` → `oos_summary_final.json` (tabla), `oos_greedy_r8.json`
  (200 salidas + scores), `oos_base_reparsed.json` (199 base + scores),
  `oos_base.json` (crudo original), `run2.log`.
