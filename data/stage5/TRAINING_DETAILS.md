# Detalles técnicos del fine-tuning (Stage 5)

**Setup:** LoRA sobre `Qwen/Qwen3-32B`, entrenado en Tinker con el *corrective
training set* (n = 589 ejemplos). Sweep de rangos r ∈ {1, 4, 8, 16}; la variante
per-DP usa r = 8.

Todos los valores de este documento están verificados contra **fuente de
verdad** (código de entrenamiento + los `adapter_config.json` reales descargados
de los adapters entrenados), no solo contra los defaults del SDK. La procedencia
de cada dato se indica en la columna *Fuente*.

---

## 1. Hiperparámetros principales

| Parámetro | Valor | Fuente |
|---|---|---|
| **Optimizer** | **Adam** | `tinker_train.py` → `types.AdamParams(...)` |
| ├ β₁ | 0.9 | Default Tinker `AdamParams` (no sobreescrito) |
| ├ β₂ | 0.95 | Default Tinker `AdamParams` (no sobreescrito) |
| ├ eps | 1e-12 | Default Tinker `AdamParams` (no sobreescrito) |
| ├ weight_decay | 0.0 | Default Tinker `AdamParams` (no sobreescrito) |
| └ grad_clip_norm | 0.0 (sin clipping) | Default Tinker `AdamParams` (no sobreescrito) |
| **Random seed** | **No fijada** (init aleatorio) | `create_lora_training_client()` sin `seed` → `seed=None`; sin `manual_seed` |
| **Batch size** | **1 secuencia** (update por ejemplo) | Loop: `forward_backward([datum])` + `optim_step` por ejemplo |
| **LoRA α** | **32** (fijo, igual en todos los rangos) | `adapter_config.json` reales (r1/r4/r8/r16 → todos α=32) |
| **Dropout (LoRA)** | **0.0** | `lora_dropout: 0` en todos los `adapter_config.json` |
| **Warmup / LR schedule** | **Ninguno** — LR constante | Mismo `lr=1e-4` en cada `optim_step`; sin warmup ni decay |

## 2. Contexto adicional

| Parámetro | Valor | Fuente |
|---|---|---|
| Modelo base | `Qwen/Qwen3-32B` | `manifest_finetune.json` |
| Learning rate | 1e-4 (constante) | `manifest_finetune.json` / `tinker_train.py` |
| Épocas | 3 | `manifest_finetune.json` |
| Rangos LoRA (sweep) | {1, 4, 8, 16}; per-DP = 8 | `manifest_finetune.json` / `manifest_per_dp_finetune.json` |
| Nº de ejemplos | 589 (corrective set) | `manifest_finetune.json` |
| Pasos totales (r8) | 1767 (= 589 × 3 épocas) | `adapters/adapter_r8.json` → `total_steps` |
| Target modules | `all-linear` (atención + MLP + unembedding) | `adapter_config.json`; `train_mlp/attn/unembed` = True |
| rsLoRA | `false` → escala efectiva = α/r (no α/√r) | `adapter_config.json` → `use_rslora: false` |
| Loss / train-on | cross-entropy sobre `LAST_ASSISTANT_MESSAGE`, reducción `mean` | `tinker_train.py` |
| Loss final (mean) | r1 0.0028 · r4 0.0031 · r8 0.0029 · r16 0.0030 | `manifest_finetune.json` |

**Nota sobre α fijo:** al no escalar α con el rango, el factor de escala efectivo
α/r difiere por rango: **r1 → 32, r4 → 8, r8 → 4, r16 → 2**. Es el comportamiento
estándar de Tinker (convención de "LoRA Without Regret"), no una anomalía.

---

## 3. Qué reportar en el paper: decisiones de diseño vs. limitaciones

Ninguno de los valores anteriores revela un **error de ejecución**. Clasificación
honesta de cara a revisores:

**Van en *Método* (decisiones descritas, no disculpas):**

- **Batch size = 1.** Elección de entrenamiento legítima (updates online, un
  `optim_step` por ejemplo); es el patrón base de Tinker. Convergió a loss
  ~0.003, así que funcionó. Solo hay que describirlo con precisión.
- **α = 32 fijo.** Default estándar de Tinker, no un descuido. *Matiz de
  interpretación:* como α no escala con r, el sweep de rango covaría rango y
  escala efectiva (α/r). La claim debe acotarse a "sweep de rango tal como se
  configura en Tinker (α fijo)", no "efecto aislado del rango/capacidad". Es una
  frase de encuadre, no un re-experimento.

**Va en *Limitaciones* (limitación real — la única de la lista):**

- **Semilla no fijada → una sola corrida por condición.** Dos consecuencias:
  (a) no hay reproducibilidad bit-a-bit; (b) sin réplicas, no hay estimación de
  varianza run-to-run, así que diferencias pequeñas entre rangos no se distinguen
  formalmente del ruido de inicialización. *Mitigación existente:* el resultado
  principal generaliza a 200 casos OOS, lo que reduce la probabilidad de que sea
  un artefacto de una corrida afortunada. No invalida los resultados, pero es un
  caveat honesto de declarar.

**Punto aparte (validez, no en la lista de hiperparámetros):**

- **Loss final ~0.003 en todos los rangos** sugiere que el modelo prácticamente
  *memorizó* los targets correctivos (n = 589). Conecta con el "collapse"
  observado. Es más relevante para la validez del experimento que cualquiera de
  los seis hiperparámetros y merece análisis (train loss vs. comportamiento OOS).

---

*Documento generado a partir de: `data/stage5/manifest_finetune.json`,
`data/stage5/manifest_per_dp_finetune.json`,
`data/stage5/adapters/adapter_r*.json`, `adapters_dl/*/adapter_config.json`,
`src/stage5_finetune/tinker_train.py`, y el SDK de Tinker (`tinker/types/`).*
