# Guía paso a paso — correr el test OOS en RunPod (para principiantes)

Objetivo: alquilar una GPU en la nube por un rato, correr el script y descargar los
resultados. Coste esperado **~$5–10**. Tiempo total **~1–2 horas** (casi todo esperando
descargas; tu atención activa son ~15 min).

Lo que necesitas a mano: el archivo **`oos_gpu_bundle.zip`** (ya lo tienes en tu carpeta
Descargas) y una tarjeta de crédito.

---

## Paso 1 — Crear cuenta y meter crédito

1. Entra en **https://runpod.io** → **Sign Up** (puedes usar Google).
2. Arriba, ve a **Billing** (facturación).
3. Añade **$20** de crédito con tu tarjeta. (Solo gastarás ~$5–10; el resto se queda en tu saldo.)

---

## Paso 2 — Alquilar la GPU (desplegar un "Pod")

1. En el menú de la izquierda pincha **Pods** → botón **Deploy** (o **+ GPU Pod**).
   > Es "Pods" / "GPU Cloud", **NO** "Serverless".
2. **Elige la GPU:** busca **H100 80GB** (o **A100 80GB PCIe** si no hay H100). Cantidad: **1**.
   > El "80GB" es obligatorio: el modelo no cabe en menos.
3. **Elige la plantilla (Template):** una oficial de **PyTorch** (p. ej. *"RunPod PyTorch 2.x"*).
   Ya trae Python, CUDA y Jupyter listos.
4. **Disco:** busca *"Volume Disk"* / *"Container Disk"* y ponlo en **100 GB** como mínimo
   (el modelo base ocupa ~65 GB al descargarse).
5. Pincha **Deploy** y espera 1–2 min hasta que el pod ponga **Running** (en verde).

---

## Paso 3 — Abrir el entorno (Jupyter)

1. En tu pod, pincha **Connect**.
2. Elige **Connect to Jupyter Lab** (se abre una pestaña nueva en el navegador).
   > Si no aparece Jupyter, usa **Start Web Terminal** — los comandos son los mismos.

---

## Paso 4 — Subir el bundle

1. En Jupyter, a la izquierda hay un panel de archivos. **Arrastra y suelta** ahí tu
   **`oos_gpu_bundle.zip`**. Espera a que suba (la barra de progreso; ~256 MB, unos minutos).
2. Abre una terminal dentro de Jupyter: menú **File → New → Terminal**.
3. Descomprime y entra en la carpeta (escribe esto y pulsa Enter en cada línea):
   ```bash
   unzip oos_gpu_bundle.zip
   cd oos_gpu_bundle
   ```

---

## Paso 5 — Instalar y lanzar

En esa misma terminal:

```bash
pip install vllm transformers safetensors
python run_oos_gpu.py --cases oos_cases.json --adapters ./adapters -o results_oos/
```

Qué verás:
- La **primera vez descarga el modelo base** (~65 GB) → 10–20 min. Es normal que parezca
  parado; está bajando.
- Luego corre la inferencia (10–30 min) y al final imprime una **tabla** así:
  ```
  arm                      PAI  repetition   entropy  % monotone
  base                   0.xxx       0.xxx     0.xxx       xx.x%
  greedy_r8              0.xxx       0.xxx     0.xxx       xx.x%
  ```
  Eso significa que terminó bien.

---

## Paso 6 — Descargar los resultados

En la terminal, comprime la carpeta de resultados:
```bash
zip -r results_oos.zip results_oos
```
Luego, en el panel de archivos de Jupyter, **clic derecho sobre `results_oos.zip` →
Download**. Guárdalo en tu Mac. **Eso es lo que hay que enviar / analizar.**

---

## Paso 7 — APAGAR el pod (¡importante, para no seguir pagando!)

1. Vuelve a la pestaña de RunPod → **Pods**.
2. En tu pod pincha **Terminate** (no solo "Stop").
   > **Terminate** = lo borra del todo y dejas de pagar. "Stop" puede seguir cobrándote el disco.

---

## Si algo falla

- **Error al cargar el adapter** (menciona `all-linear`): abre
  `adapters/greedy_r8/adapter_config.json` en Jupyter y cambia
  `"target_modules": "all-linear"` por
  `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`.
  Vuelve a lanzar el Paso 5. (Está también explicado en `README_RUN.md` dentro del bundle.)
- **Muchos `parse_error`** en la salida: avísame, ajustamos el parser.
- **Se queda sin memoria (OOM):** confirma que la GPU es de 80 GB y que no lanzaste el
  script dos veces a la vez.

---

## Resumen de costes

| Concepto | Coste |
|---|---|
| H100 80GB en RunPod | ~$2–3 / hora |
| Tiempo total (descarga + run) | ~1–2 horas |
| **Total estimado** | **~$5–10** |

Recuerda: solo pagas mientras el pod está **Running**. En cuanto haces **Terminate**, deja de contar.
