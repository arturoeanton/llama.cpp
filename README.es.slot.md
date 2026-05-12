# slotted-real — correr LLMs más grandes que la RAM en hardware común

Este documento describe el experimento slotted-real construido sobre
`llama.cpp` en este fork. El objetivo es validar, sobre un modelo GGUF
real, si un transformer puede ejecutarse en **slots** de layers contiguos,
manteniendo solo unos pocos slots residentes en memoria al mismo tiempo y
streameando el resto desde disco bajo demanda.

Si solo querés *correr* el demo, leé [`README.es.poc.md`](README.es.poc.md).
Este documento cubre diseño, fases y limitaciones conocidas.

## Resumen ejecutivo

- **Objetivo:** correr un modelo de 35 GB con ~11 GB de peak memory footprint.
- **Approach:** cargar solo `slots_resident` slots de layers; armar un
  cgraph separado por slot; pasar hidden state entre slots vía un buffer
  CPU staging; hot-swap de bytes del GGUF on-demand.
- **Validado con Llama 3.1 70B Q3_K_XL** con `slots_resident=2`,
  `--slot-layers 10` (8 slots uniformes), generando output coherente
  multi-token.
- **No validado con Gemma 4 31B** — la estructura heterogénea de layers
  (MoE intercalado con dense, n_heads variable) rompe el hot-swap
  pool-based simple. Limitación conocida.
- **Costo de performance:** ~20× más lento que un baseline totalmente
  residente. La técnica cambia wall-clock por peak memory.

## Motivación

El hardware de consumo está convergiendo a 16-32 GB de RAM unificada o
del sistema. Los pesos abiertos van saliendo en quants cada vez más grandes:
- Llama 3.1 70B Q3_K_XL ≈ 35 GB
- Llama 3.1 70B Q4_K_M ≈ 42 GB
- Llama 3.3 70B Q4_K_M ≈ 42 GB
- Gemma 4 31B Q4_K_M ≈ 18 GB
- Variantes futuras de Mixtral / OLMoE / DeepSeek MoE bien por arriba de 50 GB

Las opciones hoy son:
1. Comprar una workstation. No siempre es posible.
2. Usar un modelo más chico. Regresión de calidad.
3. Dejar que el OS haga swap. Latencias patológicas, OOM kills ocasionales.

La inferencia slotted es una cuarta opción: mantener solo `K` slots de
layers físicamente residentes, streamear el resto desde el GGUF on-demand.
Es intrínsecamente lento pero permite a un modelo que no entra en RAM
producir output correcto.

## Diseño

### Slot plan

Un *slot* es un rango contiguo de layers transformer `[il_start, il_end]`.
El transformer completo se particiona en N slots. El plan de slots se arma
una sola vez desde la metadata GGUF (sin cargar pesos), usando uno de:

- `--slot-layers N` — número fijo de layers por slot. `N` debería dividir
  exacto el total de layers para que todos los slots sean uniformes.
- `--slot-size-mb N` — bytes de peso aproximados por slot (greedy).
  Actualmente produce slots de tamaño variable, que rompe el hot-swap
  homogéneo. Usar `--slot-layers` para hot-swap.

### Loader filter (FASE 4A-1)

`include/llama.h` expone un filtro de carga per-layer:

```c
typedef bool (*llama_layer_filter_t)(int32_t il, void * user_data);

struct llama_model_params {
    ...
    llama_layer_filter_t layer_filter;
    void *               layer_filter_user_data;
};
```

Cuando el filtro devuelve `false` para layer `il`, el loader saltea
completamente los tensors de ese layer:
- `create_tensor` devuelve `nullptr` (contado como skipped, sin alloc de buffer)
- `load_all_data` nunca los itera
- `model.layers[il].wq` etc. quedan `nullptr`

Eso baja el peak footprint aproximadamente proporcional a la fracción de
layers cargados. Medido en Gemma 4 31B Q4_K_M:

| slots_resident | resident_weight_MB | peak_memory_footprint_MB |
|---:|---:|---:|
| 1 | 2,872 | 5,203 |
| 2 | 4,796 | 7,129 |
| 5 | 10,514 | 12,849 |
| 9 (baseline) | 17,806 | 20,133 |

El filtro debe saltear paths `TENSOR_DUPLICATED` (ej. `rope_freqs` de Gemma
compartido entre layers no-SWA) para mantener `n_created` consistente con
el conteo de tensors del GGUF.

### cgraph por slot (FASE 4A-2a)

El builder de cada arquitectura recibe parámetros para emitir nodos solo
para un subrango de layers, y expone el hidden state en el borde del slot
como output/input.

`llm_graph_params` (en `src/llama-graph.h`) gana tres campos nuevos, todos
con defaults que preservan el comportamiento original:

```cpp
int           slot_il_start    = 0;
int           slot_il_end      = -1;    // -1 -> full range
ggml_tensor * slot_inpL_carry  = nullptr;
```

En `src/models/llama.cpp` (y `gemma4.cpp`) el constructor los usa:

- `slot_il_start == 0` y sin carry → arma inpL desde token embeddings (path canónico).
- `slot_il_start > 0` → crea un tensor de input nuevo `slot_inp_carry` en
  el `ctx_compute` del cgraph y lo usa como `inpL`. El driver escribe data
  ahí vía `ggml_backend_tensor_set` entre cómputos de slot.
- Body del loop sin cambios: `for (int il = slot_il_start; il <= slot_il_end; ++il)`.
- `slot_il_end == n_layer - 1` → corre `output_norm + lm_head` y setea
  `res->t_logits` (slot final).
- Si no → `ggml_set_output(cur)` y `res->t_embd = cur` para que el driver
  pueda leer el hidden state per-slot.

Validación: con `slots_resident == total_slots` (sin eviction), el path
slotted produce **el mismo top-1 token y logit, bit-for-bit, a través de
16 tokens de generación greedy**, comparado contra el path `llama_decode`
sin modificar en Gemma 4 31B Q4_K_M. El builder slotted es por tanto
matemáticamente equivalente al builder monolítico.

### Runtime de hot-swap (FASE 4A-2b)

`src/llama-slotted-runtime.{h,cpp}` implementa el runtime que streamea
slots no residentes desde el GGUF a los buffers de slot residente
on-demand.

Estado clave en `slotted_hot_swap_state`:

- Un `fd` read-only separado al archivo GGUF.
- Un mapa `gguf_tensors` (armado desde la metadata de `gguf_init_from_file`)
  con `name -> (offset, size)`.
- `slot_ranges`: `[(il_start, il_end), ...]` para cada slot lógico.
- `pool_initial_layers[p]`: snapshot de `model.layers[il]` para los layers
  del pool `p`, capturado al setup.
- `pool_current_slot[p]`: qué slot lógico está actualmente materializado
  en el pool `p` (inicialmente `p`).

Para hacer residente el slot lógico `K` en el pool `P`:

1. `ggml_backend_sched_synchronize(sched)` — esperar a que cualquier
   cómputo previo que referenciaba los tensors de este pool termine.
2. Para cada tensor `blk.<phy>.X` en `model.tensors_by_name` con
   `phy_il ∈ [P_start, P_end]`:
   - Computar el nombre lógico equivalente: reemplazar `<phy_il>` con
     `K_start + (phy_il - P_start)`.
   - Buscar su `(offset, size)` en el mapa GGUF.
   - **Rechazar el swap si los tamaños no coinciden** (este es el blocker de Gemma).
   - `pread()` los bytes del fd GGUF.
   - `ggml_backend_tensor_set(dst_tensor, bytes, ...)` los escribe al buffer
     del tensor del pool (sin repack — `--no-repack` está forzado).
3. Re-bind: `model.layers[K_start..K_end] = pool_initial_layers[P]`. Ahora
   las referencias del graph builder a `model.layers[K_start + i].wq` etc.
   resuelven a los tensors del pool que físicamente tienen los bytes del slot K.
4. `pool_current_slot[P] = K`.

La asignación de pool es round-robin: `pool_idx = K % slots_resident`.
Cada forward pass re-swappea cada slot no residente porque los contenidos
del pool del pase anterior están stale (y los residentes 0..R-1 también
necesitan volver a swappearse en pases subsiguientes).

### Decode driver

`llama_context::decode_slotted_real_test` (en `src/llama-context.cpp`)
espejo las partes de `decode()` que manejan inicialización de batch,
preparación de KV cache y output ID mapping, con el loop interno de
ubatch reemplazado por:

```cpp
for cada ubatch:
    mctx->apply()
    hidden = zero_vec(n_embd * n_tokens)
    for cada slot K en plan:
        if pool[K % R] != K:
            sched_synchronize()
            hot_swap_swap_in(K, K % R)
        build slot graph (slot_il_start, slot_il_end seteados en params)
        sched_alloc_graph
        set_inputs
        if no es primer slot: tensor_set(carry, hidden)
        graph_compute
        if no es slot final: tensor_get(t_embd, hidden)
        else: tensor_get_async(t_logits -> output buffer)
sched_synchronize
finalize output_ids[]
```

### Bypasses necesarios para la integración

Algunos paths existentes de `llama.cpp` asumen que el modelo está
totalmente materializado. Los deshabilitamos cuando
`cparams.slotted_real_skip_sched_reserve` es true:

- `llama_context::sched_reserve()` — inicializa `gf_res_prev`,
  `gf_res_reserve` y `sched`, pero saltea la pasada full-graph que intentaría
  armar un cgraph tocando tensors NULL.
- `common_context_can_seq_rm()` — short-circuit del probe de decode de 2
  tokens que `server-context.cpp` hace al inicio.
- El repack buffer (`--no-repack` forzado cuando slotted-real + decode test
  están ambos activos).

## Referencia de flags CLI

Estos flags viven en `common/arg.cpp` y son CLI-only (`LLAMA_EXAMPLE_CLI`).

### Instrumentación (FASE 1-3)

| Flag | Efecto |
|---|---|
| `--slotted-test` | Arma el slot plan desde el modelo cargado y lo loguea. Sin cambio de comportamiento. |
| `--slot-size-mb N` | Agrupa layers hasta ~N MiB por slot. |
| `--slot-layers N` | Agrupa layers en slots de N layers fijos. Prioridad sobre `--slot-size-mb`. |
| `--slotted-log` | Logging verboso per-slot durante decode. |
| `--slotted-json PATH` | Escribe slot plan + summary a PATH como JSON. |
| `--slotted-simulate-prefetch` | Corre un `std::thread` por transición de slot que simula el prefetch del próximo slot. |
| `--slotted-bandwidth-mb-s N` | Bandwidth que usa `mode=sleep` para convertir bytes a ms. |
| `--slotted-prefetch-mode MODE` | `sleep` (default) o `dummy-read` (pread desde un fd separado). |

### Loader filter (FASE 4A-1)

| Flag | Efecto |
|---|---|
| `--slotted-real` | Instala el filter de layer para cargar solo los primeros `--slots-resident` slots. |
| `--slots-resident N` | Número de slots a mantener residentes en RAM. Default 2. |

### Slotted decode (FASE 4A-2a, b)

| Flag | Efecto |
|---|---|
| `--slotted-decode-test` | Corre `llama_decode_slotted_test` (o `_real_test` si se combina con `--slotted-real`) sobre el prompt `-p`, samplea `-n` tokens greedy, imprime. |
| `--slotted-decode-baseline` | Corre `llama_decode` normal sobre el mismo prompt y sampling para comparación. |
| `--slotted-chat-poc` | Chat loop interactivo usando el runtime slotted-real. Lo usa `poc.c`. |
| `--slotted-round-robin` | Opt-out de la política default pin-and-scratch del hot-swap; usar la legacy round-robin (`pool_idx = slot_idx % R`). Ver "Política del pool de hot-swap" abajo. |

## Política del pool de hot-swap

El runtime soporta dos políticas de reemplazo para el pool de buffers:

- **`pin-and-scratch` (default, cuando `slots_resident >= 2`)**: pinea los
  slots `0..R-2` en pools `0..R-2` durante toda la corrida; el pool `R-1`
  es el scratch único que rota entre `slots R-1..N-1`. Los slots `0..R-2`
  siempre dan hit. Steady-state de swaps por forward pass: `N - R + 1`.
- **`round-robin` (legacy, opt-in con `--slotted-round-robin`)**: asigna
  `pool_idx = slot_idx % R`. Cada steady-state pass re-swappea todos los
  `N` slots porque el pool termina cada pass conteniendo los últimos `R`
  slots accedidos. Steady-state de swaps por forward pass: `N`.

Para nuestro workload de referencia (Llama 3.1 70B Q3_K_XL, `N=8`, `R=2`,
prompt "The capital of France is", `-n 8`):

| Métrica               | round-robin | pin-and-scratch | delta |
|---|---:|---:|---:|
| Total swaps           | 102 | 90 | -12 (-11.8%) |
| Bytes leídos          | 425 GB | 375 GB | -50 GB (-11.8%) |
| Suma read_ms          | 264.2 s | 195.0 s | -26.2% |
| Suma set_ms           | 157.1 s | 68.7 s | -56.3% |
| **Real time**         | **311.2 s** | **251.9 s** | **-59.3 s (-19.0%)** |
| Peak memory footprint | 11,357 MB | 11,362 MB | idéntico |
| OS swaps              | 0 | 0 | idéntico |
| Texto generado        | `a city of love, art, fashion` | `a city of love, art, fashion` | idéntico |

> pin-and-scratch is the default policy because it reduced runtime by 19%
> on MacBook Air M4 24GB with Llama 3.1 70B Q3_K_XL, with identical output
> and no additional peak memory footprint.

La baja inesperadamente grande del `set_ms` (56% vs el 12% de bajada en
cantidad de swaps) se debe a que round-robin escribe a **ambos** buffers del
pool alternados, manteniendo ambos sets de páginas calientes en cache.
Pin-and-scratch solo escribe al pool `R-1` después del init, así que las
páginas del pool pineado pueden quedarse frías y hay menos competencia por
cache del lado de escritura.

## Archivos

```
include/llama.h                              # API público
common/common.h, common/arg.cpp              # common_params + CLI flags
common/common.cpp                            # forward a cparams
common/slotted-inference.h, .cpp             # plan builder, runtime de callback FASE 3
src/llama-graph.h                            # campos de slot en llm_graph_params
src/llama-cparams.h                          # cparams.slotted_real_skip_sched_reserve
src/llama-model-loader.h, .cpp               # integración del filter en create_tensor
src/llama-model.cpp                          # accounting de tamaño por layer expuesto
src/llama-context.h, .cpp                    # decode_slotted_test + decode_slotted_real_test
src/llama-slotted-runtime.h, .cpp            # pool de hot-swap + C API
src/llama.cpp                                # forward del filter desde model params al loader
src/models/llama.cpp                         # graph builder de Llama con conciencia de slot
src/models/gemma4.cpp                        # graph builder de Gemma 4 con conciencia de slot
tools/cli/cli.cpp                            # test harness + chat loop
poc.c                                        # launcher del demo (raíz del repo)
```

## Bitácora de fases

| Fase | Objetivo | Resultado |
|---|---|---|
| FASE 0 | Mapear la base de código (loader, graph, decode, CLI). | Refs file:line recolectadas. |
| FASE 1 | Agregar CLI flags + slot planner, solo log. | Plan impreso para Gemma 4 31B (60 layers, 9 slots @ 2 GB). |
| FASE 2 | Estructuras de datos del slot planner. | Plegado en FASE 1. |
| FASE 3 | Simulación de carga async + timing per-slot vía cb_eval. | `io_hidden_percent` medido en CPU (100%) y Metal partial (40-86%). |
| FASE 4A-1 | Loader filter, sin inferencia. | Peak footprint escala lineal con `slots_resident`. |
| FASE 4A-2a | Ejecución per-slot, todos los slots cargados. | Top-1 token y logit matchean el baseline monolítico bit-for-bit, validado en 16 tokens. |
| FASE 4A-2b | Hot-swap + 1 token con `slots_resident=2`. | Funciona en Llama 3.1 70B Q3_K_XL (35 GB → 11 GB peak); bloqueado en Gemma 4 31B por heterogeneidad de estructura de layers. |
| FASE 4B | M4 kernel lab (microbenchmark). | Diseñado, no implementado todavía. |
| FASE 5 | Eviction real con re-creación de tensors per slot. | Fuera de scope de esta ronda. |

## Configuración validada

```
Modelo:      Meta-Llama-3.1-70B-Instruct Q3_K_XL  (35 GB)
Backend:     CPU only (Apple M4 / 24 GB)
Build:       cmake -B build -DGGML_METAL=OFF -DLLAMA_BUILD_SERVER=ON
Flags:       --n-gpu-layers 0 --no-mmap --no-warmup --no-repack
             --ctx-size 128 --batch-size 8 --ubatch-size 8
             --slot-layers 10 --slotted-real --slots-resident 2
             --slotted-chat-poc -n {4|8|16|32}
Resultado:   Generación coherente. Peak memory footprint ~11 GB consistente.
             Cero swaps del sistema operativo. Hard page faults < 700.
             ~20-30 s por token generado en n=4..8.
```

Ejemplos: `Hello, my name is` → ` John and I am a 30-year-old` y
`The capital of France is` → ` a city of love, art, fashion`.

Los números de la fila FASE 4A-2b de la "Bitácora de fases" arriba se
midieron con la policy legacy round-robin. Con la default pin-and-scratch
el mismo run completa en 19% menos wall-clock en el mismo hardware (ver
"Política del pool de hot-swap" abajo).

## Limitaciones conocidas

1. **Layers heterogéneos rompen el hot-swap.** Modelos donde los shapes
   de tensor per-layer varían (Gemma 4 con dense/MoE intercalado, Mixtral,
   variabilidad de n_heads, atención alternativa con v_proj opcional)
   fallan con mismatches de tamaño cuando el tensor del slot lógico no
   coincide con el del pool en el mismo offset. El runtime actual rechaza
   el swap en lugar de corromper data silenciosamente.

2. **`--no-repack` es requerido.** El buffer type CPU_REPACK formatea los
   pesos al alocar para layouts SIMD-friendly. Replicar esa transformación
   on-the-fly durante un hot-swap es trabajo de FASE 5. Sin repack, el
   matmul de CPU es ~30% más lento.

3. **`--no-mmap` es requerido para medición fiel.** Con mmap el OS pagina
   los pesos lazy; el RSS reportado subestima la presión real de memoria.
   El path de hot-swap mismo no depende de que mmap esté off, pero el
   demo usa `--no-mmap` para que los números de `/usr/bin/time -l` sean
   significativos.

4. **Una sola arquitectura adaptada por vez.** Las modificaciones al graph
   builder con conciencia de slot viven en `src/models/llama.cpp` y
   `src/models/gemma4.cpp`. Otras arquitecturas necesitan el mismo patrón
   replicado: agregar lectura de `slot_il_start/end`, branchear en
   `slot_is_first` para el input de embedding, branchear en `slot_is_final`
   para el output norm / lm_head, y reemplazar los bounds del loop de layers.

5. **`sched_reserve` se saltea, no se reemplaza.** El path de decode
   slotted reserva sus propios graphs per-slot a medida que va. El primer
   forward pass paga un costo de alocación mayor que los siguientes porque
   el scheduler todavía no vio la topología de ningún graph de slot.

6. **La KV cache se aloca para todos los layers.** Es chica (~200 MiB con
   `--ctx-size 128`) y no es el cuello de botella, pero el run slotted-real
   sigue alocando espacio KV para los layers cuyos pesos fueron salteados
   por el load filter. Esto es intencional: el estado KV debe persistir a
   través de las rotaciones de slot dentro de un forward pass.

7. **Política de reemplazo del pool.** El default es `pin-and-scratch`
   (pinear R-1 slots, usar 1 scratch); la legacy round-robin queda
   disponible vía `--slotted-round-robin`. Ver "Política del pool de
   hot-swap" para el trade-off medido. Ambas siguen siendo subóptimas vs
   Belady (óptima para el patrón de acceso cíclico), pero el lookahead
   estilo Belady no está implementado todavía.

## Disclosure

Esto es código de investigación experimental en un fork privado. No está
pensado para enviar upstream a `ggml-org/llama.cpp` tal como está. Ver
`AGENTS.md` y `CONTRIBUTING.md` para la política de contribución upstream.
El trabajo de slotted-real fue desarrollado con asistencia AI sustancial
y debería ser revisado end-to-end por un contribuyente humano antes de
que cualquier parte se proponga upstream.
