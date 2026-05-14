# poc.c — launcher del demo de chat slotted-real

Programa C de 100 líneas que lanza el `llama-cli` modificado con todos los
flags necesarios para correr **Meta-Llama-3.1-70B-Instruct Q3_K_XL (~35 GB
en disco)** en una máquina donde el modelo completo no entra cómodamente en
RAM. Con `--slots-resident 2`, el peak footprint queda en ~**11 GB**.

`poc.c` es intencionalmente solo un launcher. Toda la infraestructura
slotted-real (filtro de carga, builder por slot, runtime de hot-swap) vive
dentro de `llama.cpp` mismo. Ver [`README.es.slot.md`](README.es.slot.md)
para arquitectura y notas de diseño.

## Requisitos

- macOS o Linux
- `clang` (o cualquier compilador C)
- `llama-cli` ya compilado (el de este fork, con soporte slotted-real)
- Una copia local de Llama 3.1 70B Q3_K_XL en
  `~/models/llama31-70b/Meta-Llama-3.1-70B-Instruct-Q3_K_XL.gguf`

Compilar `llama-cli` primero:

```bash
cmake -B build -DGGML_METAL=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release -j --target llama-cli
```

## Compilar el launcher

Desde la raíz del repo:

```bash
clang poc.c -o poc
```

Es un solo archivo fuente, sin dependencias más allá de `libc`.

## Correr

```bash
./poc           # 16 tokens por turno (default)
./poc 32        # 32 tokens por turno
./poc 8         # 8 tokens por turno
```

El launcher verifica que existan `./build/bin/llama-cli` y el archivo del
modelo, después hace `execv` de `llama-cli` con estos flags hardcodeados:

```
--n-gpu-layers 0
--no-mmap
--no-warmup
--no-repack
--ctx-size 128
--batch-size 8
--ubatch-size 8
--slot-layers 10
--slotted-real
--slots-resident 2
--slotted-chat-poc
-n <N>
```

`-n <N>` se toma de `argv[1]` si lo pasás, default 16.

## Qué ves al correrlo

```
=========================================================================
  llama.cpp slotted-real demo
=========================================================================
  Model:     Meta-Llama-3.1-70B-Instruct Q3_K_XL (~35 GB on disk)
  Mode:      CPU only, --no-mmap, --no-repack
  Slot plan: 8 slots x 10 layers,  --slots-resident 2

  This is intentionally slow. It proves a 35GB 70B model can run
  with a reduced memory footprint (~11 GB peak instead of ~35 GB).
=========================================================================

Loading model... |/-\
=======================================================================
  Type a prompt and press Enter. Empty line / Ctrl-D exits.
=======================================================================

> Hi
, I'm looking

> The capital of France is
 a city of love

> Hello, my name is
 John and I am a 30-year
```

Cada turno:
1. Resetea la KV cache (sin memoria de conversación entre turnos).
2. Tokeniza el prompt con BOS.
3. Lo pasa por el slotted decode (un token por llamada a
   `llama_decode_slotted_real_test`).
4. Greedy-samplea la cantidad de tokens pedida.
5. Imprime cada token a medida que se produce.

Enter en línea vacía o Ctrl-D para salir.

## Por qué cada flag está hardcodeado

| Flag | Valor | Razón |
|---|---|---|
| `--slot-layers 10` | 10 | 80 layers / 10 = 8 slots uniformes. Evita el size mismatch que rompe el grouping por tamaño. |
| `--slots-resident 2` | 2 | Mínimo viable. Peak footprint ~11 GB. Valores mayores usan más RAM y reducen frecuencia de hot-swap. |
| `--no-mmap` | on | Necesario para que el RSS reportado por el OS refleje el costo real de memoria. |
| `--no-repack` | on | El hot-swap escribe bytes raw del GGUF directo a los buffers del pool; el formato REPACK SIMD requeriría re-empaquetar on-the-fly (FASE 5). |
| `--ctx-size 128` | 128 | KV cache chica. El tamaño de KV es independiente de cuántos slots están residentes. |
| `--batch-size 8` / `--ubatch-size 8` | 8 | El path slotted es batch=1; estos son defaults seguros. |
| `--no-warmup` | on | El warmup default intenta armar un cgraph completo a través del modelo filtrado, lo que derefenciaría tensors NULL para layers no residentes. |

La política del pool de hot-swap está en **pin-and-scratch** por default
(introducida en el benchmark de FASE 4A-2b). Eso bajó 19% el runtime en
MacBook Air M4 24GB con Llama 3.1 70B Q3_K_XL vs la legacy round-robin,
con output idéntico y sin overhead de memoria. Para forzar el
comportamiento legacy (A/B), corré `llama-cli` directo agregando
`--slotted-round-robin`; el launcher no tiene flag para esto.

Si necesitás apuntar el launcher a otra ruta de modelo, edita el macro
`MODEL_REL` arriba de `poc.c`.

## Expectativas de performance

El tiempo de generación escala aproximadamente lineal con `n_predict`. En
un Apple M4 con page cache calentado:

- ~5 minutos para `./poc 4` (4 tokens, incluyendo load + procesamiento del prompt)
- ~7 minutos para `./poc 16`
- ~15 minutos para `./poc 32`

El path de hot-swap re-lee aproximadamente 30 GB por forward pass (8 slots
menos los 2 residentes, ~4.4 GB cada uno). Para generaciones largas el page
cache del OS absorbe la mayoría; `block input operations` debería quedarse bajo.

## Validación cross-platform

Tanto `poc` (70B Q3_K_XL) como `poc_tulu` (405B Q3_K_M) se smoke-testearon
también en una **laptop tipo ThinkPad Intel Core i7 Evo con 32 GB de RAM**
corriendo Linux / x86_64. Los dos llegaron a fin con output coherente, sin
cambios de código. El wall-clock fue aproximadamente **~2× del target M4
24 GB para 70B** y **~3× para 405B**, atribuible al NVMe con throughput
sostenido más bajo más el dispatch path AVX2 que está menos optimizado
que el path NEON `sdot` para `matmul` Q-quant en Apple Silicon.

La MacBook Air M4 24 GB sigue siendo el target principal de optimización.
El resultado x86 se incluye acá porque demuestra que el path slotted-real
en sí mismo no está atado a un ISA específico ni a la arquitectura de
memoria unificada de Apple Silicon — mismo binario, mismos flags,
máquina distinta, el modelo corre.

## Limitaciones

1. **Solo familia Llama 3.1.** Modelos con estructura de layers heterogénea
   (Gemma 4 con MoE, Mixtral, etc.) fallan actualmente en el hot-swap con
   mismatch de tamaño de tensor. Ver `README.es.slot.md` para detalles.
2. **CPU only.** Los paths Metal/CUDA no se adaptaron.
3. **Sin memoria de conversación.** Cada turno es independiente.
4. **Lento.** Es un demo de eficiencia de memoria, no de performance.
5. **Costo de `--no-repack`.** Sin el SIMD weight-packing, el matmul de CPU
   es ~30% más lento por layer que el path default.
6. **Los weights del slot 0 quedan válidos hasta el primer hot-swap en pool 0.**
   Después de eso, `model.layers[0..9]` siguen apuntando a los tensors del pool 0
   pero los *datos* fueron sobreescritos. Cualquier código que toque esos índices
   de layer fuera del slotted decode loop leerá el slot lógico que se cargó último.

## Archivos que toca esta PoC

- `poc.c` (este directorio) — el launcher
- `tools/cli/cli.cpp` — el chat loop de `--slotted-chat-poc`
- `src/llama-slotted-runtime.{h,cpp}` — el runtime de hot-swap pool
- `src/llama-context.cpp` / `src/llama-context.h` — `decode_slotted_real_test`
- `src/models/llama.cpp` — graph builder con conciencia de slot
- `include/llama.h` — extensiones del API C público

Para un walkthrough más profundo de qué hace cada pieza, leer
[`README.es.slot.md`](README.es.slot.md).

## Launcher hermano: `poc_tulu.c` (Llama 3.1 405B Q3_K_M)

`poc.c` apunta al 70B Q3_K_XL. Un launcher hermano, `poc_tulu.c`, apunta
el mismo binario `llama-cli` a **Llama 3.1 Tulu-3 405B Q3_K_M** (~200 GB
en 5 shards), con peak memory footprint de **~12.6 GB** en una MacBook
Air M4 de 24 GB. Esto se habilitó con FASE 4A-3 (soporte multi-shard +
mixed-quant en `src/llama-slotted-runtime.{h,cpp}`).

Compilar y correr:

```bash
clang poc_tulu.c -o poc_tulu
./poc_tulu          # 4 tokens por turno (default, ~9 min en M4 24 GB)
./poc_tulu 8        # 8 tokens por turno (~20 min)
./poc_tulu 1        # 1 token (smoke test)
```

`poc_tulu` espera el shard 1 del modelo en
`$HOME/models/llama31-405b-tulu-q3km/Llama-3.1-Tulu-3-405B-Q3_K_M/Llama-3.1-Tulu-3-405B-Q3_K_M-00001-of-00005.gguf`,
con los otros 4 shards al lado. El runtime descubre los demás desde la
metadata key `split.count`, así que sólo se pasa el path del shard 1 al
`llama-cli`. Editá `MODEL_REL` arriba de `poc_tulu.c` para apuntar a
otra ruta.

Los flags hardcodeados difieren de `poc.c` solo en el slot plan:

| Flag | `poc` (70B) | `poc_tulu` (405B) | Razón |
|---|---|---|---|
| `--slot-layers` | 10 | 3 | 80 / 10 = 8 slots vs 126 / 3 = 42 slots. Uniformes. |
| Default `-n` | 16 | 4 | El 405B va a ~138 s/token; los defaults apuntan a un budget similar de wall-clock por turno. |

Trade-off vs 70B: ~138 s/token en vez de ~20-30 s/token, porque cada
forward pass re-lee decenas de GB de pesos y el hit rate del page cache
es menor con un working set de 200 GB contra 24 GB de RAM. La latencia
es intencional; el objetivo es demostrar que la ejecución por slots
escala a modelos clase 405B en hardware de consumo, no ser rápido.

Ver [`README.es.slot.md`](README.es.slot.md), sección "Llama 3.1 Tulu-3
405B Q3_K_M (FASE 4A-3)" para la tabla completa de mediciones.

## Disclosure

Esto es código de investigación experimental en un fork privado. No está
pensado para enviar upstream a `ggml-org/llama.cpp` tal como está. El
objetivo fue validar la idea de slotted-execution sobre un modelo GGUF
real antes de decidir si vale la pena invertir en una implementación
de producción. Ver `CONTRIBUTING.md` y `AGENTS.md` para la política de
contribución del proyecto upstream.
