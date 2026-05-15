# Compact Pipeline Summary

Compact U8 output remains useful when the downstream consumer stays compact: it beats the float-output consumer path on this run.

| Path | Correct | Median time | Speedup vs float default | Data moved |
| --- | --- | ---: | ---: | ---: |
| Drop-in float default | yes | 0.1848 ms | 1.00x | 512.0 MiB |
| FP16 output | yes | 0.0629 ms | 2.94x | 256.0 MiB |
| Compact U8 input + FP16 output | yes | 0.0398 ms | 4.64x | 160.0 MiB |
| Pack U8 input setup | yes | 0.0491 ms | 3.76x | 288.0 MiB |
| Pack U8 input + FP16 output | yes | 0.0933 ms | 1.98x | 416.0 MiB |
| Pack U8 input + compact U8 output | yes | 0.0892 ms | 2.07x | 320.0 MiB |
| Pack U8 input + compact U8 output + float decode | yes | 0.1393 ms | 1.33x | 576.0 MiB |
| Float output + score consumer | yes | 0.1399 ms | 1.32x | 832.0 MiB |
| Compact U8 output + score consumer | yes | 0.1162 ms | 1.59x | 448.0 MiB |
