# C++20 升级调研笔记

本文件记录把 TensorRT-LLM 的 C++/CUDA 源码从 C++17 提升到 C++20 的调研过程、
遇到的全部问题及其修法。**本次 release 不做 C++20 升级，此分支为备份，供下个 release 继续。**

所有结论均来自在 dlcluster 上用 CI 镜像
`pytorch-26.08-py3-x86_64-ubuntu24.04-skip-tritondevel-202608251800-18126`
（nvcc 13.4.59，torch `2.14.0a0+nv26.8`）实跑 4 轮 build 得到，非静态推理。

---

## 1. 为什么需要 C++20

公开 PyPI 的 **torch 2.14.0** 在 `aten/src/ATen/ATen.h` 开头硬编码：

```cpp
#if !defined(_MSC_VER) && __cplusplus < 202002L
#error C++20 or later compatible compiler is required to use ATen.
#endif
```

torch 2.13.0 同一位置要求的还是 C++17。这个 `#error` 没有开关可关，只能真的用 `-std=c++20`。

**只有走公开 PyPI wheel 的路径会撞上**：NGC 容器自带的 `2.14.0a0+nv26.08` 分支点更早，
还没带这个 `#error`，所以主 build 一直是绿的，挂的是三个用公开 torch 的 PackageSanityCheck
stage（`A10-PY310-UB2204` / `A100X-PY312-UB2404` / `GH200-PY312-UB2404`）。

### 编译器可用性（已实测）

Rocky8 构建镜像（`LLM_ROCKYLINUX8_PY{310,312}_DOCKER_IMAGE`）里是 **gcc 11.2.1**
（`gcc-toolset-11`，另外还装了 `gcc-toolset-15`）。实测 `-std=c++20` 下
`__cplusplus == 202002L`，concepts / ranges / `bit_cast` / designated init /
括号聚合初始化全部可用。**编译器不是障碍。**

---

## 2. 改了哪些地方

### 2.1 标准切换本身（7 个文件，8 行）

`cpp/CMakeLists.txt` 的 `CMAKE_CXX_STANDARD 17` → `20`
（`CMAKE_CUDA_STANDARD` 是既有的 `${CMAKE_CXX_STANDARD}`，会跟着变）。

另外 6 处显式写死 `"17"` 的 target 必须一起改，否则会覆盖全局值：
`tensorrt_llm/`、`batch_manager/`、`executor/`、`executor/.../ucx_utils/`、
`micro_benchmarks/`、`tests/`（后者是 `cxx_std_17` → `cxx_std_20`）。

不碰 torch 的 vendored target（`deep_gemm`、`deep_ep`、`mamba2MTPSSMCache`、
`fmha_v2`、`xqa`）保持 C++17。

**CUDA 侧无法单独留在 17**：只有 18 个 `.cu` 传递包含 torch 头，其中 12 个在不参与主 build 的
`fmha_v2/train_ops`；但剩下的 `kernels/cuda_graph_grouped_gemm.cu` 与 `gptKernels.cu`
同属 `kernels_src` 这一个 target，而 `CUDA_STANDARD` 是 target 级属性，没法只给单个源文件设。

### 2.2 代码适配（12 个文件，+121/−72）

见下面四个问题。

---

## 3. 四个 C++20 问题

### 问题 1：P0960 括号聚合初始化打断 fp8 → float2/float4 转换

**症状**：`decoderMaskedMultiheadAttentionUtils.h` 46 处 +
`llama4MinLatencyKernels/` 23 处 + `decoderMaskedMultiheadAttentionTemplate.h` 2 处，
共 70 处报

```
error: no suitable conversion function from "__nv_fp8x2_e4m3" to "float" exists
      float2 fb0 = float2(fp8_2[0]);
```

**机理**：`cuda_fp8.hpp` 里的转换运算符是 **`explicit`** 的：

```cpp
explicit __CUDA_HOSTDEVICE_FP8__ operator float2() const;   // __nv_fp8x2_e4m3
explicit __CUDA_HOSTDEVICE_FP8__ operator float4() const;   // __nv_fp8x4_e4m3
```

而 `float2`/`float4` 是 `vector_types.h` 里的**纯聚合**（无构造函数）。
C++17 下 `float2(v)` 是显式转换，走那个 explicit operator；C++20 起
`T(单实参)` 对聚合改走括号聚合初始化（P0960），编译器转而尝试把 fp8 值转成
**第一个成员 `float`**。注意报错写的是 "to **float**" 而非 "to float2"，
这是判定聚合初始化被选中的关键线索。

**实测矩阵（nvcc 13.4）**：

| 写法 | c++17 | c++20 |
|---|---|---|
| `float2(v)` 函数式转换 | OK | **FAIL** |
| `float2 r = v;` 拷贝初始化 | **FAIL** | **FAIL** |
| `static_cast<float2>(v)` | OK | OK |
| `v.operator float2()` | OK | OK |
| `__half22float2(__half2(v))` | OK | OK |

⚠️ 拷贝初始化两个标准下都不行 —— 因为 operator 是 `explicit`，拷贝初始化不允许用它。
**第一次修错就栽在这里**：从"C++17 能编过"推出"存在非 explicit 的转换运算符"，
第二步是错的，上机跑到 32/5511 才炸出来。

**修法**：在 `common/cudaTypeUtils.cuh` 的 `#ifdef ENABLE_FP8` 段加两个 helper，
内部具名调用转换运算符：

```cpp
__device__ inline float2 fp8x2_to_float2(__nv_fp8x2_e4m3 val) { return val.operator float2(); }
__device__ inline float4 fp8x4_to_float4(__nv_fp8x4_e4m3 val) { return val.operator float4(); }
```

`static_cast` 同样可行且可读性更好，可按 review 偏好二选一。

**等价性已在二进制层面证明**：同一个 kernel 用旧写法（`-std=c++17`）和新 helper
（`-std=c++20`）编译，`cuobjdump -sass` **逐行完全相同**，指令数都是 50。
生成的是 Hopper 的单条硬件转换指令 `F2FP.F16.E4M3.UNPACK_B` + 两条 `HADD2.F32`，
说明转换路径没变，精度与性能零影响。

> 曾考虑过的另一条路 —— 改用仓库已有的 `bf1622float2(fp8x2_e4m3_to_bfloat2(&val))`
> 走 bf16 中转 —— **不要用**。虽然 e4m3 的 3 bit 尾数经 bf16 的 7 bit 尾数中转
> 数学上无损，但那是不同的指令序列，需要额外论证。具名调用转换运算符是零风险的。

### 问题 2：P0960 经 `std::is_constructible` 静默改变 `if constexpr` 分支

**症状**：24 个 cutlass grouped GEMM 实例化报

```
moe_gemm_tma_ws_launcher.inl(78): error: a value of type "const float **"
    cannot be used to initialize an entity of type "float"
        return ReturnType{std::forward<Args>(args)...};
```

**机理**：`moe_gemm_tma_ws_launcher.inl:603`

```cpp
constexpr bool IsSimpleAlphaBeta
    = std::is_constructible_v<EpilogueScalars, ElementAccumulator, ElementAccumulator>;
```

`EpilogueScalars` 是 CUTLASS 的
`FusionCallbacks<Sm90PtrArrayTmaWarpSpecialized, LinearCombination, ...>::Arguments`，
一个纯聚合（无构造函数，只有 NSDMI + 一个转换运算符）：

```cpp
struct Arguments {
    ElementScalar alpha = ElementScalar(1);          // ← 第 1 个成员
    ElementScalar beta = ElementScalar(0);
    ElementScalar const* alpha_ptr = nullptr;
    ElementScalar const* beta_ptr = nullptr;
    ElementScalar const* const* alpha_ptr_array = nullptr;   // ← 第 5 个成员
    ElementScalar const* const* beta_ptr_array = nullptr;
    StrideAlpha dAlpha; StrideBeta dBeta;
};
```

于是：

| | `is_constructible_v<Arguments,float,float>` | `IsSimpleAlphaBeta` | 选中分支 |
|---|---|---|---|
| C++17 | **false**（聚合不能括号初始化） | false | line 635，传全套 8 个参数 ✅ |
| C++20 | **true**（P0960） | true | line 645，只传 1 个 `const float**` ❌ |

C++20 选中的分支把 `const float**` 喂给第一个成员 `float alpha`。

**已用最小 repro 实证**（gcc，同一个纯聚合类型）：

```
c++17:  is_constructible<Agg,float,float> = 0    is_aggregate<Agg> = 1
c++20:  is_constructible<Agg,float,float> = 1    is_aggregate<Agg> = 1
```

**修法**：把判据拉回 C++17 语义，只认"真有 (alpha, beta) 构造函数"的类型：

```cpp
constexpr bool IsSimpleAlphaBeta
    = std::is_constructible_v<EpilogueScalars, ElementAccumulator, ElementAccumulator>
    && !std::is_aggregate_v<EpilogueScalars>;
```

C++17 下 `is_constructible` 为 true 的类型必然非聚合，所以这个 `&&` 不改变 C++17 行为，
只把 C++20 拉回原分支，运行时语义零变化。

⚠️ **这是整个升级里唯一改变了编译期分支选择的改动**，review 时应重点看。
另外它也说明：C++20 未修复时代码走的是一条**在 C++17 下从未被编译过的路径**；
假如它侥幸编过，epilogue 的 alpha/beta 就会被喂错值，属于静默数值错误。

**升级 CUTLASS 救不了这个（已实查 v4.8.0dev）**：
- `pip install nvidia-cutlass-dsl` 是 Python CuTe DSL 包，与 C++ 编译无关；
  C++ header 的 pin 在 `3rdparty/fetch_content.json` 的 `cutlass.git_tag`（现 `v4.4.2`）
- v4.4.2 与 v4.8.0dev 的那个 `Arguments` **逐字段完全相同**，都是纯聚合
  （构造函数声明数 = 0）→ C++20 下 `is_constructible` 依旧为 true
- CUTLASS 4.8.0dev 的 `CMakeLists.txt` 仍是 `CMAKE_CXX_STANDARD 17`，上游未为 C++20 适配
- 若要升 C++ header 到 4.8，必须同步改 `3rdparty/CMakeLists.txt:115` 的
  `MSA_VALIDATED_CUTLASS_TAG`（硬编码 `"v4.4.2"`，不同步会 `FATAL_ERROR`）

### 问题 3 / 4：P1008 —— `= default` 构造函数使类不再是聚合

C++17 里只有 *user-provided* 构造函数才排除聚合；C++20 起任何 *user-declared*
（包括 `= default` / `= delete`）都排除。于是这两处的花括号初始化在 C++20 下失效：

| 位置 | 声明 | 用法 |
|---|---|---|
| `include/.../batch_manager/microBatchScheduler.h:32` | `ContextChunkingConfig() = default;` | `ContextChunkingConfig{policy, chunkUnitSize}` |
| `tests/unit_tests/runtime/localizationTest.cu:152` | `StreamHolder() = default;` + deleted copy | `StreamHolder{createLocalizedStream(0)}` |

前者影响面更大 —— 它在 public header 里，波及 nanobind 绑定（`bindings.cpp`）
和 `microBatchSchedulerTest.cpp` 的 8 处调用。

**修法**：各补一个显式构造函数（保留原 `= default`，不改变任何现有语义）。

**已做过全仓扫描**：找"只有 defaulted/deleted 构造函数、却被多参数花括号初始化"的类型，
71 个候选里命中 2 个（`RequestInfo`、`StagingBufferManager`），逐个核实后**都是误判**
（它们有真正的 user-provided 构造函数，只是声明与定义分离）。与实编结果一致。
**P1008 隐患就是上面两处。**

---

## 4. 不属于 C++20 的问题（已排除，勿混淆）

### 6 个 gtest 链接失败 —— 与 C++20 无关

```
ld: ucxCommTest.cpp.o: undefined reference to symbol 'c10::IValue::reportToTensorTypeError()'
ld: /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so:
    error adding symbols: DSO missing from command line
```

涉及 `agentCommTest`、`genUniqueAgentNameTest`、`serializeUtilsTest`、`ucxCommTest`、
`blockScaleMoeActivationTest`、`routingKernelsTest`。CI 会通过
`tests/integration/defs/cpp/conftest.py:172` 的 `extra_make_targets=["google-tests"]` 构建它们。

**三阶段对照实验（同一份代码、同一容器，只动一个变量）**：

| 阶段 | 配置 | 结果 |
|---|---|---|
| A | tests 用 `cxx_std_20`（现状） | 6 个全部失败 |
| B | 仅把 tests 退回 `cxx_std_17` | **同样 6 个全部失败** |
| C | `cxx_std_20` + `target_link_libraries(${test_name} PUBLIC ${TORCH_LIBRARIES})` | **全部成功** |

Phase B 证明与 C++ 标准无关。根因是 `cpp/tests/CMakeLists.txt` 的 `add_gtest`
从来没链接过 torch（只链接 `${SHARED_TARGET}`），最可能是 DLFW 26.08 容器里
torch 2.14.0a0 的符号分布变化才让它暴露 —— 应归 DLFW 升级，不属于 C++20 或 torch pin 改动。

**修法已验证**：在 `add_gtest` 里加 `target_link_libraries(${test_name} PUBLIC ${TORCH_LIBRARIES})`。
小瑕疵：会给所有 gtest 都加上，包括带 `NO_TLLM_LINKAGE` 的那些其实不需要 torch 的 target
（无害，只是多链一个已加载的 so）。

### fmha 的 `-Wdeprecated-enum-enum-conversion` 警告 12572 条

C++20 新增的弃用警告（不同枚举类型间做算术）。fmha 不在加 `-Werror` 的五个目录
（runtime / common / batch_manager / executor / thop）内，所以**不致命**，
但会严重污染 build 日志、拖慢编译。下次升级时值得顺手清理或显式压制。

### OOM

`fp8_rowwise_gemm_bf16/fp16.cu.o` 在 `-j32` 下被 OOM killer 杀掉
（`gcc: fatal error: Killed signal terminated program cc1plus`）。
32 核 / 125G 内存，CUTLASS 大模板每个 TU 峰值可超 4G。**`-j16` 后为 0**。
CI 用 `-j8`，本来就不会遇到。不是代码问题。

---

## 5. 升级前做过的风险调研（结论：风险低，勿重做）

- 语法层扫 1366 个文件：C++20 新关键字当标识符、`u8""` 字面量、C++20 移除的标准库设施
  （`result_of`/`is_pod`/`uncaught_exception`/…）**全部 0 命中**
- `thop` 的 `th_utils`/`th_common`（106 个 TU）+ `flash_mla` 早就是 `CXX_STANDARD 20`
  且带 `-Werror` 常绿 → 仓库核心公共头在 C++20 下已被证明干净
- `-Werror` 只作用于 CXX，且只在 runtime/common/batch_manager/executor/thop 五个目录，
  CUDA 编译一行都不带 → 几千个 `.cu` 提标准后即使冒新弃用警告也不会 fail
- `[=]` 全仓 27 处，落在 `-Werror` 目录的仅 4 处（都在 batch_manager），逐个看过都没隐式捕获 this
- torch 头的传递闭包 = 159/697 个 host TU，其中 105 个在 thop（已 20）→ 真正新增的只有 ~54 个
- `-Werror` 目录里的 `operator==` 全是 `bool operator==(Same const&) const` 成员形式，
  不会触发 C++20 反转候选歧义

---

## 6. 最终验证状态

用 CI 的 Debug/`90-real` 配置（逐字取自 Jenkins 日志）实跑：

```bash
python3 scripts/build_wheel.py --use_ccache -G Ninja -j 16 \
    -D "WARNING_IS_ERROR=ON" -a "90-real" -b Debug \
    --micro_benchmarks --extra-cmake-vars NVRTC_DYNAMIC_LINKING=ON
```

结果：**`BUILD_WHEEL_EXIT=0`，成功产出 `tensorrt_llm-1.3.0rc26-cp312-cp312-linux_x86_64.whl`**，
OOM 计数 0，问题 1~4 全部验证修复。

**尚未验证**：
- 只跑了 `90-real` 单 arch + Debug。其他 arch（sm100/103/120）的 `#if` 分支、
  以及 CI 的另外三个配置（Release 全 arch、clang、`ENABLE_MULTI_DEVICE=0`）未覆盖
- SBSA（aarch64）完全未跑
- 运行时行为未测（只验证了编译 + fp8 转换的 SASS 等价性）

## 7. 复现环境备忘

在 dlcluster 上复跑 CI build 的配方见团队记忆 / `reference_dlcluster_build_repro`。
要点：
- `/tmp` 是 job-private（Slurm `job_container/tmpfs`），**作业一结束连同 19G 镜像和日志一起销毁**，
  `#SBATCH -o` 必须写 home
- home 只有 5G 且长期满，只能写摘要
- 汇总错误时必须排除 warning，否则 12572 条 enum 警告会把真正的 error 挤出 top N
- `ninja -k 0` 一次扫到底，避免每轮只暴露一批
- `build_wheel.py` 的参数风格混用：`--extra-cmake-vars` / `--extra-make-targets` 是连字符，
  `--skip_building_wheel` 是下划线
