#include "operator_base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"

/* ========== FP32 <-> FP16 批量转换（ARM NEON） ========== */
static void float32_to_float16(const float* src, uint16_t* dst, int count) {
    for (int i = 0; i < count; i += 8) {
        float32x4_t f32_0 = vld1q_f32(src + i);
        float32x4_t f32_1 = vld1q_f32(src + i + 4);
        float16x4_t f16_0 = vcvt_f16_f32(f32_0);
        float16x4_t f16_1 = vcvt_f16_f32(f32_1);
        vst1_u16(dst + i, vreinterpret_u16_f16(f16_0));
        vst1_u16(dst + i + 4, vreinterpret_u16_f16(f16_1));
    }
}

static void float16_to_float32(const uint16_t* src, float* dst, int count) {
    for (int i = 0; i < count; i += 8) {
        uint16x4_t f16_0 = vld1_u16(src + i);
        uint16x4_t f16_1 = vld1_u16(src + i + 4);
        float32x4_t f32_0 = vcvt_f32_f16(vreinterpret_f16_u16(f16_0));
        float32x4_t f32_1 = vcvt_f32_f16(vreinterpret_f16_u16(f16_1));
        vst1q_f32(dst + i, f32_0);
        vst1q_f32(dst + i + 4, f32_1);
    }
}

/* ========== 算子初始化 ========== */
void rope_init(OperatorTest* op) {
    op->input_dims[0] = 1; op->input_dims[1] = 128;
    op->input_dims[2] = 12; op->input_dims[3] = 64;
    op->input_size = 1 * 128 * 12 * 64 * sizeof(float);
    memcpy(op->output_dims, op->input_dims, sizeof(op->input_dims));
    op->output_size = op->input_size;
    op->input_data = (float*)malloc(op->input_size);
    op->weight_data = NULL;
    op->cpu_output = (float*)malloc(op->output_size);
    op->npu_output = (float*)malloc(op->output_size);
    unsigned int seed = (unsigned int)time(NULL);
    op->seed = seed; srand(seed);
    int elem = op->input_size / sizeof(float);
    for (int i = 0; i < elem; i++)
        op->input_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

/* ========== 生成 cos/sin 频率表（FP32） ========== */
static void generate_cos_sin(int seq_len, int head_dim, float* cos, float* sin) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float angle = pos / pow(10000.0, 2.0 * i / head_dim);
            cos[pos * head_dim + i] = cosf(angle);
            cos[pos * head_dim + head_dim/2 + i] = cosf(angle);
            sin[pos * head_dim + i] = sinf(angle);
            sin[pos * head_dim + head_dim/2 + i] = sinf(angle);
        }
    }
}

/* ========== CPU 计算（使用 float16 模拟 NPU 精度） ========== */
void rope_run_cpu(OperatorTest* op) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int seq_len = 128, num_heads = 12, head_dim = 64;
    int M = seq_len * num_heads;      // 1536 行
    int K = head_dim;                 // 64 列
    int half = K / 2;                 // 半宽 32

    // 1. 生成 FP32 的 cos/sin 表
    float *cos32 = (float*)malloc(seq_len * K * sizeof(float));
    float *sin32 = (float*)malloc(seq_len * K * sizeof(float));
    generate_cos_sin(seq_len, K, cos32, sin32);

    // 2. 转为 FP16 存放
    uint16_t *cos16_base = (uint16_t*)malloc(seq_len * K * sizeof(uint16_t));
    uint16_t *sin16_base = (uint16_t*)malloc(seq_len * K * sizeof(uint16_t));
    float32_to_float16(cos32, cos16_base, seq_len * K);
    float32_to_float16(sin32, sin16_base, seq_len * K);
    free(cos32); free(sin32);

    // 3. 广播到每个 head（FP16）
    uint16_t *cos16 = (uint16_t*)malloc(M * K * sizeof(uint16_t));
    uint16_t *sin16 = (uint16_t*)malloc(M * K * sizeof(uint16_t));
    for (int s = 0; s < seq_len; s++)
        for (int h = 0; h < num_heads; h++) {
            memcpy(cos16 + (s * num_heads + h) * K, cos16_base + s * K, K * sizeof(uint16_t));
            memcpy(sin16 + (s * num_heads + h) * K, sin16_base + s * K, K * sizeof(uint16_t));
        }
    free(cos16_base); free(sin16_base);

    // 4. 输入数据也转为 FP16
    uint16_t *x16 = (uint16_t*)malloc(M * K * sizeof(uint16_t));
    float32_to_float16(op->input_data, x16, M * K);

    // 5. 输出缓冲区（FP16）
    uint16_t *out16 = (uint16_t*)malloc(M * K * sizeof(uint16_t));

    // 6. 使用 __fp16 指针进行向量化计算
    __fp16 *x_ptr = (__fp16*)x16;
    __fp16 *c_ptr = (__fp16*)cos16;
    __fp16 *s_ptr = (__fp16*)sin16;
    __fp16 *o_ptr = (__fp16*)out16;

    for (int m = 0; m < M; m++) {
        int base = m * K;
        __fp16 *x1 = x_ptr + base;
        __fp16 *x2 = x_ptr + base + half;
        __fp16 *c1 = c_ptr + base;
        __fp16 *c2 = c_ptr + base + half;
        __fp16 *s1 = s_ptr + base;
        __fp16 *s2 = s_ptr + base + half;
        __fp16 *o1 = o_ptr + base;
        __fp16 *o2 = o_ptr + base + half;

        // RoPE 旋转公式（全部在 float16 下完成）
        for (int i = 0; i < half; i++) {
            o1[i] = x1[i] * c1[i] - x2[i] * s1[i];
            o2[i] = x2[i] * c2[i] + x1[i] * s2[i];
        }
    }

    // 7. 输出转回 FP32
    float16_to_float32(out16, op->cpu_output, M * K);

    // 释放临时内存
    free(x16);
    free(cos16);
    free(sin16);
    free(out16);

    clock_gettime(CLOCK_MONOTONIC, &end);
    op->cpu_time_us = (end.tv_sec - start.tv_sec) * 1e6 +
                      (end.tv_nsec - start.tv_nsec) / 1e3;
}

/* ========== NPU 推理（加载模型，与原版一致） ========== */
int rope_run_npu(OperatorTest* op, rknn_context ctx) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    static rknn_context rope_ctx = 0;
    if (rope_ctx == 0) {
        int ret = rknn_init(&rope_ctx, (void*)"./models/RoPE.rknn", 0, 0, NULL);
        if (ret != 0) {
            printf("❌ RoPE: load model failed, error %d\n", ret);
            clock_gettime(CLOCK_MONOTONIC, &end);
            op->npu_time_us = (end.tv_sec - start.tv_sec) * 1e6 +
                              (end.tv_nsec - start.tv_nsec) / 1e3;
            return ret;
        }
    }

    rknn_input_output_num io_num;
    int ret = rknn_query(rope_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr input_attrs[3], output_attr;
    for (int i = 0; i < 3; i++) {
        memset(&input_attrs[i], 0, sizeof(rknn_tensor_attr));
        input_attrs[i].index = i;
        rknn_query(rope_ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
    }
    memset(&output_attr, 0, sizeof(output_attr)); output_attr.index = 0;
    rknn_query(rope_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn_tensor_attr));

    rknn_tensor_mem *xm = rknn_create_mem(rope_ctx, input_attrs[0].size);
    rknn_tensor_mem *cm = rknn_create_mem(rope_ctx, input_attrs[1].size);
    rknn_tensor_mem *sm = rknn_create_mem(rope_ctx, input_attrs[2].size);
    rknn_tensor_mem *om = rknn_create_mem(rope_ctx, output_attr.size);
    if (!xm || !cm || !sm || !om) {
        printf("❌ RoPE: memory alloc failed\n");
        ret = -1; goto cleanup;
    }

    int seq_len = 128, num_heads = 12, head_dim = 64;
    int M = seq_len * num_heads, K = head_dim;

    float *cb = malloc(seq_len * K * sizeof(float));
    float *sb = malloc(seq_len * K * sizeof(float));
    generate_cos_sin(seq_len, K, cb, sb);

    float *cf = malloc(M * K * sizeof(float));
    float *sf = malloc(M * K * sizeof(float));
    for (int s = 0; s < seq_len; s++)
        for (int h = 0; h < num_heads; h++) {
            memcpy(cf + (s * num_heads + h) * K, cb + s * K, K * sizeof(float));
            memcpy(sf + (s * num_heads + h) * K, sb + s * K, K * sizeof(float));
        }
    free(cb); free(sb);

    uint16_t *cf16 = malloc(input_attrs[1].size);
    uint16_t *sf16 = malloc(input_attrs[2].size);
    float32_to_float16(cf, cf16, M * K);
    float32_to_float16(sf, sf16, M * K);
    memcpy(cm->virt_addr, cf16, input_attrs[1].size);
    memcpy(sm->virt_addr, sf16, input_attrs[2].size);
    free(cf); free(sf); free(cf16); free(sf16);

    uint16_t *xf = malloc(input_attrs[0].size);
    float32_to_float16(op->input_data, xf, M * K);
    memcpy(xm->virt_addr, xf, input_attrs[0].size);
    free(xf);

    ret = rknn_set_io_mem(rope_ctx, xm, &input_attrs[0]); if (ret) goto cleanup;
    ret = rknn_set_io_mem(rope_ctx, cm, &input_attrs[1]); if (ret) goto cleanup;
    ret = rknn_set_io_mem(rope_ctx, sm, &input_attrs[2]); if (ret) goto cleanup;
    ret = rknn_set_io_mem(rope_ctx, om, &output_attr); if (ret) goto cleanup;
    ret = rknn_run(rope_ctx, NULL); if (ret) goto cleanup;
    ret = rknn_mem_sync(rope_ctx, om, RKNN_MEMORY_SYNC_FROM_DEVICE); if (ret) goto cleanup;

    int total_float32 = output_attr.size / sizeof(uint16_t);
    float* npu_raw = malloc(total_float32 * sizeof(float));
    uint16_t* out_half = (uint16_t*)om->virt_addr;
    float16_to_float32(out_half, npu_raw, total_float32);
    memcpy(op->npu_output, npu_raw, total_float32 * sizeof(float));
    free(npu_raw);

cleanup:
    if (xm) rknn_destroy_mem(rope_ctx, xm);
    if (cm) rknn_destroy_mem(rope_ctx, cm);
    if (sm) rknn_destroy_mem(rope_ctx, sm);
    if (om) rknn_destroy_mem(rope_ctx, om);

    clock_gettime(CLOCK_MONOTONIC, &end);
    op->npu_time_us = (end.tv_sec - start.tv_sec) * 1e6 +
                      (end.tv_nsec - start.tv_nsec) / 1e3;
    return ret;
}

/* ========== 注册算子 ========== */
void register_rope_operator() {
    OperatorInterface iface = {
        .init    = rope_init,
        .run_cpu = rope_run_cpu,
        .run_npu = rope_run_npu
    };
    register_operator("RoPE", iface);
}
