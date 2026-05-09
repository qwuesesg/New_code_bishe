/**
 * @file operator_silu.c
 * @brief SiLU (Swish) 激活函数: x * sigmoid(x)
 */

#include "operator_base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"

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

// ==================== SiLU 初始化 ====================
void silu_init(OperatorTest* op) {
    op->input_dims[0] = 1;
    op->input_dims[1] = 12;
    op->input_dims[2] = 64;
    op->input_dims[3] = 0;
    op->input_size = 1 * 12 * 64 * sizeof(float);
    memcpy(op->output_dims, op->input_dims, sizeof(op->input_dims));
    op->output_size = op->input_size;

    op->input_data  = (float*)malloc(op->input_size);
    op->weight_data = NULL;
    op->cpu_output  = (float*)malloc(op->output_size);
    op->npu_output  = (float*)malloc(op->output_size);

    unsigned int seed = (unsigned int)time(NULL);
    op->seed = seed;
    srand(seed);

    int elem = op->input_size / sizeof(float);
    for (int i = 0; i < elem; i++) {
        op->input_data[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;
    }
}

// ==================== CPU 参考计算 ====================
void silu_run_cpu(OperatorTest* op) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int elem = op->input_size / sizeof(float);
    for (int i = 0; i < elem; i++) {
        float x = op->input_data[i];
        op->cpu_output[i] = x / (1.0f + expf(-x));
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    op->cpu_time_us = (end.tv_sec - start.tv_sec) * 1e6 +
                      (end.tv_nsec - start.tv_nsec) / 1e3;
}

// ==================== NPU 推理实现 ====================
int silu_run_npu(OperatorTest* op, rknn_context ctx) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    static rknn_context silu_ctx = 0;
    if (silu_ctx == 0) {
        const char* model_path = "./models/SiLU.rknn";
        int ret = rknn_init(&silu_ctx, (void*)model_path, 0, 0, NULL);
        if (ret != 0) {
            printf("❌ SiLU: 加载模型失败，错误码：%d\n", ret);
            clock_gettime(CLOCK_MONOTONIC, &end);
            op->npu_time_us = (end.tv_sec - start.tv_sec) * 1e6 +
                              (end.tv_nsec - start.tv_nsec) / 1e3;
            return ret;
        }
    }

    rknn_input_output_num io_num;
    int ret = rknn_query(silu_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret) goto final;
    if (io_num.n_input != 1 || io_num.n_output != 1) { ret = -1; goto final; }

    rknn_tensor_attr input_attr, output_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    ret = rknn_query(silu_ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(rknn_tensor_attr));
    if (ret) goto final;
    memset(&output_attr, 0, sizeof(output_attr));
    output_attr.index = 0;
    ret = rknn_query(silu_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(rknn_tensor_attr));
    if (ret) goto final;

    size_t model_in_size = input_attr.size;
    size_t model_out_size = output_attr.size;

    rknn_tensor_mem* input_mem = rknn_create_mem(silu_ctx, model_in_size);
    rknn_tensor_mem* output_mem = rknn_create_mem(silu_ctx, model_out_size);
    if (!input_mem || !output_mem) {
        printf("❌ SiLU: NPU内存分配失败\n");
        if (input_mem) rknn_destroy_mem(silu_ctx, input_mem);
        if (output_mem) rknn_destroy_mem(silu_ctx, output_mem);
        ret = -1; goto final;
    }

    if (model_in_size == op->input_size / 2) {
        int elem = op->input_size / sizeof(float);
        uint16_t* temp = (uint16_t*)malloc(model_in_size);
        if (!temp) {
            printf("❌ SiLU: 临时内存分配失败\n");
            rknn_destroy_mem(silu_ctx, input_mem);
            rknn_destroy_mem(silu_ctx, output_mem);
            ret = -1; goto final;
        }
        float32_to_float16(op->input_data, temp, elem);
        memcpy(input_mem->virt_addr, temp, model_in_size);
        free(temp);
    } else if (model_in_size == op->input_size) {
        memcpy(input_mem->virt_addr, op->input_data, model_in_size);
    } else {
        printf("⚠️ SiLU: 模型输入大小不匹配\n");
        rknn_destroy_mem(silu_ctx, input_mem);
        rknn_destroy_mem(silu_ctx, output_mem);
        ret = -1; goto final;
    }

    ret = rknn_set_io_mem(silu_ctx, input_mem, &input_attr);
    if (ret) { printf("❌ SiLU: 绑定输入失败\n"); goto cleanup; }
    ret = rknn_set_io_mem(silu_ctx, output_mem, &output_attr);
    if (ret) { printf("❌ SiLU: 绑定输出失败\n"); goto cleanup; }
    ret = rknn_run(silu_ctx, NULL);
    if (ret) { printf("❌ SiLU: NPU执行失败\n"); goto cleanup; }
    ret = rknn_mem_sync(silu_ctx, output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (ret) { printf("❌ SiLU: 内存同步失败\n"); goto cleanup; }

    if (output_attr.type == RKNN_TENSOR_FLOAT16) {
        int elem_count = model_out_size / sizeof(uint16_t);
        uint16_t* out_half = (uint16_t*)output_mem->virt_addr;
        float16_to_float32(out_half, op->npu_output, elem_count);
    } else {
        memcpy(op->npu_output, output_mem->virt_addr, model_out_size);
    }

cleanup:
    rknn_destroy_mem(silu_ctx, input_mem);
    rknn_destroy_mem(silu_ctx, output_mem);
final:
    clock_gettime(CLOCK_MONOTONIC, &end);
    op->npu_time_us = (end.tv_sec - start.tv_sec) * 1e6 +
                      (end.tv_nsec - start.tv_nsec) / 1e3;
    return ret;
}

void register_silu_operator() {
    OperatorInterface iface = {
        .init    = silu_init,
        .run_cpu = silu_run_cpu,
        .run_npu = silu_run_npu
    };
    register_operator("SiLU", iface);
}
