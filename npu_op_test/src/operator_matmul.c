#include "operator_base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <arm_neon.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"

void matmul_init(OperatorTest* op) {
    op->input_dims[0] = 1; op->input_dims[1] = 64; op->input_dims[2] = 1; op->input_dims[3] = 128;
    op->input_size = 1 * 64 * 1 * 128 * sizeof(float);

    int K = 128, N = 64;
    op->output_dims[0] = 1; op->output_dims[1] = 64; op->output_dims[2] = 1; op->output_dims[3] = N;
    op->output_size = 1 * 64 * 1 * N * sizeof(float);

    int weight_size = K * N * sizeof(float);
    op->weight_data = (float*)malloc(weight_size);
    op->input_data  = (float*)malloc(op->input_size);
    op->cpu_output  = (float*)malloc(op->output_size);
    op->npu_output  = (float*)malloc(op->output_size);

    unsigned int seed = 42; op->seed = seed; srand(seed);
    for (int i = 0; i < op->input_size / sizeof(float); i++)
        op->input_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < weight_size / sizeof(float); i++)
        op->weight_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

void matmul_run_cpu(OperatorTest* op) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int M = 64, K = 128, N = 64;
    memset(op->cpu_output, 0, op->output_size);
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++) {
            float a_val = op->input_data[m*K + k];
            if (a_val == 0.0f) continue;
            for (int n = 0; n < N; n++)
                op->cpu_output[m*N + n] += a_val * op->weight_data[k*N + n];
        }
    clock_gettime(CLOCK_MONOTONIC, &end);
    op->cpu_time_us = (end.tv_sec - start.tv_sec) * 1e6 +
                      (end.tv_nsec - start.tv_nsec) / 1e3;
}

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

int rknn_matmul_forward(rknn_context ctx, const float* A, const float* B, float* C, int M, int K, int N) {
    rknn_matmul_ctx matmul_ctx;
    rknn_matmul_info info = {0};
    rknn_matmul_io_attr io_attr = {0};
    info.M = M; info.K = K; info.N = N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = RKNN_MM_LAYOUT_NORM;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;

    int ret = rknn_matmul_create(&matmul_ctx, &info, &io_attr);
    if (ret) return ret;

    int A_elem = M * K, B_elem = K * N, C_elem = M * N;
    size_t A_sz = A_elem * sizeof(uint16_t);
    size_t B_sz = B_elem * sizeof(uint16_t);
    size_t C_sz = C_elem * sizeof(float);

    rknn_tensor_mem* Am = rknn_create_mem(ctx, A_sz);
    rknn_tensor_mem* Bm = rknn_create_mem(ctx, B_sz);
    rknn_tensor_mem* Cm = rknn_create_mem(ctx, C_sz);
    if (!Am || !Bm || !Cm) {
        if (Am) rknn_destroy_mem(ctx, Am);
        if (Bm) rknn_destroy_mem(ctx, Bm);
        if (Cm) rknn_destroy_mem(ctx, Cm);
        rknn_matmul_destroy(matmul_ctx);
        return -1;
    }

    uint16_t* Af = (uint16_t*)malloc(A_sz);
    uint16_t* Bf = (uint16_t*)malloc(B_sz);
    float32_to_float16(A, Af, A_elem);
    float32_to_float16(B, Bf, B_elem);
    memcpy(Am->virt_addr, Af, A_sz);
    memcpy(Bm->virt_addr, Bf, B_sz);
    free(Af); free(Bf);

    ret = rknn_matmul_set_io_mem(matmul_ctx, Am, &io_attr.A);
    if (!ret) ret = rknn_matmul_set_io_mem(matmul_ctx, Bm, &io_attr.B);
    if (!ret) ret = rknn_matmul_set_io_mem(matmul_ctx, Cm, &io_attr.C);
    if (!ret) ret = rknn_matmul_run(matmul_ctx);
    if (!ret) ret = rknn_mem_sync(ctx, Cm, RKNN_MEMORY_SYNC_FROM_DEVICE);
    if (!ret) memcpy(C, Cm->virt_addr, C_sz);

    rknn_destroy_mem(ctx, Am);
    rknn_destroy_mem(ctx, Bm);
    rknn_destroy_mem(ctx, Cm);
    rknn_matmul_destroy(matmul_ctx);
    return ret;
}

int matmul_run_npu(OperatorTest* op, rknn_context ctx) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int ret = rknn_matmul_forward(ctx, op->input_data, op->weight_data, op->npu_output, 64, 128, 64);
    clock_gettime(CLOCK_MONOTONIC, &end);
    op->npu_time_us = (end.tv_sec - start.tv_sec) * 1e6 +
                      (end.tv_nsec - start.tv_nsec) / 1e3;
    if (ret) printf("❌ MatMul：NPU执行失败\n");
    return ret;
}

void register_matmul_operator() {
    OperatorInterface iface = { .init=matmul_init, .run_cpu=matmul_run_cpu, .run_npu=matmul_run_npu };
    register_operator("MatMul", iface);
}
