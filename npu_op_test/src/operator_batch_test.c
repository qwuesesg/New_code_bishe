#include "operator_base.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

extern char g_media_dir[512];
extern char g_report_content[8192];

void register_add_operator();
void register_conv2d_operator();
void register_matmul_operator();
void register_relu_operator();
void register_softmax_operator();
void register_silu_operator();
void register_layernorm_operator();
void register_rope_operator();

void register_all_operators() {
    register_add_operator();
    register_conv2d_operator();
    register_matmul_operator();
    register_relu_operator();
    register_softmax_operator();   
    register_silu_operator();      
    register_layernorm_operator(); 
    register_rope_operator();
}

int main() {
    const char* env_media = getenv("MEDIA_DIR");
    if (env_media) {
        snprintf(g_media_dir, sizeof(g_media_dir), "%s", env_media);
    } else {
        snprintf(g_media_dir, sizeof(g_media_dir), "../media/");
    }
    char mkdir_cmd[1024];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", g_media_dir);
    system(mkdir_cmd);
    printf("🔹 创建media目录完成：%s\n", g_media_dir);

    register_all_operators();
    printf("✅ 共注册 %d 个算子：", get_operator_count());
    for (int i = 0; i < get_operator_count(); i++) {
        printf(" %s", get_operator_name(i));
    }
    printf("\n\n");

    rknn_context ctx_main;
    const char* rknn_model_path = "/root/npu_test/min_model.rknn";
    int ret = rknn_init(&ctx_main, (void*)rknn_model_path, 0, 0, NULL);
    if (ret != 0) {
        printf("❌ NPU初始化失败，错误码：%d\n", ret);
        free_op_registry();
        return -1;
    }
    printf("✅ NPU主上下文初始化成功\n\n");

    int op_count = get_operator_count();
    int pass_count = 0;
    strcat(g_report_content, "📋 RK3588 NPU算子批量测试报告\n");
    strcat(g_report_content, "=====================================\n");
    time_t now = time(NULL);
    strcat(g_report_content, "测试时间：");
    strcat(g_report_content, ctime(&now));
    strcat(g_report_content, "=====================================\n");

    printf("%-10s %10s %10s %10s %8s %8s\n", 
           "算子", "误差(%)", "CPU(ms)", "NPU(ms)", "加速比", "状态");
    printf("-------------------------------------------------------------\n");

    for (int i = 0; i < op_count; i++) {
        const char* op_name = get_operator_name(i);
        OperatorTest* op = get_operator_test(i);
        OperatorInterface* iface = get_operator_interface(i);
        rknn_context ctx;

        if (strcmp(op_name, "MatMul") == 0) {
            ctx = ctx_main;
        } else {
            char model_path[256];
            snprintf(model_path, sizeof(model_path), "./models/%s.rknn", op_name);
            ret = rknn_init(&ctx, model_path, 0, 0, NULL);
            if (ret != 0) {
                printf("%-10s %10s %10s %10s %8s %8s\n", op_name, "-", "-", "-", "-", "模型失败");
                // 报告中也记录模型加载失败
                char fail_str[128];
                snprintf(fail_str, sizeof(fail_str), "\n【算子：%s】模型加载失败\n", op_name);
                strcat(g_report_content, fail_str);
                continue;
            }
        }

        iface->init(op);
        printf("  ✅ 初始化完成（%s，种子：%u）\n", op_name, op->seed);

        // ===== 新增：报告记录维度、种子、随机数范围 =====
        char dim_str[256];
        snprintf(dim_str, sizeof(dim_str),
                 "\n【算子：%s】\n"
                 "  输入维度：%d x %d x %d x %d，种子：%u\n"
                 "  随机数范围：[-1, 1]\n",
                 op_name,
                 op->input_dims[0], op->input_dims[1], op->input_dims[2], op->input_dims[3],
                 op->seed);
        strcat(g_report_content, dim_str);
        strcat(g_report_content, "  CPU计算：完成\n");

        iface->run_cpu(op);
        ret = iface->run_npu(op, ctx);

        if (ret != 0) {
            // NPU 失败时仍输出表格和记录
            printf("%-10s %10s %10.3f %10.3f %8s %8s\n", 
                   op_name, "N/A", 
                   op->cpu_time_us/1000.0, op->npu_time_us/1000.0, 
                   "-", "失败");
            strcat(g_report_content, "  NPU执行：失败\n");
            if (strcmp(op_name, "MatMul") != 0) rknn_destroy(ctx);
            continue;
        }

        int data_size = op->output_size / sizeof(float);
        float err = calc_relative_error(op->cpu_output, op->npu_output, data_size);
        op->relative_error = err;

        // ===== 新增：前三个元素对比 =====
        char cmp_str[512];
        snprintf(cmp_str, sizeof(cmp_str),
                 "  前三个元素对比：\n"
                 "    CPU: %.6f  %.6f  %.6f\n"
                 "    NPU: %.6f  %.6f  %.6f\n",
                 op->cpu_output[0], op->cpu_output[1], op->cpu_output[2],
                 op->npu_output[0], op->npu_output[1], op->npu_output[2]);
        strcat(g_report_content, cmp_str);
        printf("  %s", cmp_str);   // 控制台显示

        // ===== 新增：写入误差到报告 =====
        char err_str[64];
        snprintf(err_str, sizeof(err_str), "  平均相对误差：%.4f%%\n", err);
        strcat(g_report_content, err_str);

        double cpu_ms = op->cpu_time_us / 1000.0;
        double npu_ms = op->npu_time_us / 1000.0;
        double speedup = (npu_ms > 0) ? cpu_ms / npu_ms : 0.0;

        printf("%-10s %10.4f %10.3f %10.3f %8.2f %8s\n",
               op_name, err,
               cpu_ms, npu_ms, speedup,
               err < 1.0f ? "通过" : "超标");

        // ===== 新增：状态写入报告 =====
        if (err < 1.0f) {
            op->passed = true;
            pass_count++;
            strcat(g_report_content, "  状态：通过\n");
        } else {
            op->passed = false;
            strcat(g_report_content, "  状态：失败（误差超标）\n");
        }
        // 记录 NPU 完成状态
        strcat(g_report_content, "  NPU执行：完成\n");

        char filename[128];
        snprintf(filename, sizeof(filename), "%s_input.bin", op_name);
        save_float_data_to_media(filename, op->input_data, op->input_size / sizeof(float));

        if (op->weight_data) {
            snprintf(filename, sizeof(filename), "%s_weight.bin", op_name);
            int weight_elem = (op_name[0] == 'C') ? (16 * 3 * 3 * 3) : (op->input_size / sizeof(float));
            save_float_data_to_media(filename, op->weight_data, weight_elem);
        }
        snprintf(filename, sizeof(filename), "%s_cpu_output.bin", op_name);
        save_float_data_to_media(filename, op->cpu_output, data_size);
        snprintf(filename, sizeof(filename), "%s_npu_output.bin", op_name);
        save_float_data_to_media(filename, op->npu_output, data_size);
        snprintf(filename, sizeof(filename), "%s_seed.txt", op_name);
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", g_media_dir, filename);
        FILE* fp = fopen(path, "w");
        if (fp) { fprintf(fp, "%u\n", op->seed); fclose(fp); }

        if (strcmp(op_name, "MatMul") != 0) {
            rknn_destroy(ctx);
        }
    }
    printf("-------------------------------------------------------------\n");

    char summary[256];
    snprintf(summary, sizeof(summary), "📊 共 %d 个算子，通过 %d 个，失败 %d 个\n",
             op_count, pass_count, op_count - pass_count);
    printf("%s", summary);
    strcat(g_report_content, summary);
    save_test_report_to_media(g_report_content);

    for (int i = 0; i < op_count; i++) {
        OperatorTest* op = get_operator_test(i);
        if (op->input_data) free(op->input_data);
        if (op->weight_data) free(op->weight_data);
        if (op->cpu_output) free(op->cpu_output);
        if (op->npu_output) free(op->npu_output);
    }
    rknn_destroy(ctx_main);
    free_op_registry();
    return 0;
}
