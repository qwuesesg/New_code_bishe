#include "operator_base.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>

char g_media_dir[512] = {0};
char g_report_content[8192] = {0};

typedef struct OperatorNode {
    char name[32];
    OperatorTest test;
    OperatorInterface iface;
    struct OperatorNode* next;
} OperatorNode;

static OperatorNode* g_op_list = NULL;
static int g_op_count = 0;

void register_operator(const char* name, OperatorInterface iface) {
    OperatorNode* node = (OperatorNode*)malloc(sizeof(OperatorNode));
    strncpy(node->name, name, sizeof(node->name)-1);
    node->name[sizeof(node->name)-1] = '\0';
    node->iface = iface;

    strncpy(node->test.name, name, sizeof(node->test.name)-1);
    node->test.name[sizeof(node->test.name)-1] = '\0';

    node->test.input_data = NULL;
    node->test.weight_data = NULL;
    node->test.cpu_output = NULL;
    node->test.npu_output = NULL;
    node->test.input_size = 0;
    node->test.output_size = 0;
    node->test.relative_error = 0.0f;
    node->test.passed = false;
    node->test.seed = 0;
    node->test.cpu_time_us = 0.0;
    node->test.npu_time_us = 0.0;

    node->next = g_op_list;
    g_op_list = node;
    g_op_count++;
}

int get_operator_count() { return g_op_count; }

const char* get_operator_name(int idx) {
    if (idx < 0 || idx >= g_op_count) return NULL;
    OperatorNode* node = g_op_list;
    for (int i = 0; i < idx; i++) node = node->next;
    return node->name;
}

OperatorTest* get_operator_test(int idx) {
    if (idx < 0 || idx >= g_op_count) return NULL;
    OperatorNode* node = g_op_list;
    for (int i = 0; i < idx; i++) node = node->next;
    return &node->test;
}

OperatorInterface* get_operator_interface(int idx) {
    if (idx < 0 || idx >= g_op_count) return NULL;
    OperatorNode* node = g_op_list;
    for (int i = 0; i < idx; i++) node = node->next;
    return &node->iface;
}

void free_op_registry() {
    OperatorNode* node = g_op_list;
    while (node) {
        OperatorNode* temp = node;
        node = node->next;
        free(temp);
    }
    g_op_list = NULL;
    g_op_count = 0;
}

float calc_relative_error(float* cpu_out, float* npu_out, int size) {
    float total_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float cpu_val = cpu_out[i];
        float npu_val = npu_out[i];
        float abs_err = fabs(cpu_val - npu_val);
        float denom = (fabs(cpu_val) < 1e-6) ? 1e-6 : fabs(cpu_val);
        total_err += (abs_err / denom) * 100.0f;
    }
    return total_err / size;
}

int save_float_data_to_media(const char* filename, float* data, int size) {
    if (!filename || !data || size <= 0) return -1;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", g_media_dir, filename);
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        printf("❌ 保存%s失败：无法打开文件（路径：%s）\n", filename, path);
        return -1;
    }
    fwrite(data, sizeof(float), size, fp);
    fclose(fp);
    return 0;
}

int save_test_report_to_media(const char* report_content) {
    if (!report_content) return -1;
    char path[1024];
    snprintf(path, sizeof(path), "%s/test_report.log", g_media_dir);
    FILE* fp = fopen(path, "w");
    if (!fp) {
        printf("❌ 保存报告失败：无法打开文件（路径：%s）\n", path);
        return -1;
    }
    fprintf(fp, "%s", report_content);
    fclose(fp);
    return 0;
}
