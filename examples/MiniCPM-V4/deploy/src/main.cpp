// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>

#include "image_enc.h"
#include "rkllm.h"

#define IMAGE_HEIGHT 448
#define IMAGE_WIDTH 448
#define IMAGE_TOKEN_NUM 64
#define EMBED_SIZE 2560

using namespace std;
LLMHandle llmHandle = nullptr;

void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        {
            cout << "程序即将退出" << endl;
            LLMHandle _tmp = llmHandle;
            llmHandle = nullptr;
            rkllm_destroy(_tmp);
        }
    }
    exit(signal);
}

int callback(RKLLMResult *result, void *userdata, LLMCallState state)
{

    if (state == RKLLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        printf("\\run error\n");
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        printf("%s", result->text);
        // for(int i=0; i<result->num; i++)
        // {
        //     printf("%d token_id: %d logprob: %f\n", i, result->tokens[i].id, result->tokens[i].logprob);
        // }
    }
    return 0;
}

// Expand the image into a square and fill it with the specified background color
cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color) {
    int width = img.cols;
    int height = img.rows;

    // If the width and height are equal, return to the original image directly
    if (width == height) {
        return img.clone();
    }

    // Calculate the new size and create a new image
    int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background_color);

    // Calculate the image paste position
    int x_offset = (size - width) / 2;
    int y_offset = (size - height) / 2;

    // Paste the original image into the center of the new image
    cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));

    return result;
}

int main(int argc, char** argv)
{
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " image_path encoder_model_path llm_model_path max_new_tokens max_context_len rknn_core_num\n";
        return -1;
    }

    const char * image_path = argv[1];
    const char * encoder_model_path = argv[2];

    //设置llm参数及初始化
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[3];
    param.top_k = 1;
    param.max_new_tokens = std::atoi(argv[4]);
    param.max_context_len = std::atoi(argv[5]);
    param.skip_special_token = true;
    param.img_start = "<image>";
    param.img_end = "</image>";
    param.img_content = "<unk>";
    param.extend_param.base_domain_id = 1;
    int ret;

    // imgenc初始化
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    std::chrono::high_resolution_clock::time_point t_start_us = std::chrono::high_resolution_clock::now();

    const int core_num = atoi(argv[6]);
    ret = init_imgenc(encoder_model_path, &rknn_app_ctx, core_num);
    if (ret != 0) {
        printf("init_imgenc fail! ret=%d model_path=%s\n", ret, encoder_model_path);
        return -1;
    }
    std::chrono::high_resolution_clock::time_point t_load_end_us = std::chrono::high_resolution_clock::now();

    t_start_us = std::chrono::high_resolution_clock::now();

    ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }
    t_load_end_us = std::chrono::high_resolution_clock::now();

    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: LLM Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

    

    

    load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: ImgEnc Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

    // The image is read in BGR format
    cv::Mat img = cv::imread(image_path);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    t_start_us = std::chrono::high_resolution_clock::now();
    // Expand the image into a square and fill it with the specified background color (According the modeling_minicpmv.py)
    cv::Scalar background_color(127.5, 127.5, 127.5);
    cv::Mat square_img = expand2square(img, background_color);

    // Resize the image to 392x392
    cv::Mat resized_img;
    cv::Size new_size(IMAGE_WIDTH, IMAGE_HEIGHT);
    cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

    size_t n_image_tokens = IMAGE_TOKEN_NUM;
    size_t image_embed_len = EMBED_SIZE;
    int rkllm_image_embed_len = n_image_tokens * image_embed_len;
    float img_vec[rkllm_image_embed_len];
    ret = run_imgenc(&rknn_app_ctx, resized_img.data, img_vec);
    if (ret != 0) {
        printf("run_imgenc fail! ret=%d\n", ret);
    }
    t_load_end_us = std::chrono::high_resolution_clock::now();
    load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: ImgEnc Model run in %8.2f ms\n", __func__, load_time.count() / 1000.0);
    
    RKLLMInput rkllm_input;

    // 初始化 infer 参数结构体
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    
    rkllm_infer_params.keep_history = 0;
    std::string system_prompt = "<|im_start|>system\n你是安防场景的视频图像分析助手。请进行目标与行为的描述，并输出严格遵循提供的结构化模板，不要输出其他无关内容。若无法判断，请填“否”。\n"
        "# 结构化模板\n"
        "```json\n"
        "{\n"
        "    \"画面描述\": {\n"
        "        \"场景\": \"简要描述图像场景，20字左右\",\n"
        "        \"起火\": \"是/否\",\n"
        "        \"冒烟\": \"是/否\",\n"
        "        \"路面积水\": \"是/否\",\n"
        "        \"场地整洁\": \"是/否\",\n"
        "    },\n"
        "    \"人物描述\": [\n"
        "        \"人物1\": {\n"
        "            \"坐标\": [x1, y1, x2, y2],\n"
        "            \"描述\": \"简要描述人物1的行为、服饰，20字左右\",\n"
        "            \"打电话\": \"是/否\",\n"
        "            \"吸烟\": \"是/否\",\n"
        "            \"佩戴口罩\": \"是/否\",\n"
        "            \"摔倒\": \"是/否\"\n"
        "        }\n"
        "    ]\n"
        "}\n"
        "```\n"
        "# 注意事项\n"
        "1. 目标坐标是图像的绝对坐标，左上角(x1,y1)，右下角(x2,y2)。\n"
        "2. 如果有多个人物，按照同样格式依次描述，放在\"人物描述\"列表中。\n"
        "<|im_end|>\n";

    std::string user_prompt = "你是安防场景的视频图像分析助手。请进行目标与行为的描述，并输出严格遵循提供的结构化模板，不要输出其他无关内容。若无法判断，请填“否”。\n"
        "# 结构化模板\n"
        "```json\n"
        "{\n"
        "    \"画面描述\": {\n"
        "        \"场景\": \"简要描述图像场景，20字左右\",\n"
        "        \"起火\": \"是/否\",\n"
        "        \"冒烟\": \"是/否\",\n"
        "        \"路面积水\": \"是/否\",\n"
        "        \"场地整洁\": \"是/否\",\n"
        "    },\n"
        "    \"人物描述\": [\n"
        "        \"人物1\": {\n"
        "            \"坐标\": [x1, y1, x2, y2],\n"
        "            \"描述\": \"简要描述人物1的行为、服饰，20字左右\",\n"
        "            \"打电话\": \"是/否\",\n"
        "            \"吸烟\": \"是/否\",\n"
        "            \"佩戴口罩\": \"是/否\",\n"
        "            \"摔倒\": \"是/否\"\n"
        "        }\n"
        "    ]\n"
        "}\n"
        "```\n"
        "# 注意事项\n"
        "1. 目标坐标是图像的绝对坐标，左上角(x1,y1)，右下角(x2,y2)。\n"
        "2. 如果有多个人物，按照同样格式依次描述，放在\"人物描述\"列表中。\n";


    // rkllm_set_chat_template(llmHandle, system_prompt.c_str(), "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");
    rkllm_set_chat_template(llmHandle, "", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");


    vector<string> pre_input;
    pre_input.push_back("<image>What is in the image?");
    pre_input.push_back(std::string("<image>") + user_prompt);
    pre_input.push_back("<image>仔细描述这张图");
    pre_input.push_back("<image>Question: The object shown in this figure:\nOptions:\nA. Person falling\nB. Water on the floor\nC. Fire\nD. Smoke\nE. None of them\nNo explanation. Select the correct answer from [A, B, C, D, E] directly. \n");
    cout << "\n**********************可输入以下问题对应序号获取回答/或自定义输入********************\n"
         << endl;
    for (int i = 0; i < (int)pre_input.size(); i++)
    {
        cout << "[" << i << "] " << pre_input[i] << endl;
    }
    cout << "\n*************************************************************************\n"
         << endl;

    while(true) {
        std::string input_str;
        printf("\n");
        printf("user: ");
        std::getline(std::cin, input_str);
        if (input_str == "exit")
        {
            break;
        }
        if (input_str == "clear")
        {
            ret = rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
            if (ret != 0)
            {
                printf("clear kv cache failed!\n");
            }
            continue;
        }
        for (int i = 0; i < (int)pre_input.size(); i++)
        {
            if (input_str == to_string(i))
            {
                input_str = pre_input[i];
                cout << input_str << endl;
            }
        }
        if (input_str.find("<image>") == std::string::npos) 
        {
            std::cout << "===================>LLM run with text input" << std::endl;
            rkllm_input.input_type = RKLLM_INPUT_PROMPT;
            rkllm_input.role = "user";
            rkllm_input.prompt_input = (char*)input_str.c_str();
        } else {
            std::cout << "===================>MLLM run with text input" << std::endl;
            rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
            rkllm_input.role = "user";
            rkllm_input.multimodal_input.prompt = (char*)input_str.c_str();
            rkllm_input.multimodal_input.image_embed = img_vec;
            rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
            rkllm_input.multimodal_input.n_image = 1;
            rkllm_input.multimodal_input.image_height = IMAGE_HEIGHT;
            rkllm_input.multimodal_input.image_width = IMAGE_WIDTH;
        }
        printf("robot: ");
        t_start_us = std::chrono::high_resolution_clock::now();
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
        t_load_end_us = std::chrono::high_resolution_clock::now();
        load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
        printf("%s: llm run in %8.2f ms\n", __func__, load_time.count() / 1000.0);
    }

    ret = release_imgenc(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_imgenc fail! ret=%d\n", ret);
    }
    rkllm_destroy(llmHandle);

    return 0;
}