<!--
 * @Author: jielong.wang jielong.wang@akuvox.com
 * @Date: 2025-09-19 16:12:24
 * @LastEditors: jielong.wang jielong.wang@akuvox.com
 * @LastEditTime: 2025-09-19 16:14:23
 * @FilePath: /rknn-llm/examples/Qwen2.5-VL-instruct-3b/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Qwen2.5-VL-Instruct-3B-RKNN

参考Qwen2-VL_Demo，将Qwen2.5-VL-instruct-3b模型转换为RKNN格式，并进行推理。
注意点：导出视觉编码器onnx时，opset<18，不然转rknn会报错。