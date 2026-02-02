#!/usr/bin/env python
"""
增强版动作预测器 - 基于Qwen3-VL-2B-Instruct模型
新增通用聊天功能，支持纯文本或图文对话
"""

import torch
import os
import json
import re
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from peft import PeftModel


class ActionPredictor:
    def __init__(self, model_path=None, adapter_path=None, device=None):
        """
        初始化动作预测器
        :param model_path: 基础模型路径
        :param adapter_path: LoRA适配器路径
        :param device: 设备设置 ('cuda:0', 'cpu'等)
        """
        if model_path is None:
            # 默认使用全量微调模型的checkpoint目录
            self.model_path = '/mnt/workspace/qwen3-vl-2b-instruct-lora_llm/checkpoint-30'
        else:
            self.model_path = model_path

        self.adapter_path = adapter_path

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[INFO] 使用设备: {self.device}")
        self._load_model()

    def _load_model(self):
        """加载模型和处理器"""
        print("[INFO] 正在加载Qwen3-VL-2B-Instruct模型...")

        # 检测模型类型：全量微调模型 或 LoRA适配器
        is_lora = (os.path.exists(os.path.join(self.model_path, "adapter_model.bin")) or
                   os.path.exists(os.path.join(self.model_path, "adapter_model.safetensors"))) and \
                  os.path.exists(os.path.join(self.model_path, "adapter_config.json"))

        is_full = (os.path.exists(os.path.join(self.model_path, "pytorch_model.bin")) or
                   os.path.exists(os.path.join(self.model_path, "model.safetensors"))) and \
                  os.path.exists(os.path.join(self.model_path, "config.json"))

        # 确定处理器路径
        processor_path = self.model_path

        if is_lora or self.adapter_path:
            # LoRA模式：需指定基础模型
            if self.adapter_path:
                # 如果提供了adapter_path，使用它作为LoRA路径
                adapter_dir = self.adapter_path
            else:
                # 否则使用model_path作为LoRA路径
                adapter_dir = self.model_path

            # 从adapter_config.json中获取基础模型名称，如果不存在则使用默认值
            adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)
                base_model = adapter_config.get("base_model_name_or_path", "/root/.cache/modelscope/hub/models/qwen/Qwen3-VL-2B-Instruct")
            else:
                base_model = "/root/.cache/modelscope/hub/models/qwen/Qwen3-VL-2B-Instruct"

            print(f"→ 检测到LoRA模型，加载基础模型: {base_model}")

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )

            # 加载适配器
            self.model = PeftModel.from_pretrained(self.model, adapter_dir, is_trainable=False)
            print(f"✓ LoRA适配器加载成功: {adapter_dir}")

            # 如果是LoRA且目标路径缺少tokenizer配置，从基础模型加载处理器
            if not os.path.exists(os.path.join(self.model_path, "tokenizer_config.json")):
                processor_path = base_model

        elif is_full:
            # 全量模型：直接加载
            print(f"→ 加载全量微调模型: {self.model_path}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )
        else:
            # 如果都不是，尝试作为基础模型加载
            print(f"→ 尝试作为基础模型加载: {self.model_path}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True
            )

        # 加载处理器
        self.processor = Qwen3VLProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True
        )

        self.model.eval()
        print("[INFO] 模型加载完成 ✓")
    
    def _process_image(self, img):
        """统一处理各种格式的图片输入"""
        if isinstance(img, Image.Image):
            return img.convert('RGB')
        
        if isinstance(img, str):
            if img.startswith('data:image'):
                import base64
                from io import BytesIO
                base64_str = img.split(',', 1)[1] if ',' in img else img
                image_bytes = base64.b64decode(base64_str)
                return Image.open(BytesIO(image_bytes)).convert('RGB')
            elif os.path.exists(img):
                return Image.open(img).convert('RGB')
            else:
                raise FileNotFoundError(f"图像文件不存在: {img}")
        
        raise ValueError(f"不支持的图像格式: {type(img)}")
    
    def get_action(self, image1, image2):
        """
        从两张连续图片预测动作
        :param image1: 第一张图片 (路径/PIL/base64)
        :param image2: 第二张图片 (路径/PIL/base64)
        :return: (direction_bool, distance_int) 
                 direction_bool: True="+", False="-"
                 distance_int: 移动距离(整数)
        """
        # 处理图片
        pil_img1 = self._process_image(image1)
        pil_img2 = self._process_image(image2)
        
        # 构建专业提示词
        prompt = """请分析两张连续拍摄的显微镜图像，判断当前聚焦状态的变化趋势，并据此推断电机应向哪个方向移动多少步以接近最佳聚焦位置。 - 如果第二张图像比第一张更模糊，说明焦点正在远离最佳位置，电机应向负方向（"-"）移动； - 如果第二张图像比第一张更清晰，说明焦点正在接近最佳位置，电机应继续向正方向（"+"）移动。 请基于图像清晰度变化，估计电机需移动的步数（取整数），并严格按照以下 JSON 格式返回结果，不要包含任何额外文本或解释： {"analysis": "电机应该向{x}方向移动{y}步。", "direction": "{x}", "distance": {y}} 注意： - "direction" 只能是 "+" 或 "-"； - "distance" 必须是非负整数（如 0, 1, 2, ...）； - "analysis" 中的方向和步数必须与 direction 和 distance 字段一致； - 输出必须是纯 JSON，无 Markdown、无注释、无多余空格或换行。"""
        
        # 构建消息
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"}, 
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }]
        
        return self._extract_action(messages, [pil_img1, pil_img2])
    
    def chat(self, prompt, images=None, max_new_tokens=512, temperature=0.3):
        """
        通用图文对话功能
        :param prompt: 用户文本提示
        :param images: 可选，图片列表（支持0～N张，格式同get_action）
        :param max_new_tokens: 生成最大token数
        :param temperature: 采样温度
        :return: 模型生成的完整文本回复（字符串）
        """
        # 处理图片
        pil_images = []
        content = []

        if images:
            if not isinstance(images, list):
                images = [images]

            for img in images:
                # 使用更健壮的图片处理方式
                pil_img = self._process_image(img)
                pil_images.append(pil_img)
                content.append({"type": "image"})

        # 添加文本提示
        content.append({"type": "text", "text": prompt})

        # 构建消息
        messages = [{"role": "user", "content": content}]

        # 处理输入
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 使用更健壮的输入处理方式
        if pil_images:
            inputs = self.processor(
                text=[text],
                images=pil_images,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            )

        # 确保所有输入张量都在同一设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 生成回复
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        # 提取生成内容（去除输入部分）
        generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        return response
    
    def _extract_action(self, messages, pil_images):
        """内部方法：从模型输出提取动作信息"""
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 使用更健壮的输入处理方式
        inputs = self.processor(
            text=[text],
            images=pil_images,
            padding=True,
            return_tensors="pt"
        )

        # 确保所有输入张量都在同一设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True
        )[0].strip()

        print(f"[DEBUG] 模型原始输出: {output}")

        # 尝试提取JSON
        try:
            # 多种JSON提取策略
            json_str = None
            patterns = [
                r'\{[^{}]*"direction"[^{}]*"distance"[^{}]*\}',
                r'\{.*?\}',
                output.strip()
            ]

            for pattern in patterns:
                if pattern.startswith('{'):
                    json_str = pattern
                    break
                match = re.search(pattern, output, re.DOTALL)
                if match:
                    json_str = match.group()
                    break

            if json_str:
                result = json.loads(json_str)
                direction = str(result.get("direction", "-")).strip()
                distance = result.get("distance", 0)

                # 转换格式
                direction_bool = direction == "+"
                distance_int = int(distance) if isinstance(distance, (int, float)) else 0

                print(f"[SUCCESS] 解析结果: direction={direction_bool} ({direction}), distance={distance_int}")
                return direction_bool, distance_int

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[WARNING] JSON解析失败: {e}")

        # 失败回退策略：正则提取
        print("[INFO] 尝试正则回退解析...")
        dir_match = re.search(r'"direction"\s*:\s*"([^"]+)"', output)
        dist_match = re.search(r'"distance"\s*:\s*(\d+)', output)

        if dir_match and dist_match:
            direction_bool = dir_match.group(1).strip() == "+"
            distance_int = int(dist_match.group(1))
            print(f"[SUCCESS] 正则解析成功: direction={direction_bool}, distance={distance_int}")
            return direction_bool, distance_int

        print("[ERROR] 无法解析动作信息，返回默认值")
        return False, 0
    
    def close(self):
        """释放资源"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        torch.cuda.empty_cache()
        print("[INFO] 资源已释放")


def main():
    """完整功能测试"""
    print("\n" + "="*60)
    print("🚀 增强版动作预测器 - 功能测试")
    print("="*60)

    try:
        # 初始化预测器
        predictor = ActionPredictor()

        # # ========== 测试1: 通用聊天功能（纯文本）==========
        # print("\n" + "-"*60)
        # print("📌 测试1: 通用聊天功能（纯文本提问）")
        # print("-"*60)
        # text_prompt = "显微镜使用操作流程与注意事项"
        # print(f"\n👤 用户提问: {text_prompt}")
        # print("\n🤖 模型回复:")
        # try:
        #     response = predictor.chat(text_prompt, max_new_tokens=400, temperature=0.5)
        #     print(response)
        # except Exception as e:
        #     print(f"[ERROR] 聊天功能出错: {e}")

#         # # ========== 测试2: 通用聊天功能（带图片）==========
#         print("\n" + "-"*60)
#         print("📌 测试2: 通用聊天功能（带图片描述）")
#         print("-"*60)
#         img_prompt = "请描述这张图片中写的有什么字"
#         # 提供具体的图片路径
#         test_img_path = "./test.png"    # 替换为实际的图片路径
#         print(f"\n🖼️  使用的图片路径: {test_img_path}")
#         print(f"👤 用户提问: {img_prompt}")
#         print("\n🤖 模型回复:")
#         try:
#             response = predictor.chat(img_prompt, images=[test_img_path], max_new_tokens=200)
#             print(response)
#         except Exception as e:
#             print(f"[ERROR] 图片聊天功能出错: {e}")

#         # ========== 测试3: 动作预测功能 ==========
        print("\n" + "-"*60)
        print("📌 测试3: 动作预测功能（连续帧）")
        print("-"*60)
        # 提供两张有位移变化的图片路径
        frame1_path = "./data/vlm_finetune_dataset_fixed/images/sample_0_25_0.png"  # 替换为实际的第一张图片路径
        frame2_path = "./data/vlm_finetune_dataset_fixed/images/sample_0_25_1.png"  # 替换为实际的第二张图片路径
        print(f"使用的图片路径:\n   Frame 1: {frame1_path}\n   Frame 2: {frame2_path}")
        try:
            direction, distance = predictor.get_action(frame1_path, frame2_path)
            print(f"\n🎯 预测结果:")
            print(f"   • 方向: {'→ 向右 (+)' if direction else '← 向左 (-)'}")  # 修正方向描述
            print(f"   • 距离: {distance} 像素")
        except Exception as e:
            print(f"[ERROR] 动作预测测试出错: {e}")

        print("\n" + "="*60)
        print("✅ 所有测试完成！")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 确保资源释放
        if 'predictor' in locals():
            predictor.close()

if __name__ == "__main__":
    main()
