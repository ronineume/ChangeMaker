#!/usr/bin/env python3
"""
TruthfulQA 改写数据集评估脚本
- 读取 truthfulqa_rewrites_deepseek 目录中的问题
- 对于改写类型 10-13，去除 Answer: \\boxed{{}} 部分
- 调用 SiliconFlow API 获取 LLM 回答
- 只保存 LLM 生成的答案
"""

import os
import re
import json
import time
import argparse
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional


# ========== 配置 ==========
# SiliconFlow API 配置
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 需要去除答案部分的改写类型（10-13）
REWRITE_TYPES_WITH_ANSWER = [
    "10_cause_effect_reversal",
    "11_premise_conclusion_inversion",
    "12_whole_part_reversal",
    "13_covariant_transformation"
]


def get_api_keys() -> List[str]:
    """获取 API keys，优先硬编码，其次环境变量"""
    # 硬编码 API keys（修改这里）
    HARDCODED_KEYS = [
        "sk-srnnnhejkqbbcxcfkogaamjbzolgmsympmmlajwsraulibns",  
        "sk-fpnyaeclqmgugagnapzqvddfzaodgpwrwmlfflbwbksodutz",  
        
    ]
    
    # 检查是否强制使用硬编码
    if os.environ.get("USE_HARDCODED_KEYS", "true").lower() == "true":
        print("[INFO] 使用硬编码 API keys")
        return HARDCODED_KEYS
    
    # 从环境变量读取
    env_keys = os.environ.get("SILICONFLOW_API_KEYS", "").strip()
    if env_keys:
        keys = [k.strip() for k in env_keys.replace("；", ",").split(",") if k.strip()]
        if keys:
            return keys
    
    single_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if single_key:
        return [single_key]
    
    print("[WARNING] 未找到 API key，使用硬编码")
    return HARDCODED_KEYS


class TruthfulQAEvaluator:
    """TruthfulQA 改写数据集评估器"""
    
    def __init__(self, api_keys: List[str], model: str = DEFAULT_MODEL, 
                 temperature: float = 1.0, max_tokens: int = 4096):
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.key_lock = threading.Lock()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 统计
        self.stats_lock = threading.Lock()
        self.total_requests = 0
        self.success_count = 0
        self.error_count = 0
        
    def get_next_api_key(self) -> str:
        """轮询获取下一个 API key"""
        with self.key_lock:
            key = self.api_keys[self.current_key_idx]
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            return key
    
    def extract_question_text(self, file_content: str, rewrite_type: str) -> str:
        """
        从文件内容中提取问题文本
        对于类型 10-13，需要去除 Answer: \\boxed{{}} 部分
        """
        content = file_content.strip()
        
        # 检查是否需要去除答案部分
        if any(rt in rewrite_type for rt in REWRITE_TYPES_WITH_ANSWER):
            # 匹配 "Answer:" 及其后面的内容（包括 \\boxed{}）
            # 支持多种格式：Answer:, Answer：, Answer:\n, 等
            patterns = [
                r'\n?\s*Answer[:：]\s*\n.*',  # Answer: 后面所有内容
                r'\n?\s*Answer[:：].*',         # Answer: 在同一行
            ]
            
            for pattern in patterns:
                content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        return content.strip()
    
    def call_llm_single(self, question_text: str) -> Optional[str]:
        """
        单次调用 SiliconFlow API
        失败时休眠1秒后返回None
        """
        try:
            api_key = self.get_next_api_key()
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # 添加指令要求输出格式
            prompt = question_text + "\n\nLet's output the final answer."
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant. Please answer questions accurately and concisely."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                SILICONFLOW_API_URL,
                headers=headers,
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                return answer
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                print(f"    Rate limit, wait {retry_after}s...")
                time.sleep(retry_after)
                return None
            else:
                print(f"    API error {response.status_code}: {response.text[:100]}")
                time.sleep(1)  # 失败休眠1秒
                return None
                
        except Exception as e:
            print(f"    Request error: {e}")
            time.sleep(1)  # 失败休眠1秒
            return None
    
    def call_llm(self, question_text: str, n_trials: int = 20) -> List[str]:
        """
        调用 SiliconFlow API 多次（默认20次）
        每次失败都会休眠1秒后重试，直到成功20次或达到最大尝试次数
        """
        answers = []
        max_attempts = n_trials * 5  # 最大尝试次数为目标的5倍
        attempts = 0
        
        while len(answers) < n_trials and attempts < max_attempts:
            attempts += 1
            answer = self.call_llm_single(question_text)
            
            if answer is not None:
                answers.append(answer)
                with self.stats_lock:
                    self.success_count += 1
                    self.total_requests += 1
            else:
                with self.stats_lock:
                    self.total_requests += 1
        
        # 统计失败的请求数
        failed = attempts - len(answers)
        if failed > 0:
            with self.stats_lock:
                self.error_count += failed
        
        return answers
    
    def process_single_file(self, file_path: Path, output_dir: Path, n_trials: int = 20) -> Dict:
        """处理单个问题文件，获取 n_trials 次回答"""
        # 解析文件名: Q000001__1_sentence_reordering.txt
        stem = file_path.stem
        parts = stem.split("__")
        if len(parts) != 2:
            return {"file": str(file_path), "status": "invalid_filename"}
        
        problem_id = parts[0]
        rewrite_type = parts[1]
        
        # 输出文件路径
        output_file = output_dir / f"{stem}.json"
        
        # 如果已存在且有效（已有20个回答），跳过
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                responses = existing.get("llm_responses", [])
                if len(responses) >= n_trials:
                    return {
                        "file": str(file_path),
                        "problem_id": problem_id,
                        "rewrite_type": rewrite_type,
                        "status": "skipped",
                        "responses_count": len(responses)
                    }
            except:
                pass  # 文件损坏，重新处理
        
        # 读取问题文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            return {"file": str(file_path), "status": "read_error", "error": str(e)}
        
        # 提取问题文本（对于10-13类型去除答案部分）
        question_text = self.extract_question_text(file_content, rewrite_type)
        
        # 调用 LLM 20次
        print(f"  Processing: {stem} (目标 {n_trials} 次回答)")
        llm_responses = self.call_llm(question_text, n_trials=n_trials)
        
        # 保存结果
        result = {
            "problem_id": problem_id,
            "rewrite_type": rewrite_type,
            "original_question": file_content,
            "processed_question": question_text,
            "llm_responses": llm_responses,  # 20个回答的列表
            "responses_count": len(llm_responses),
            "target_count": n_trials,
            "status": "success" if len(llm_responses) >= n_trials else "partial" if llm_responses else "failed"
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return {
            "file": str(file_path),
            "problem_id": problem_id,
            "rewrite_type": rewrite_type,
            "status": result["status"],
            "responses_count": len(llm_responses)
        }
    
    def run_evaluation(self, rewrites_dir: str, output_dir: str, 
                       max_workers: int = 5, limit: Optional[int] = None,
                       n_trials: int = 20):
        """运行评估"""
        rewrites_path = Path(rewrites_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有问题文件
        all_files = sorted(rewrites_path.glob("Q*__*.txt"))
        
        # 排除 _processing_summary.json 等非问题文件
        all_files = [f for f in all_files if "_processing_summary" not in f.name]
        
        if limit:
            all_files = all_files[:limit]
        
        print("=" * 60)
        print("TruthfulQA Rewrites Evaluation")
        print("=" * 60)
        print(f"Total files to process: {len(all_files)}")
        print(f"Trials per question: {n_trials}")
        print(f"API keys: {len(self.api_keys)}")
        print(f"Workers: {max_workers}")
        print(f"Model: {self.model}")
        print(f"Output: {output_path}")
        print("=" * 60)
        
        # 统计改写类型
        type_counts = {}
        for f in all_files:
            rt = f.stem.split("__")[1] if "__" in f.stem else "unknown"
            type_counts[rt] = type_counts.get(rt, 0) + 1
        
        print("\nRewrite type distribution:")
        for rt, count in sorted(type_counts.items()):
            marker = " [no answer shown]" if any(x in rt for x in REWRITE_TYPES_WITH_ANSWER) else ""
            print(f"  {rt}: {count}{marker}")
        print()
        
        # 并行处理
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_single_file, f, output_path, n_trials): f 
                for f in all_files
            }
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                
                if (i + 1) % 50 == 0 or (i + 1) == len(all_files):
                    print(f"[{i+1}/{len(all_files)}] processed, "
                          f"success: {self.success_count}, errors: {self.error_count}")
        
        # 保存汇总
        summary = {
            "total_files": len(all_files),
            "n_trials_per_question": n_trials,
            "success": self.success_count,
            "errors": self.error_count,
            "model": self.model,
            "rewrite_types": dict(type_counts)
        }
        
        summary_file = output_path / "_evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("\n" + "=" * 60)
        print("Evaluation completed!")
        print(f"  Total: {len(all_files)}")
        print(f"  Success: {self.success_count}")
        print(f"  Errors: {self.error_count}")
        print(f"  Output: {output_path}")
        print(f"  Summary: {summary_file}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TruthfulQA rewrites evaluation")
    parser.add_argument("--rewrites-dir", type=str, 
                       default="truthfulqa_rewrites_deepseek",
                       help="改写问题目录")
    parser.add_argument("--output", type=str, 
                       default="truthfulqa_eval_results",
                       help="输出目录")
    parser.add_argument("--model", type=str, 
                       default=DEFAULT_MODEL,
                       help=f"模型名称 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=5,
                       help="并发数 (默认: 5)")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制处理的文件数量（用于测试）")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="温度参数 (默认: 1.0)")
    parser.add_argument("--n-trials", type=int, default=20,
                       help="每个问题的尝试次数 (默认: 20)")
    parser.add_argument("--use-env-keys", action="store_true",
                       help="使用环境变量中的 API keys（默认使用硬编码）")
    
    args = parser.parse_args()
    
    # 设置是否使用环境变量
    if args.use_env_keys:
        os.environ["USE_HARDCODED_KEYS"] = "false"
    
    # 获取 API keys
    api_keys = get_api_keys()
    if not api_keys:
        print("[ERROR] 未找到 API keys")
        return
    
    # 创建评估器并运行
    evaluator = TruthfulQAEvaluator(
        api_keys=api_keys,
        model=args.model,
        temperature=args.temperature
    )
    
    evaluator.run_evaluation(
        rewrites_dir=args.rewrites_dir,
        output_dir=args.output,
        max_workers=args.workers,
        limit=args.limit,
        n_trials=args.n_trials
    )


if __name__ == "__main__":
    main()
