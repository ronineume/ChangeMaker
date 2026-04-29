#!/usr/bin/env python3
"""
TruthfulQA 改写数据集单次评估脚本 (温度=0)
- 每个问题只测1次
- 温度设为0（确定性输出）
- 对于改写类型 10-13，去除 Answer: \\boxed{{}} 部分
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
    """获取 API keys"""
    HARDCODED_KEYS = [
        "sk-srnnnhejkqb",
        "sk-f",
    ]
    
    if os.environ.get("USE_HARDCODED_KEYS", "true").lower() == "true":
        print("[INFO] 使用硬编码 API keys")
        return HARDCODED_KEYS
    
    env_keys = os.environ.get("SILICONFLOW_API_KEYS", "").strip()
    if env_keys:
        keys = [k.strip() for k in env_keys.replace("；", ",").split(",") if k.strip()]
        if keys:
            return keys
    
    single_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if single_key:
        return [single_key]
    
    return HARDCODED_KEYS


class TruthfulQASingleEvaluator:
    """TruthfulQA 单次评估器 (温度=0)"""
    
    def __init__(self, api_keys: List[str], model: str = DEFAULT_MODEL, max_tokens: int = 2048):
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.key_lock = threading.Lock()
        self.model = model
        self.max_tokens = max_tokens
        # 温度固定为0
        self.temperature = 0.0
        
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
        """提取问题文本，10-13类型去除答案部分"""
        content = file_content.strip()
        
        if any(rt in rewrite_type for rt in REWRITE_TYPES_WITH_ANSWER):
            patterns = [
                r'\n?\s*Answer[:：]\s*\n.*',
                r'\n?\s*Answer[:：].*',
            ]
            for pattern in patterns:
                content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        return content.strip()
    
    def call_llm(self, question_text: str, max_retries: int = 5) -> Optional[str]:
        """调用 LLM 单次（温度=0）"""
        for attempt in range(max_retries):
            try:
                api_key = self.get_next_api_key()
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                prompt = question_text + "\n\nLet's output the final answer."
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant. Please answer questions accurately and concisely."},
                        {"role": "user", "content": prompt}
                    ],
                    # 温度固定为0
                    "temperature": 0.0,
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
                    
                    with self.stats_lock:
                        self.success_count += 1
                        self.total_requests += 1
                    
                    return answer
                    
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 1))
                    print(f"    Rate limit, wait {retry_after}s...")
                    time.sleep(retry_after)
                else:
                    print(f"    API error {response.status_code}: {response.text[:100]}")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"    Request error: {e}")
                time.sleep(1)
        
        with self.stats_lock:
            self.error_count += 1
            self.total_requests += 1
        
        return None
    
    def process_single_file(self, file_path: Path, output_dir: Path) -> Dict:
        """处理单个问题文件，只获取1次回答"""
        stem = file_path.stem
        parts = stem.split("__")
        if len(parts) != 2:
            return {"file": str(file_path), "status": "invalid_filename"}
        
        problem_id = parts[0]
        rewrite_type = parts[1]
        
        output_file = output_dir / f"{stem}.json"
        
        # 如果已存在，跳过
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if existing.get("llm_response"):
                    return {
                        "file": str(file_path),
                        "problem_id": problem_id,
                        "rewrite_type": rewrite_type,
                        "status": "skipped"
                    }
            except:
                pass
        
        # 读取问题文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            return {"file": str(file_path), "status": "read_error", "error": str(e)}
        
        # 提取问题文本
        question_text = self.extract_question_text(file_content, rewrite_type)
        
        # 调用 LLM 1次
        print(f"  Processing: {stem}")
        llm_response = self.call_llm(question_text)
        
        # 保存结果
        result = {
            "problem_id": problem_id,
            "rewrite_type": rewrite_type,
            "original_question": file_content,
            "processed_question": question_text,
            "llm_response": llm_response,  # 单次回答
            "temperature": 0.0,
            "status": "success" if llm_response else "failed"
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return {
            "file": str(file_path),
            "problem_id": problem_id,
            "rewrite_type": rewrite_type,
            "status": result["status"]
        }
    
    def run_evaluation(self, rewrites_dir: str, output_dir: str, 
                       max_workers: int = 5, limit: Optional[int] = None,
                       rewrite_type_filter: Optional[str] = None):
        """运行评估"""
        rewrites_path = Path(rewrites_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有问题文件
        all_files = sorted(rewrites_path.glob("Q*__*.txt"))
        all_files = [f for f in all_files if "_processing_summary" not in f.name]
        
        # 按改写类型过滤
        if rewrite_type_filter:
            all_files = [f for f in all_files if rewrite_type_filter in f.stem]
            print(f"[INFO] 过滤改写类型: {rewrite_type_filter}")
        
        if limit:
            all_files = all_files[:limit]
        
        print("=" * 60)
        print("TruthfulQA Single-Run Evaluation (Temperature=0)")
        print("=" * 60)
        print(f"Total files to process: {len(all_files)}")
        print(f"API keys: {len(self.api_keys)}")
        print(f"Workers: {max_workers}")
        print(f"Model: {self.model}")
        print(f"Temperature: 0.0 (固定)")
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
                executor.submit(self.process_single_file, f, output_path): f 
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
            "success": self.success_count,
            "errors": self.error_count,
            "model": self.model,
            "temperature": 0.0,
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
    parser = argparse.ArgumentParser(description="TruthfulQA single-run evaluation (temp=0)")
    parser.add_argument("--rewrites-dir", type=str, 
                       default="truthfulqa_rewrites_deepseek",
                       help="改写问题目录")
    parser.add_argument("--output", type=str, 
                       default="truthfulqa_eval_single",
                       help="输出目录")
    parser.add_argument("--model", type=str, 
                       default=DEFAULT_MODEL,
                       help=f"模型名称 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--workers", type=int, default=5,
                       help="并发数 (默认: 5)")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制处理的文件数量")
    parser.add_argument("--rewrite-type", type=str, default=None,
                       help="只处理特定改写类型，如 '1_sentence_reordering'")
    parser.add_argument("--use-env-keys", action="store_true",
                       help="使用环境变量中的 API keys")
    
    args = parser.parse_args()
    
    if args.use_env_keys:
        os.environ["USE_HARDCODED_KEYS"] = "false"
    
    api_keys = get_api_keys()
    if not api_keys:
        print("[ERROR] 未找到 API keys")
        return
    
    evaluator = TruthfulQASingleEvaluator(
        api_keys=api_keys,
        model=args.model
    )
    
    evaluator.run_evaluation(
        rewrites_dir=args.rewrites_dir,
        output_dir=args.output,
        max_workers=args.workers,
        limit=args.limit,
        rewrite_type_filter=args.rewrite_type
    )


if __name__ == "__main__":
    main()
