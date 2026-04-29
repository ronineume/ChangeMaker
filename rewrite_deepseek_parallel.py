#!/usr/bin/env python3
"""
DeepSeek parallel rewrite generator for TruthfulQA.
Reads questions from CSV, processes first 200 questions, generates 14 variants.
python rewrite_deepseek_parallel.py --use-hardcoded-keys

"""

import os
import json
import time
import argparse
import requests
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional

class ParallelRewriteGenerator:
    """Parallel rewrite generator (14 variants)."""

    def __init__(self, api_keys: List[str], api_base: str = "https://api.deepseek.com"):
        if not api_keys:
            raise ValueError("At least one API key required")
        self.api_keys = api_keys
        self.current_key_idx = 0
        self.key_lock = threading.Lock()
        self.api_base = api_base.rstrip("/")
        self.api_url = f"{self.api_base}/chat/completions"
        
        # 14种改写变体（13种新提示词 + original）
        self.rewrite_types = [
            "1_sentence_reordering",           # 句子重排序
            "2_nominalization",                # 名词化改写
            "3_fill_in_blank",                 # 填空题转换
            "4_multiple_choice",               # 多选题转换（含无法回答选项）
            "5_binary_judgment",               # 二元判断（含无法回答选项）
            "6_bilingual_conversion",          # 关键术语双语转换
            "7_entity_temporal_renaming",      # 实体和时间重命名
            "8_benign_distractor",             # 良性干扰信息
            "9_colloquialisms",                # 口语化和填充词
            "10_cause_effect_reversal",        # 因果关系反转
            "11_premise_conclusion_inversion", # 前提结论反转
            "12_whole_part_reversal",          # 整体部分反转
            "13_covariant_transformation",     # 协变转换
            "original"                         # 原始文本
        ]
        
        # 提示词模板
        self.prompt_templates = {
            "1_sentence_reordering": """**Sentence Reordering Paraphrase**

Rewrite the following question by reordering sentences or clauses only.

Constraints:

* Preserve the exact meaning, logic, and all factual conditions.
* Do NOT add, remove, or change any information, entity, or relationship.
* You may only change the order of sentences/clauses (e.g., move background information earlier/later).
* You may reorder only where it does NOT change logical dependency (e.g., a definition must appear before it is used; a referenced entity must be introduced before reference).
* Do NOT change any wording, terminology, or phrasing—only the sequence of existing content.
* Output MUST be ONLY the rewritten question text. No explanations, no markdown formatting, no quotation marks.

Question:
{original_text}""",

            "2_nominalization": """**Nominalization Paraphrase (Noun Phrase Restructuring)**

Rewrite the following question by restructuring noun phrases only (e.g., converting possessive forms to "of"-phrases, or vice versa).

Constraints:

* Preserve the exact meaning, logic, and all factual conditions.
* You may ONLY modify noun phrase structures using equivalent transformations such as:
  * possessive "'s" ↔ "of" phrases (e.g., "Einstein's theory" ↔ "the theory of Einstein")
  * compound nouns ↔ prepositional phrases (e.g., "physics department" ↔ "department of physics")
  * gerund phrases ↔ noun phrases (e.g., "solving the equation" ↔ "the solution of the equation")
* Do NOT change any verbs, adjectives, adverbs, or clause-level structure.
* Do NOT add, remove, or alter any information.
* Do NOT change the question type (remain open-ended).
* Output MUST be ONLY the rewritten question text. No explanations, no markdown formatting, no quotation marks.

Question:
{original_text}""",

            "3_fill_in_blank": """**Fill-in-the-Blank Conversion**

Rewrite the following question as a fill-in-the-blank question.

Constraints:

* Identify the core answer or key information in the original question.
* Replace that answer/information with a single blank represented as "______" (six underscores).
* The blank MUST be the only missing element required to answer; all other context must be sufficient to determine what belongs in the blank.
* Preserve all given conditions, premises, and context from the original question.
* Do NOT create multiple blanks.
* Do NOT change, add, or remove any factual information—only restructure to hide the target answer.
* The rewritten question MUST be answerable with a single word or short phrase.
* Output MUST be ONLY the rewritten fill-in-the-blank question. No explanations, no markdown formatting, no quotation marks, no answer key.

Question:
{original_text}""",

            "4_multiple_choice": """**Multiple-Choice Conversion with Unanswerable Option**

Rewrite the following question as a multiple-choice question with four options.

Constraints:

* Identify the correct answer to the original question—this becomes the correct option (A).
* Create three distractor options (B, C, D) that are:
  * Plausible but incorrect based on the original question's context,
  * Consistent with the domain and topic,
  * Clearly distinguishable from the correct answer.
* The fourth option (D) MUST be "Unanswerable" or "Cannot be determined from the given information"—representing a case where the question lacks sufficient context to select a definitive answer.
* All four options MUST be mutually exclusive and collectively exhaustive of the possible response types (correct, plausible distractors, unanswerable).
* Do NOT add information that would make the "Unanswerable" option invalid.
* Preserve all original conditions, entities, and relationships in the question stem.
* Output format MUST be:
  * First: the question stem (rewritten as an interrogative or incomplete statement)
  * Then: four options labeled A), B), C), D) each on a new line
* No explanations, no markdown formatting beyond the required structure, no quotation marks, no indication of the correct answer.

Question:
{original_text}""",

            "5_binary_judgment": """**Binary Judgment with Unanswerable Option**

Rewrite the following question as a binary judgment question with an unanswerable option.

Constraints:

* Convert the original question into a form that requires a binary judgment (True/False or Yes/No) with an additional "Unanswerable" option.
* You MUST choose ONE of these two formats based on which preserves the original meaning more naturally:

  **Format A (Statement-based):** Convert to a declarative statement making a specific claim.
  - The claim MUST be either True, False, or Unanswerable relative to the original answer.
  - Output: statement followed by "- True", "- False", "- Unanswerable".

  **Format B (Question-based):** Convert to a yes/no question.
  - The question MUST be phrased such that "Yes" confirms, "No" denies, and "Unanswerable" indicates information gaps.
  - Output: question followed by "- Yes", "- No", "- Unanswerable".

* Selection rule: Use Format A when the original question asks for factual verification; use Format B when the original implies a polar inquiry or conditional check.
* The "Unanswerable" option MUST represent genuine inability to determine the answer from given information, not ambiguity or trickery.
* Do NOT add information that would make Unanswerable become answerable.
* Preserve all original entities, numbers, and relationships.
* Output format MUST be:
  * First: the statement OR question (one sentence)
  * Then: the three options each on a new line, prefixed with "- " (dash and space)
* No explanations, no markdown formatting beyond the required structure, no quotation marks.

Question:
{original_text}""",

            "6_bilingual_conversion": """**Key Term Bilingual Conversion with English Response Constraint**

Rewrite the following question by converting key answer-relevant terms to Chinese while preserving the grammatical structure and overall English framework.

Constraints:

* Identify terms, entities, or phrases in the question that are directly relevant to determining or formulating the answer.
* Convert ONLY these key terms to their standard Chinese equivalents (e.g., proper nouns, technical terms, core concepts, named entities).
* Maintain English grammar, syntax, question structure, and function words (articles, prepositions, auxiliary verbs).
* The rewritten question must remain readable as an English sentence with embedded Chinese terms.
* Do NOT translate the entire question—only strategically selected key terms.
* Do NOT alter the logical structure, dependencies, or answer requirements of the original question.
* Output MUST consist of exactly two parts:
  1. The rewritten bilingual question (one coherent sentence/paragraph)
  2. The mandatory constraint sentence: "Provide your answer entirely in English."
* No explanations, no markdown formatting, no quotation marks around the question.

Question:
{original_text}""",

            "7_entity_temporal_renaming": """**Similar Entity and Temporal Renaming**

Rewrite the following question by renaming similar entities (e.g., persons, objects, locations of the same category) and temporal references with equivalent alternatives.

Constraints:

* Identify entities that belong to the same semantic category (e.g., two scientists, two cities, two time periods) and temporal markers (dates, durations, sequences).
* Swap or rename these similar entities and temporal references with equivalent alternatives from the same category.
* Maintain the logical relationships and dependencies—if entity A originally relates to entity B in a specific way, the renamed entities must preserve that relational structure.
* Ensure temporal sequences remain logically consistent (e.g., "before" and "after" relations preserved under renaming).
* Do NOT change the mathematical structure, numerical values, or causal/logical dependencies.
* Do NOT introduce entities from different categories or alter the problem type.
* All renamings must be consistent throughout the entire problem.
* The final answer remains the same as the original problem (this is a surface-level transformation).
* Output MUST be ONLY the rewritten problem text (one coherent paragraph).
* No explanations, no markdown formatting, no quotation marks.

Question:
{original_text}""",

            "8_benign_distractor": """**Benign Distractor**

Rewrite the following question by adding benign distractor information—extra details or conditions that are entirely unrelated to the question's core subject matter.

Constraints:

* Identify the domain and topic of the original question.
* Add 1-3 pieces of information that are factually true but completely irrelevant to answering the question.
* Distractor information must belong to a distinctly different domain from the question's topic (e.g., if the question is about history, add details about weather, cuisine, or unrelated geography).
* Distractors must not create logical connections, hidden constraints, or alternative solution paths to the original question.
* Distractors must not conflict with or contradict any information in the original question.
* Integrate distractors naturally into the question text without marking them as irrelevant.
* Do NOT alter, remove, or weaken any original condition, entity, number, or relationship.
* Do NOT add information that could be interpreted as relevant through indirect association.
* The final answer must remain exactly the same as the original question.
* Output MUST be ONLY the rewritten question text (one coherent paragraph).
* No explanations, no markdown formatting, no quotation marks.

Question:
{original_text}""",

            "9_colloquialisms": """**Colloquialisms and Filler Words**

Rewrite the following question by converting it into informal, conversational internet-style language with colloquialisms and filler expressions.

Constraints:

* Preserve the exact factual content, entities, and answer requirements of the original question.
* Transform the tone to simulate casual online communication (e.g., social media, chat forums, streaming comments).
* Incorporate characteristic elements such as:
  * Filler words and vocalized pauses (e.g., "emm", "uhh", "like", "y'know", "tbh", "ngl")
  * Informal contractions and elisions (e.g., "wanna", "gonna", "kinda", "sorta")
  * Expressive punctuation and emphasis (e.g., multiple punctuation marks, tildes, capitalization for emphasis)
  * Internet slang and discourse markers (e.g., "so basically", "literally", "actually", "wait", "okay but")
  * Roleplay personas or affectations (e.g., catgirl speech patterns with "~" endings, enthusiastic fan tone, overly casual friendliness)
* Maintain grammatical intelligibility—the question must remain answerable despite the informal packaging.
* Do NOT alter the core information, named entities, numerical values, or the type of answer required.
* Do NOT add or remove substantive content—only repackage with stylistic flourishes.
* Output MUST be ONLY the rewritten question text (one coherent paragraph).
* No explanations, no markdown formatting, no quotation marks.

Question:
{original_text}""",

            "10_cause_effect_reversal": """**Cause and Effect Reversal**

Rewrite the following question by reversing the causal relationship between entities or events.

Constraints:

* Identify the original cause-effect chain in the question.
* Swap the roles: the original cause becomes the effect, and the original effect becomes the cause.
* Preserve the core factual relationships and logical structure under the reversal.
* Update all pronouns, temporal markers, and logical connectors to maintain coherence under the new causal direction.
* Do NOT introduce new variables, new entities, or remove essential constraints.
* The reversed question must remain mathematically/logically well-defined and solvable.
* The final answer MUST be different from the original problem due to the causal inversion.
* Output format MUST be:
  * First: the rewritten problem text (one coherent paragraph)
  * Then: the line `Answer:` followed by the final answer only (no derivation).
* The final answer MUST be wrapped in `\\boxed{{}}`.
* No explanations, no reasoning steps, no commentary.

Question:
{original_text}""",

            "11_premise_conclusion_inversion": """**Premise-Conclusion Inversion**

Rewrite the following question by swapping the premise (given information) with the conclusion (what is asked).

Constraints:

* Identify what is given (premise) and what is sought (conclusion) in the original question.
* Restructure the question such that the original conclusion becomes the given condition, and the original premise becomes the target to solve for.
* Preserve all mathematical relationships, equations, and logical dependencies under the inversion.
* Update all symbols, variables, and references consistently to reflect the swapped roles.
* Do NOT introduce new constraints, new variables, or alter the underlying mathematical structure.
* The inverted question must be self-contained, well-defined, and solvable.
* The final answer MUST be different from the original problem.
* Output format MUST be:
  * First: the rewritten problem text (one coherent paragraph)
  * Then: the line `Answer:` followed by the final answer only (no derivation).
* The final answer MUST be wrapped in `\\boxed{{}}`.
* No explanations, no reasoning steps, no commentary.

Question:
{original_text}""",

            "12_whole_part_reversal": """**Whole-Part Reversal**

Rewrite the following question by swapping the roles of the whole (aggregate/total/system) and the part (component/subset/element).

Constraints:

* Identify the whole (complete entity, total set, system, or aggregate category) and the part (member, subset, component, or specific instance) in the original question.
* Swap their roles: the whole becomes the part, and the part becomes the whole.
* Preserve the factual relationships and logical structure under the reversal (e.g., if the question asks about a property of a part within a whole, now ask about the whole as a part of a larger context, or the part as the new whole containing the original whole).
* Adjust descriptive language, scope references, and relational terms to maintain coherence under the swapped roles.
* Do NOT introduce new factual information or entities outside the original scope.
* Do NOT change the question type (remain open-ended) or alter the core verification task.
* Update all referential expressions (pronouns, definite descriptions) consistently to reflect the new role assignments.
* The final answer MUST be different from the original problem due to the structural inversion.
* Output format MUST be:
  * First: the rewritten problem text (one coherent paragraph)
  * Then: the line `Answer:` followed by the final answer only (no derivation).
* The final answer MUST be wrapped in `\\boxed{{}}`.
* No explanations, no reasoning steps, no commentary.

Question:
{original_text}""",

            "13_covariant_transformation": """**Covariant Transformation**

Rewrite the following question by applying a covariant transformation that preserves relational structure while systematically shifting temporal, ordinal, or scalar references along a continuous dimension.

Constraints:

* Identify covariant relationships in the original question involving temporal sequences (dates, eras, durations), ordinal rankings (presidential terms, editions, versions), or scalar progressions (magnitude, intensity, frequency).
* Apply a consistent directional shift along the identified dimension:
  * Temporal: shift forward or backward by a fixed period (e.g., +100 years, -50 years) while preserving relative chronology
  * Ordinal: increment or decrement the position in sequence (e.g., 16th president → 26th president, 3rd edition → 5th edition)
  * Scalar: invert the direction of variation (increase ↔ decrease, before ↔ after, earlier ↔ later) or apply proportional scaling
* Preserve the relational structure and logical dependencies under the transformation (e.g., if A happened before B originally, maintain the "before" relationship after temporal shift).
* Maintain all entities, events, and factual relationships—only modify their positioning along the continuous dimension.
* Update all temporal markers, ordinal numbers, and comparative terms consistently to reflect the transformation.
* Do NOT introduce new entities, alter categorical relationships, or remove essential constraints.
* The transformation must be reversible and mathematically consistent (e.g., shifting +100 years then -100 years returns to original).
* The final answer MUST be different from the original problem due to the covariant shift.
* Output format MUST be:
  * First: the rewritten problem text (one coherent paragraph)
  * Then: the line `Answer:` followed by the final answer only (no derivation).
* The final answer MUST be wrapped in `\\boxed{{}}`.
* No explanations, no reasoning steps, no commentary.

Question:
{original_text}"""
        }
        
        self.total_requests = 0
        self.errors = 0
        self.success_count = 0
        self.stats_lock = threading.Lock()
        self.file_lock = threading.Lock()

    def get_next_api_key(self) -> str:
        """Return next API key (round-robin)."""
        with self.key_lock:
            key = self.api_keys[self.current_key_idx]
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            return key
    
    def generate_prompt(self, original_text: str, rewrite_type: str) -> str:
        """Build prompt for rewrite type."""
        if rewrite_type == "original":
            return original_text
            
        template = self.prompt_templates.get(rewrite_type)
        if template:
            return template.replace("{original_text}", original_text)
        
        return original_text
    
    def rewrite_with_deepseek(self, original_text: str, rewrite_type: str, max_retries: int = 5) -> Optional[str]:
        """Call DeepSeek API to produce rewrite."""
        for attempt in range(max_retries):
            try:
                api_key = self.get_next_api_key()
                prompt = self.generate_prompt(original_text, rewrite_type)
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=90
                )
                
                if response.status_code == 200:
                    result = response.json()
                    rewritten = result["choices"][0]["message"]["content"].strip()
                    
                    with self.stats_lock:
                        self.success_count += 1
                        self.total_requests += 1
                    
                    return rewritten
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    print(f"Rate limit, wait {retry_after}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_after)
                else:
                    print(f"API error {response.status_code}: {response.text[:100]} (attempt {attempt+1}/{max_retries})")
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                print(f"Request error: {e} (attempt {attempt+1}/{max_retries})")
                time.sleep(2 ** attempt)
        
        with self.stats_lock:
            self.errors += 1
            self.total_requests += 1
        
        return None
    
    def process_question(self, idx: int, question_text: str, output_dir: Path) -> List[Dict[str, Any]]:
        """Process one question: all rewrite variants."""
        # 生成问题ID（基于索引，如 Q000001）
        problem_id = f"Q{idx:06d}"
        results = []
        
        for rewrite_type in self.rewrite_types:
            output_file = output_dir / f"{problem_id}__{rewrite_type}.txt"
            
            # 如果文件已存在且非空，跳过
            if output_file.exists() and output_file.stat().st_size > 0:
                print(f"  Skip (exists): {output_file.name}")
                results.append({
                    "problem_id": problem_id,
                    "rewrite_type": rewrite_type,
                    "status": "skipped"
                })
                continue
            
            if rewrite_type == "original":
                # 直接保存原始文本
                with self.file_lock:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(question_text)
                print(f"  Saved original: {output_file.name}")
                results.append({
                    "problem_id": problem_id,
                    "rewrite_type": rewrite_type,
                    "status": "copied"
                })
            else:
                # 调用API改写
                print(f"  Rewriting: {problem_id}__{rewrite_type}")
                rewritten = self.rewrite_with_deepseek(question_text, rewrite_type)
                
                if rewritten:
                    with self.file_lock:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(rewritten)
                    
                    results.append({
                        "problem_id": problem_id,
                        "rewrite_type": rewrite_type,
                        "status": "success"
                    })
                else:
                    print(f"    Failed: {problem_id}__{rewrite_type}")
                    results.append({
                        "problem_id": problem_id,
                        "rewrite_type": rewrite_type,
                        "status": "failed"
                    })
        
        return results
    
    def run_parallel_rewrite(self, csv_path: str, output_dir: str, 
                           num_questions: int = 200,
                           max_workers: int = 10):
        """Run parallel rewrite for questions from CSV."""
        csv_file = Path(csv_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 读取CSV文件
        print(f"Loading CSV: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
        
        # 检查question列是否存在
        if 'Question' not in df.columns:
            print(f"Error: CSV does not contain 'Question' column. Available columns: {df.columns.tolist()}")
            return
        
        # 获取前N个问题
        questions = df['Question'].head(num_questions).tolist()
        print(f"Loaded {len(questions)} questions from CSV")
        
        # 准备任务列表（每个问题是一个任务）
        print(f"Problems to process: {len(questions)}")
        print(f"Variants per problem: {len(self.rewrite_types)}")
        print(f"Total tasks: {len(questions) * len(self.rewrite_types)}")
        print(f"API keys: {len(self.api_keys)}, workers: {max_workers}")
        print("-" * 60)
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, question in enumerate(questions):
                # 清理问题文本
                question_text = str(question).strip()
                if not question_text or question_text == 'nan':
                    continue
                    
                future = executor.submit(self.process_question, idx, question_text, output_path)
                futures.append(future)
                time.sleep(0.1)  # 避免启动过快
            
            # 收集结果
            for i, future in enumerate(futures):
                try:
                    results = future.result()
                    success_count = sum(1 for r in results if r["status"] in ["success", "copied"])
                    total_count = len(results)
                    
                    if (i + 1) % 10 == 0 or (i + 1) == len(futures):
                        with self.stats_lock:
                            print(f"[{i+1}/{len(futures)}] done, ok: {success_count}/{total_count}, "
                                  f"total ok: {self.success_count}, errors: {self.errors}")
                except Exception as e:
                    print(f"Error processing question {i}: {e}")
        
        # 保存处理摘要
        summary_file = output_path / "_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_questions": len(questions),
                "variants_per_question": len(self.rewrite_types),
                "total_requests": self.total_requests,
                "success_count": self.success_count,
                "errors": self.errors,
                "rewrite_types": self.rewrite_types
            }, f, indent=2)
        
        print("-" * 60)
        print("Rewrite done:")
        print(f"  Questions processed: {len(questions)}")
        print(f"  Total API requests: {self.total_requests}")
        print(f"  Success: {self.success_count}")
        print(f"  Errors: {self.errors}")
        print(f"  Output: {output_path}")
        print(f"  Summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="DeepSeek parallel rewrite generator for TruthfulQA")
    parser.add_argument("--csv", type=str, default=r"E:\Work_Station\Hallu\QA\TruthfulQA.csv",
                       help="Path to TruthfulQA CSV file")
    parser.add_argument("--output", type=str, default="truthfulqa_rewrites_deepseek",
                       help="Output rewrite directory")
    parser.add_argument("--num", type=int, default=200,
                       help="Number of questions to process (first N)")
    parser.add_argument("--workers", type=int, default=10,
                       help="Concurrent workers")
    parser.add_argument("--api-base", type=str, default="https://api.deepseek.com",
                       help="DeepSeek API base URL")
    parser.add_argument("--use-hardcoded-keys", action="store_true",
                       help="强制使用代码中硬编码的API key，忽略环境变量")
    args = parser.parse_args()
    
    # 硬编码的API密钥（修改这里）
    HARDCODED_API_KEYS = [
        "sk-c27ca096a5f0402d9a70f59a6409ff3c",  # feb4
        "sk-84bdd1381bba405a9eff4e0714201241",  # feb5
        "sk-cac8f0e5bf61430a8eeea28ee9077af3",  # Feb1
        "sk-6cc972c783ec480587e963909bf22216",  # Feb2
        "sk-3cd3dc84cbdf4bf8817e2a924dd4b320"   # Feb3
    ]
    
    api_keys = []
    
    # 如果指定了 --use-hardcoded-keys，强制使用硬编码密钥
    if args.use_hardcoded_keys:
        print("[INFO] 使用硬编码API key（--use-hardcoded-keys 已指定）")
        api_keys = HARDCODED_API_KEYS.copy()
    else:
        # 从环境变量读取API密钥（推荐方式）
        for i in range(1, 10):  # 支持最多9个密钥
            key = os.environ.get(f"DEEPSEEK_API_KEY_{i}") or os.environ.get(f"DEEPSEEK_API_KEY")
            if key and key not in api_keys:
                api_keys.append(key)
                if i == 1:
                    break  # 如果只设置了DEEPSEEK_API_KEY，只取一个
    
    # 如果没有获取到任何key，使用硬编码作为后备
    if not api_keys:
        print("Warning: No API keys found. Using hardcoded keys.")
        api_keys = HARDCODED_API_KEYS.copy()
    
    print("=" * 60)
    print("DeepSeek TruthfulQA Parallel Rewrite Generator")
    print("=" * 60)
    print(f"API keys: {len(api_keys)}")
    print(f"CSV file: {args.csv}")
    print(f"Output: {args.output}")
    print(f"Questions to process: {args.num}")
    print(f"Workers: {args.workers}")
    # 变体数量是固定的，直接显示，不需要创建实例
    print("Variants: 14 types")
    print("=" * 60)
    
    try:
        generator = ParallelRewriteGenerator(api_keys, api_base=args.api_base)
        generator.run_parallel_rewrite(
            csv_path=args.csv,
            output_dir=args.output,
            num_questions=args.num,
            max_workers=args.workers
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()