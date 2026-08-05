import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

orig_dataset_dir = "/vepfs/group09/speech_data/Emilia_data"
gen_dataset_dir = "/vepfs/group09/F5-TTS_qingyu_MultilingualTTS/F5-TTS-main/Emilia_ZH_EN_gen_fixed"
langs = ["ZH", "EN"]
max_workers = 32

def process_single_jsonl(orig_jsonl_path: Path):
    """
    处理单个jsonl文件
    
    Args:
        orig_jsonl_path: 原始jsonl文件路径，如 /path/to/Emilia_Dataset_raw3/EN/EN_B00000.jsonl
    
    Returns:
        tuple: (处理的条目数, 跳过的条目数, 输出jsonl路径)
    """
    # 获取语言前缀和文件名
    lang = orig_jsonl_path.parent.name  # "EN" or "ZH"
    jsonl_name = orig_jsonl_path.name   # "EN_B00000.jsonl"
    
    # 构建输出路径
    output_jsonl_path = Path(gen_dataset_dir) / lang / jsonl_name
    
    # 确保输出目录存在
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    skipped_count = 0
    results = []
    
    with open(orig_jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
        for line in tqdm(lines, desc=f"{orig_jsonl_path.stem}"):
            obj = json.loads(line.strip())
            sample_id = obj["id"]  # 例如 "EN_B00000_S00000_W000000"
            
            # 从obj["wav"]获取相对路径，例如 "EN_B00000/EN_B00000_S00000/mp3/EN_B00000_S00000_W000000.mp3"
            wav_rel_path = obj["wav"]
            # 替换.mp3为.pt和.json，构建合成数据的相对路径
            base_rel_path = os.path.splitext(wav_rel_path)[0]  # 去掉.mp3后缀
            
            # 构建合成数据文件的完整路径
            gen_pt_path = Path(gen_dataset_dir) / lang / (base_rel_path + ".pt")
            gen_json_path = Path(gen_dataset_dir) / lang / (base_rel_path + ".json")
            
            # 检查是否同时存在.pt和.json文件
            if gen_pt_path.exists() and gen_json_path.exists():
                # 读取合成数据的json信息
                try:
                    with open(gen_json_path, "r", encoding="utf-8") as gen_f:
                        gen_info = json.load(gen_f)
                    
                    # 合并原始metadata和合成数据信息
                    new_obj = obj.copy()
                    new_obj["gen_len"] = gen_info["gen_len"]
                    new_obj["gen_text"] = gen_info["text"]  # 重命名为gen_text以区分原始text
                    
                    results.append(new_obj)
                    processed_count += 1
                except Exception as e:
                    print(f"Error reading {gen_json_path}: {e}")
                    skipped_count += 1
            else:
                skipped_count += 1
    
    # 写入新的jsonl文件
    with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for item in results:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return processed_count, skipped_count, str(output_jsonl_path)


def main():
    """主函数"""
    print(f"orig_dataset_dir: {orig_dataset_dir}")
    print(f"gen_dataset_dir: {gen_dataset_dir}")
    print(f"langs: {langs}")
    print()
    
    # 收集所有需要处理的jsonl文件
    jsonl_files = []
    for lang in langs:
        lang_dir = Path(orig_dataset_dir) / lang
        if lang_dir.exists():
            # 查找所有jsonl文件
            for jsonl_file in lang_dir.glob("*.jsonl"):
                jsonl_files.append(jsonl_file)
    
    print(f"Found {len(jsonl_files)} jsonl files to process")
    print()
    
    # 使用多进程并行处理
    total_processed = 0
    total_skipped = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_jsonl, jsonl_path) for jsonl_path in jsonl_files]
        
        for future in tqdm(futures, total=len(futures), desc="Processing jsonl files"):
            processed, skipped, output_path = future.result()
            total_processed += processed
            total_skipped += skipped
            # print(f"  -> {output_path}: processed={processed}, skipped={skipped}")
    
    print()
    print("=" * 50)
    print(f"Processing completed!")
    print(f"Total entries processed: {total_processed}")
    print(f"Total entries skipped: {total_skipped}")
    print(f"New jsonl files saved in: {gen_dataset_dir}")

if __name__ == "__main__":
    main()