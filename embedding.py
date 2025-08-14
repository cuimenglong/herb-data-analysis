#embedding process

from sentence_transformers import SentenceTransformer
import os 
from matplotlib import pyplot as plt
import json
import torch

#1.加载数据
json_path = "*"   #please enter your herb_dataset path
file_names = os.listdir(json_path)
result_dict = {}

for file_name in file_names:
    if file_name == ".ipynb_checkpoints":
        continue
    file_path = os.path.join(json_path,file_name)
    # print(file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result_dict[data["药物"]] = (data["治疗"])

herb_diseases = {}
for herb in list(result_dict.keys()):
    herb_diseases[herb] = []
    # print(herb)
    for case in result_dict[herb]:
        herb_diseases[herb].append(case["疾病"])


        
# 2. 设置模型
model = SentenceTransformer('Qwen/Qwen3-Embedding-8B', # 要使用的预训练模型
cache_folder=r"cache",device='cuda' if torch.cuda.is_available() else 'cpu') # 指定本地缓存路径
model.eval()


# 3.嵌入并取平均
with torch.no_grad():
    entity_embeddings = {}
    n = 0
    for herb, uses in herb_diseases.items():
        if not uses:  # 检查用途列表是否为空
            continue
        use_embeddings = model.encode(uses)  # 嵌入形状: [n_uses, embedding_dim]
        avg_embedding = use_embeddings.mean(axis=0)  # 平均所有用途的嵌入
        entity_embeddings[herb] = avg_embedding
        n += 1
        if n % 100 == 0 :
            print(n)


    # 保存嵌入向量
    output_path = "entity_embeddings.json"

    #转换成 Python 列表
    entity_embeddings_serializable = {
        herb: embedding.tolist()  # 关键：.tolist() 转换 NumPy 数组
        for herb, embedding in entity_embeddings.items()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entity_embeddings_serializable, f, ensure_ascii=False, indent=4)

        
